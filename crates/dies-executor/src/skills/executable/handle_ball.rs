//! HandleBall — unified acquire + carry + terminal-action ball handling.
//!
//! The merged successor to the `PickupBall` → `DribbleShoot` sequence. Because
//! acquire *and* the terminal action are one [`SkillCommand`] **variant**, the
//! [`SkillExecutor`](crate::control::skill_executor::SkillExecutor) updates this
//! skill's params in place when the strategy swaps the action (e.g. once the ball
//! is secured) rather than tearing it down and rebuilding — so the delicate
//! acquire→act transition no longer discards capture-phase state.
//!
//! The capture front-end (`select_approach_dir`, staging/commit) is shared with
//! [`PickupBallSkill`](super::pickup_ball::PickupBallSkill); the launch-point and
//! orbit/verify logic is shared with
//! [`DribbleShootSkill`](super::dribble_shoot::DribbleShootSkill). Constants live
//! in those modules so there is one tuning site.
//!
//! ## Terminal contract (keeps the seam removal sound)
//! - `Hold`/`Carry` **never** return `Done` — they run forever (the caller decides
//!   arrival). Every in-possession action swap is therefore a live param update.
//! - A kick (`Shoot`/`Strike`) is the only `Done(Success)`.
//! - Internal capture completion (breakbeam) is an internal stage edge, not a
//!   `Done` — that is what makes the acquire→act seam disappear.
//!
//! ## Silent re-acquire
//! Losing the ball mid-aim/carry returns to `Acquire` (debounced) instead of
//! failing up to the strategy. Bounded by [`MAX_REACQUIRE`] and the acquire/aim
//! backstops so a persistent loss still surfaces as `Done(Failure)` and the
//! planner can re-elect a capturer. `Strike` never re-acquires (double-touch safe).

use dies_core::{Angle, Vector2};
use dies_strategy_protocol::{BallAction, SkillCommand, SkillStatus};

use super::dribble_shoot::{self as ds, choose_launch, lane_blocked};
use super::pickup_ball::{self as pickup, clamp_into_field, select_approach_dir};
use crate::control::skill_executor::{ExecutableSkill, SkillContext, SkillProgress};
use crate::control::{KickerControlInput, PlayerControlInput, Velocity};

/// Debounce (s) before a ball loss during an act stage triggers a re-acquire — a
/// pass-style ride on the dribbler edge routinely drops the breakbeam for a frame
/// or two. Mirrors the pass coordinator's `SETUP_BALL_LOST_GRACE`.
const BALL_LOST_GRACE: f64 = 0.2;
/// Backstop (s): if we have *never* secured the ball this long after the first
/// tick, give up so the planner can re-elect a capturer.
const ACQUIRE_BACKSTOP: f64 = 6.0;
/// Backstop (s) for the aim stage once we hold the ball (a permanently blocked
/// lane, say). In concerto the driver's per-action timeout fires well before this.
const AIM_BACKSTOP: f64 = 6.0;
/// Max ball losses before the skill gives up instead of re-acquiring forever.
const MAX_REACQUIRE: u32 = 6;
/// Dribbler speed while carrying the ball to a `Carry` target (matches `Dribble`).
const CARRY_DRIBBLER_SPEED: f64 = 0.5;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    /// No possession yet — run the capture front-end.
    Acquire,
    /// Dribble the held ball to a `Carry` target.
    Carry,
    /// Orbit the held ball to the shot axis (`Shoot`).
    Aim,
    /// Kick commanded this tick.
    Kicking,
    /// Waiting to confirm the ball left the dribbler.
    Verifying,
    /// Possess + face a heading; never self-completes.
    Hold,
}

pub struct HandleBallSkill {
    action: BallAction,
    /// Exit-bias heading for the acquire sub-phase; `None` derives from `action`.
    approach: Option<Angle>,
    status: SkillStatus,
    stage: Stage,
    /// World time (`t_received`, s) of the first tick. Sim-clock based for
    /// deterministic faster-than-real sim (see CLAUDE.md).
    first_tick: Option<f64>,
    /// World time the current stage was entered (per-stage timers).
    stage_entered: f64,
    /// Whether possession has ever been confirmed (gates the acquire backstop).
    had_ball: bool,
    /// World time a mid-act ball loss began (re-acquire debounce); `None` when held.
    lost_since: Option<f64>,
    /// Ball losses so far this episode (bounds silent re-acquire).
    reacquires: u32,
    // ── capture state (shared with PickupBall) ──
    chosen_dir: Option<Vector2>,
    commit_pos: Option<Vector2>,
    commit_ball: Option<Vector2>,
    // ── strike (reflex) state ──
    kick_ball_pos: Option<Vector2>,
    armed_at: Option<f64>,
    // ── aim / verify state ──
    launch: Option<Vector2>,
    kick_time: Option<f64>,
}

impl HandleBallSkill {
    pub fn new(action: BallAction, approach: Option<Angle>) -> Self {
        Self {
            action,
            approach,
            status: SkillStatus::Running,
            stage: Stage::Acquire,
            first_tick: None,
            stage_entered: 0.0,
            had_ball: false,
            lost_since: None,
            reacquires: 0,
            chosen_dir: None,
            commit_pos: None,
            commit_ball: None,
            kick_ball_pos: None,
            armed_at: None,
            launch: None,
            kick_time: None,
        }
    }

    /// Reconfigure (used by the pass coordinator's Secure phase, which drives this
    /// skill directly rather than through the executor).
    pub fn reconfigure(&mut self, action: BallAction, approach: Option<Angle>) {
        self.action = action;
        self.approach = approach;
    }

    /// Exit-bias heading for the acquire sub-phase.
    fn acquire_heading(&self, ball_pos: Vector2) -> Angle {
        if let Some(a) = self.approach {
            return a;
        }
        match self.action {
            BallAction::Shoot { target } | BallAction::Strike { target } => {
                Angle::from_vector(target - ball_pos)
            }
            BallAction::Carry { heading, .. } | BallAction::Hold { heading } => heading,
        }
    }

    fn hold_heading(&self) -> Angle {
        match self.action {
            BallAction::Hold { heading } | BallAction::Carry { heading, .. } => heading,
            _ => Angle::from_radians(0.0),
        }
    }

    /// Whether the current act stage matches the (possibly just-swapped) action.
    fn stage_matches_action(&self) -> bool {
        matches!(
            (self.stage, self.action),
            (Stage::Hold, BallAction::Hold { .. })
                | (Stage::Carry, BallAction::Carry { .. })
                | (Stage::Aim, BallAction::Shoot { .. })
        )
    }

    /// Enter the act stage for the current action once the ball is held.
    fn enter_act(&mut self, now: f64) {
        self.stage = match self.action {
            BallAction::Carry { .. } => Stage::Carry,
            BallAction::Shoot { .. } => Stage::Aim,
            // Hold, and the unreachable Strike (handled before this point).
            _ => Stage::Hold,
        };
        self.stage_entered = now;
        self.launch = None;
        self.lost_since = None;
    }

    /// Reset to a fresh capture after a ball loss (bounded silent re-acquire).
    fn reacquire(&mut self, now: f64) {
        self.stage = Stage::Acquire;
        self.stage_entered = now;
        self.chosen_dir = None;
        self.commit_pos = None;
        self.commit_ball = None;
        self.launch = None;
        self.lost_since = None;
        self.reacquires += 1;
    }

    fn fail(&mut self) -> SkillProgress {
        self.status = SkillStatus::Failed;
        SkillProgress::failure()
    }

    /// `Strike`: a one-motion reflex strike-through. Never holds, never
    /// re-acquires — a whiff fails so the driver re-stages a clean approach.
    fn tick_strike(&mut self, ctx: &SkillContext<'_>, target: Vector2, now: f64) -> SkillProgress {
        let Some(ball) = ctx.world.ball.as_ref() else {
            return SkillProgress::Continue(PlayerControlInput::default());
        };
        let ball_pos = ball.position.xy();
        let player_pos = ctx.player.position;
        let heading = self
            .approach
            .unwrap_or_else(|| Angle::from_vector(target - ball_pos));
        let dir = heading.to_vector();

        let rel = player_pos - ball_pos;
        let along = rel.dot(&dir);
        let perp_vec = rel - dir * along;
        let perp = perp_vec.norm();

        let mut input = PlayerControlInput::new();
        input.with_yaw(heading);
        input.with_dribbling(pickup::DRIBBLER_SPEED);

        let committed =
            along < 0.0 && -along < pickup::COMMIT_DISTANCE && perp < pickup::COMMIT_PERP;
        if committed {
            input.avoid_ball = false;
            let gate = (1.0 - perp / pickup::GATE_PERP).clamp(0.0, 1.0);
            let speed = (-along) * pickup::APPROACH_GAIN + pickup::APPROACH_MIN_SPEED;
            input.add_global_velocity(dir * speed * gate - perp_vec * pickup::LATERAL_GAIN);

            input.with_kicker(KickerControlInput::ReflexKick);
            let kick_ball = *self.kick_ball_pos.get_or_insert(ball_pos);
            let armed_at = *self.armed_at.get_or_insert(now);
            let along_depart = (ball_pos - kick_ball).dot(&dir);
            if along_depart > pickup::KICK_DEPART_DIST || ball.velocity.norm() > pickup::KICK_DEPART_SPEED
            {
                self.status = SkillStatus::Succeeded;
                return SkillProgress::success();
            }
            if now - armed_at > pickup::REFLEX_TIMEOUT.as_secs_f64() {
                log::warn!("handle_ball: reflex strike did not connect");
                return self.fail();
            }
        } else {
            let stage = ball_pos - dir * pickup::APPROACH_DISTANCE;
            input.with_position(clamp_into_field(stage, ctx.world.field_geom.as_ref()));
            input.avoid_ball = true;
            input.avoid_ball_care = pickup::APPROACH_CARE;
        }

        self.status = SkillStatus::Running;
        SkillProgress::Continue(input)
    }

    /// Acquire (capture) front-end — shared with `PickupBallSkill`.
    fn drive_acquire(
        &mut self,
        ctx: &SkillContext<'_>,
        ball_pos: Vector2,
        player_pos: Vector2,
        now: f64,
    ) -> SkillProgress {
        let exit = self.acquire_heading(ball_pos);
        let dir = select_approach_dir(ctx, ball_pos, player_pos, exit, &mut self.chosen_dir);

        let rel = player_pos - ball_pos;
        let along = rel.dot(&dir);
        let perp_vec = rel - dir * along;
        let perp = perp_vec.norm();

        let mut input = PlayerControlInput::new();
        input.with_yaw(exit);
        input.with_dribbling(pickup::DRIBBLER_SPEED);

        let committed =
            along < 0.0 && -along < pickup::COMMIT_DISTANCE && perp < pickup::COMMIT_PERP;
        if committed {
            input.avoid_ball = false;
            let gate = (1.0 - perp / pickup::GATE_PERP).clamp(0.0, 1.0);
            let speed = (-along) * pickup::APPROACH_GAIN + pickup::APPROACH_MIN_SPEED;
            input.add_global_velocity(dir * speed * gate - perp_vec * pickup::LATERAL_GAIN);

            let commit_ball = *self.commit_ball.get_or_insert(ball_pos);
            let commit_pos = *self.commit_pos.get_or_insert(player_pos);
            if (ball_pos - commit_ball).norm() > pickup::BALL_MOVED_FAIL
                || (player_pos - commit_pos).norm() > pickup::DRIVEN_FAIL
            {
                // Ball squirted away during the commit drive — re-stage internally
                // (bounded by the re-acquire budget) instead of failing.
                self.reacquire(now);
            }
        } else {
            let stage = ball_pos - dir * pickup::APPROACH_DISTANCE;
            input.with_position(clamp_into_field(stage, ctx.world.field_geom.as_ref()));
            input.avoid_ball = true;
            input.avoid_ball_care = pickup::APPROACH_CARE;
            self.commit_pos = Some(player_pos);
            self.commit_ball = Some(ball_pos);
        }

        SkillProgress::Continue(input)
    }

    fn drive_hold(&mut self, heading: Angle) -> SkillProgress {
        let mut input = PlayerControlInput::new();
        input.with_dribbling(pickup::DRIBBLER_SPEED);
        input.with_yaw(heading);
        SkillProgress::Continue(input)
    }

    fn drive_carry(&mut self) -> SkillProgress {
        let (to, heading) = match self.action {
            BallAction::Carry { to, heading } => (to, heading),
            _ => return self.drive_hold(self.hold_heading()),
        };
        let mut input = PlayerControlInput::new();
        input.with_dribbling(CARRY_DRIBBLER_SPEED);
        input.with_position(to);
        input.with_yaw(heading);
        input.with_acceleration_limit(ds::CARRY_ACCEL_LIMIT);
        input.with_angular_speed_limit(ds::CARRY_ANGULAR_LIMIT);
        SkillProgress::Continue(input)
    }

    /// Aim (orbit) toward the shot target, repositioning first if the ball is
    /// jammed — shared launch/orbit logic with `DribbleShootSkill`.
    fn drive_aim(
        &mut self,
        ctx: &SkillContext<'_>,
        ball_pos: Vector2,
        player_pos: Vector2,
        now: f64,
    ) -> SkillProgress {
        let BallAction::Shoot { target } = self.action else {
            return self.drive_hold(self.hold_heading());
        };
        if now - self.stage_entered > AIM_BACKSTOP {
            log::warn!("handle_ball: aim timed out");
            return self.fail();
        }

        let launch = *self
            .launch
            .get_or_insert_with(|| choose_launch(ctx, ball_pos, target));

        let mut input = PlayerControlInput::new();
        input.with_dribbling(ds::DRIBBLER_SPEED);

        // Carry the ball to the launch point first if it's away from it.
        if (ball_pos - launch).norm() > ds::REPOSITION_ARRIVE {
            let axis = (target - launch)
                .try_normalize(1e-6)
                .unwrap_or_else(|| Vector2::new(1.0, 0.0));
            let pose = launch - axis * ds::BALL_TO_ROBOT_DISTANCE;
            input.with_position(pose);
            input.with_yaw(Angle::from_vector(target - launch));
            input.with_acceleration_limit(ds::CARRY_ACCEL_LIMIT);
            input.with_angular_speed_limit(ds::CARRY_ANGULAR_LIMIT);
            return SkillProgress::Continue(input);
        }

        // Orbit the ball to align the shot axis, then commit.
        input.avoid_robots = false;
        input.with_angular_speed_limit(1000.0);

        let to_target = target - ball_pos;
        let target_heading = Angle::from_vector(to_target);
        let blocked = lane_blocked(ctx, ball_pos, target_heading);
        let err = target_heading - ctx.player.yaw;
        if err.abs() < ds::YAW_TOLERANCE && !blocked && ctx.player.has_ball {
            self.stage = Stage::Kicking;
            self.stage_entered = now;
        }

        let r = player_pos - ball_pos;
        let r_hat = r
            .try_normalize(1e-6)
            .unwrap_or_else(|| Vector2::new(1.0, 0.0));
        let tangent = Vector2::new(-r_hat.y, r_hat.x); // CCW
        const RADIUS_KP: f64 = 2.0;
        let v_rad = -RADIUS_KP * (r.norm() - ds::BALL_TO_ROBOT_DISTANCE) * r_hat;
        let v_tan = if blocked {
            Vector2::zeros()
        } else {
            let speed = (ds::ORBIT_GAIN * err.abs()).clamp(ds::MIN_ORBIT_SPEED, ds::ORBIT_SPEED);
            err.signum() * speed * tangent
        };
        input.velocity = Velocity::global(v_tan + v_rad);
        input.with_yaw(Angle::from_vector(-r));
        SkillProgress::Continue(input)
    }

    fn drive_kick(&mut self, ball_pos: Vector2, now: f64) -> SkillProgress {
        let BallAction::Shoot { target } = self.action else {
            return self.fail();
        };
        let target_heading = Angle::from_vector(target - ball_pos);
        let mut input = PlayerControlInput::new();
        input.with_yaw(target_heading);
        input.with_kicker(KickerControlInput::Kick);
        input.kick_speed = Some(ds::KICK_SPEED);
        input.with_dribbling(0.0);
        self.kick_ball_pos = Some(ball_pos);
        self.kick_time = Some(now);
        self.stage = Stage::Verifying;
        self.stage_entered = now;
        SkillProgress::Continue(input)
    }

    fn drive_verify(
        &mut self,
        ball_pos: Vector2,
        ball_vel_norm: f64,
        now: f64,
    ) -> SkillProgress {
        let departed = self
            .kick_ball_pos
            .map(|p0| (ball_pos - p0).norm() > ds::KICK_DEPART_DIST)
            .unwrap_or(false)
            || ball_vel_norm > ds::KICK_DEPART_SPEED;
        if departed {
            self.status = SkillStatus::Succeeded;
            return SkillProgress::success();
        }
        if self
            .kick_time
            .map(|t| now - t > ds::VERIFY_WINDOW.as_secs_f64())
            .unwrap_or(true)
        {
            log::warn!("handle_ball: kick did not connect");
            return self.fail();
        }
        let mut input = PlayerControlInput::new();
        if let BallAction::Shoot { target } = self.action {
            input.with_yaw(Angle::from_vector(target - ball_pos));
        }
        input.with_dribbling(0.0);
        SkillProgress::Continue(input)
    }
}

impl ExecutableSkill for HandleBallSkill {
    fn matches_command(&self, command: &SkillCommand) -> bool {
        matches!(command, SkillCommand::HandleBall { .. })
    }

    fn update_params(&mut self, command: &SkillCommand) {
        if let SkillCommand::HandleBall { action, approach } = command {
            self.action = *action;
            self.approach = *approach;
        }
    }

    fn tick(&mut self, ctx: SkillContext<'_>) -> SkillProgress {
        let now = ctx.world.t_received;
        let first = *self.first_tick.get_or_insert(now);
        let has_ball = ctx.player.has_ball;
        if has_ball {
            self.had_ball = true;
        }
        let player_pos = ctx.player.position;

        // Strike is a one-motion reflex strike-through: handled separately (never
        // holds, never re-acquires).
        if let BallAction::Strike { target } = self.action {
            return self.tick_strike(&ctx, target, now);
        }

        // Acquisition / thrash backstops so a persistent loss surfaces as a
        // failure and the planner can re-elect a capturer.
        if !self.had_ball && now - first > ACQUIRE_BACKSTOP {
            log::warn!("handle_ball: could not acquire the ball");
            return self.fail();
        }
        if self.reacquires > MAX_REACQUIRE {
            log::warn!("handle_ball: lost the ball too many times");
            return self.fail();
        }

        let Some(ball) = ctx.world.ball.as_ref() else {
            // No ball observation: hold if we have it, otherwise idle.
            let mut input = PlayerControlInput::new();
            if has_ball {
                input.with_dribbling(pickup::DRIBBLER_SPEED);
            }
            self.status = SkillStatus::Running;
            return SkillProgress::Continue(input);
        };
        let ball_pos = ball.position.xy();
        let ball_vel_norm = ball.velocity.norm();

        // Debounced silent re-acquire if the ball is lost during an act stage.
        if matches!(self.stage, Stage::Aim | Stage::Carry | Stage::Hold) {
            let far = (player_pos - ball_pos).norm() > ds::LOST_BALL_DISTANCE;
            if !has_ball && far {
                let lost = *self.lost_since.get_or_insert(now);
                if now - lost > BALL_LOST_GRACE {
                    self.reacquire(now);
                }
            } else {
                self.lost_since = None;
            }
        }

        // Route: acquire until we hold the ball, then enter / re-route the act stage.
        if matches!(self.stage, Stage::Acquire) {
            if has_ball {
                self.enter_act(now);
            } else {
                self.status = SkillStatus::Running;
                return self.drive_acquire(&ctx, ball_pos, player_pos, now);
            }
        } else if matches!(self.stage, Stage::Aim | Stage::Carry | Stage::Hold)
            && !self.stage_matches_action()
        {
            // Action swapped live (e.g. Hold -> Shoot) — re-route. Kicking/Verifying
            // are committed and never re-routed.
            self.enter_act(now);
        }

        self.status = SkillStatus::Running;
        match self.stage {
            Stage::Hold => self.drive_hold(self.hold_heading()),
            Stage::Carry => self.drive_carry(),
            Stage::Aim => self.drive_aim(&ctx, ball_pos, player_pos, now),
            Stage::Kicking => self.drive_kick(ball_pos, now),
            Stage::Verifying => self.drive_verify(ball_pos, ball_vel_norm, now),
            Stage::Acquire => unreachable!("acquire handled above"),
        }
    }

    fn status(&self) -> SkillStatus {
        self.status
    }

    fn skill_type(&self) -> &'static str {
        "HandleBall"
    }

    fn is_oneshot(&self) -> bool {
        true
    }

    fn description(&self) -> String {
        let act = match self.action {
            BallAction::Shoot { .. } => "shoot",
            BallAction::Strike { .. } => "strike",
            BallAction::Carry { .. } => "carry",
            BallAction::Hold { .. } => "hold",
        };
        let stage = match self.stage {
            Stage::Acquire => "acquiring",
            Stage::Carry => "carrying",
            Stage::Aim => "aiming",
            Stage::Kicking => "kicking",
            Stage::Verifying => "verifying",
            Stage::Hold => "holding",
        };
        format!("{act}: {stage}")
    }
}
