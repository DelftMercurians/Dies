//! `ReflexReceive` — a one-timer: intercept a pass with the firmware reflex kick
//! pre-armed and the robot facing the shot target, so the kicker fires the instant
//! the ball trips the breakbeam (zero handling latency).
//!
//! Positioning reuses the pass-line interception solve from [`super::receive`]
//! ([`solve_intercept`]); the only skill-specific logic is facing the shot target
//! (not the passer), holding the reflex armed, and the departure / no-arrival
//! terminal conditions. All timing is world-clock based (no `Instant`) so it is
//! deterministic under faster-than-real sim (see CLAUDE.md).

use dies_core::{Angle, DebugColor, Vector2};
use dies_strategy_protocol::{SkillCommand, SkillStatus};
use dies_tunables_macro::tunables;

use super::receive::solve_intercept;
use crate::control::skill_executor::{ExecutableSkill, SkillContext, SkillProgress};
use crate::control::{KickerControlInput, PlayerControlInput};

tunables! {
    section "ReflexReceive";

    /// Full-power kick speed armed on the reflex while waiting for the pass. (The
    /// simulator uses a fixed kicker strength and ignores this; it matters on
    /// hardware.)
    #[tunable(unit = "mm/s", min = 1000.0, max = 6500.0, step = 100.0)]
    REFLEX_KICK_SPEED: f64 = 4000.0;
    /// Dribbler speed while waiting — a little forward roll helps seat a marginal
    /// ball against the kicker plate so the breakbeam trips.
    #[tunable(min = 0.0, max = 1.0, step = 0.05)]
    DRIBBLER_SPEED: f64 = 0.4;
    /// Robot→ball distance below which the ball is "at the mouth": arms the
    /// departure detector (we only call success for a ball that actually arrived).
    #[tunable(unit = "mm", min = 100.0, max = 400.0, step = 10.0)]
    NEAR_DIST: f64 = 200.0;
    /// Outgoing ball speed (component away from the robot) that confirms the reflex
    /// fired the ball onward.
    #[tunable(unit = "mm/s", min = 200.0, max = 3000.0, step = 100.0)]
    DEPART_SPEED: f64 = 1000.0;
    /// Robot→ball distance beyond which a fast, outgoing ball counts as departed.
    #[tunable(unit = "mm", min = 120.0, max = 600.0, step = 10.0)]
    DEPART_DIST: f64 = 250.0;
    /// No-arrival timeout: if no ball ever reaches the mouth this long after the
    /// skill starts, disarm (by ending the skill) and fail so the strategy re-plans.
    #[tunable(unit = "s", min = 0.5, max = 6.0, step = 0.5)]
    ARRIVE_TIMEOUT: f64 = 3.0;
}

fn dkey(ctx: &SkillContext<'_>, tag: &str) -> String {
    format!("p{}.rr.{}", ctx.player.id.as_u32(), tag)
}

#[derive(Clone)]
pub struct ReflexReceiveSkill {
    from_pos: Vector2,
    intercept_pos: Vector2,
    target: Vector2,
    capture_limit: f64,
    status: SkillStatus,
    /// World time (`t_received`, s) of the first tick; drives the no-arrival timeout.
    first_tick: Option<f64>,
    /// Whether the ball ever reached the mouth (gates the departure success edge, so
    /// a ball whizzing past at range never counts as a fired one-timer).
    was_near: bool,
}

impl ReflexReceiveSkill {
    pub fn new(
        from_pos: Vector2,
        intercept_pos: Vector2,
        target: Vector2,
        capture_limit: f64,
    ) -> Self {
        Self {
            from_pos,
            intercept_pos,
            target,
            capture_limit,
            status: SkillStatus::Running,
            first_tick: None,
            was_near: false,
        }
    }

    fn fail(&mut self) -> SkillProgress {
        self.status = SkillStatus::Failed;
        SkillProgress::failure()
    }
}

impl ExecutableSkill for ReflexReceiveSkill {
    fn matches_command(&self, command: &SkillCommand) -> bool {
        matches!(command, SkillCommand::ReflexReceive { .. })
    }

    fn update_params(&mut self, command: &SkillCommand) {
        if let SkillCommand::ReflexReceive {
            from_pos,
            intercept_pos,
            target,
            capture_limit,
        } = command
        {
            self.from_pos = *from_pos;
            self.intercept_pos = *intercept_pos;
            self.target = *target;
            self.capture_limit = *capture_limit;
        }
    }

    fn tick(&mut self, ctx: SkillContext<'_>) -> SkillProgress {
        let now = ctx.world.t_received;
        let first = *self.first_tick.get_or_insert(now);
        let player_pos = ctx.player.position;

        let mut input = PlayerControlInput::new();
        // Face the shot target so the reflex fires the ball toward it.
        input.with_yaw(Angle::from_vector(self.target - player_pos));
        input.with_dribbling(DRIBBLER_SPEED());
        // Hold the reflex armed at full power — a held/level signal that keeps the
        // capacitor charged and the firmware armed (packet-loss safe). Ending the
        // skill (success/fail) stops this stream, so the controller disarms; no
        // explicit disarm is needed.
        input.with_kicker(KickerControlInput::ReflexKick);
        input.kick_speed = Some(REFLEX_KICK_SPEED());

        // Slide along the pass line to meet the incoming ball (shared with Receive).
        let ball = ctx
            .world
            .ball
            .as_ref()
            .map(|b| (b.position.xy(), b.velocity.xy()));
        let ix = solve_intercept(self.from_pos, self.intercept_pos, self.capture_limit, ball);
        input.with_position(ix.position);

        let tc = ctx.team_context;
        tc.debug_string(dkey(&ctx, "stage"), "armed");
        tc.debug_value(dkey(&ctx, "armed_ms"), (now - first) * 1000.0);
        tc.debug_line_colored(dkey(&ctx, "shot"), player_pos, self.target, DebugColor::Red);

        if let Some((ball_pos, ball_vel)) = ball {
            let to_ball = ball_pos - player_pos;
            let dist = to_ball.norm();
            tc.debug_value(dkey(&ctx, "ball_dist"), dist);
            if dist < NEAR_DIST() {
                self.was_near = true;
            }
            // Departure: the ball reached our mouth and is now speeding away → the
            // reflex fired it onward.
            let away = to_ball
                .try_normalize(1e-6)
                .map(|u| ball_vel.dot(&u))
                .unwrap_or(0.0);
            tc.debug_value(dkey(&ctx, "depart_speed"), away);
            if self.was_near && dist > DEPART_DIST() && away > DEPART_SPEED() {
                self.status = SkillStatus::Succeeded;
                return SkillProgress::success();
            }
        }

        if now - first > ARRIVE_TIMEOUT() {
            log::warn!("reflex_receive: no ball arrived, disarming");
            return self.fail();
        }

        self.status = SkillStatus::Running;
        SkillProgress::Continue(input)
    }

    fn status(&self) -> SkillStatus {
        self.status
    }

    fn skill_type(&self) -> &'static str {
        "ReflexReceive"
    }

    fn is_oneshot(&self) -> bool {
        true
    }

    fn description(&self) -> String {
        format!(
            "one-timer from ({:.0}, {:.0}) → ({:.0}, {:.0})",
            self.from_pos.x, self.from_pos.y, self.target.x, self.target.y
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::test_support::{player, team_ctx, world};
    use crate::control::{ObstacleSet, TeamContext};
    use dies_core::{PlayerData, TeamData, Vector3};

    const FROM: Vector2 = Vector2::new(0.0, 0.0);
    const INTERCEPT: Vector2 = Vector2::new(2000.0, 0.0);
    const TARGET: Vector2 = Vector2::new(4000.0, 0.0);

    fn skill() -> ReflexReceiveSkill {
        ReflexReceiveSkill::new(FROM, INTERCEPT, TARGET, 500.0)
    }

    fn ctx<'a>(w: &'a TeamData, tc: &'a TeamContext, p: &'a PlayerData) -> SkillContext<'a> {
        SkillContext {
            player: p,
            world: w,
            team_context: tc,
            debug_prefix: tc.key(format!("p{}", p.id)),
            obstacles: ObstacleSet::default(),
        }
    }

    /// Build a world with the ball at `ball_pos`, velocity `ball_vel`, at time `t`.
    fn world_at(p: &PlayerData, ball_pos: Vector2, ball_vel: Vector2, t: f64) -> TeamData {
        let mut w = world(vec![p.clone()], Some(ball_pos), 0.016);
        w.t_received = t;
        if let Some(b) = w.ball.as_mut() {
            b.velocity = Vector3::new(ball_vel.x, ball_vel.y, 0.0);
        }
        w
    }

    #[test]
    fn arms_reflex_at_full_power_while_waiting() {
        let mut s = skill();
        let p = player(0, INTERCEPT, 0.0);
        let w = world_at(&p, FROM, Vector2::zeros(), 0.0);
        let tc = team_ctx();
        match s.tick(ctx(&w, &tc, &p)) {
            SkillProgress::Continue(input) => {
                assert_eq!(input.kicker, KickerControlInput::ReflexKick);
                assert_eq!(input.kick_speed, Some(REFLEX_KICK_SPEED()));
                // Faces the shot target (+x from the intercept point).
                assert!(input.yaw.unwrap().radians().abs() < 1e-6);
            }
            _ => panic!("expected Continue while waiting for the pass"),
        }
        assert_eq!(s.status(), SkillStatus::Running);
    }

    #[test]
    fn fails_on_timeout_when_no_ball_arrives() {
        let mut s = skill();
        let p = player(0, INTERCEPT, 0.0);
        let tc = team_ctx();
        // Frame 1 at t=0 sets first_tick; ball idle far up the line (never near).
        let w0 = world_at(&p, FROM, Vector2::zeros(), 0.0);
        assert!(matches!(
            s.tick(ctx(&w0, &tc, &p)),
            SkillProgress::Continue(_)
        ));
        // Frame 2 well past ARRIVE_TIMEOUT → disarm + fail.
        let w1 = world_at(&p, FROM, Vector2::zeros(), ARRIVE_TIMEOUT() + 1.0);
        assert!(matches!(s.tick(ctx(&w1, &tc, &p)), SkillProgress::Done(_)));
        assert_eq!(s.status(), SkillStatus::Failed);
    }

    #[test]
    fn succeeds_when_ball_departs_after_arriving() {
        let mut s = skill();
        let p = player(0, INTERCEPT, 0.0);
        let tc = team_ctx();
        // Frame 1: ball at the mouth (within NEAR_DIST) → arms the departure edge.
        let near = INTERCEPT + Vector2::new(50.0, 0.0);
        let w0 = world_at(&p, near, Vector2::new(200.0, 0.0), 0.0);
        assert!(matches!(
            s.tick(ctx(&w0, &tc, &p)),
            SkillProgress::Continue(_)
        ));
        // Frame 2: ball fired onward — far and fast, moving away from the robot.
        let gone = INTERCEPT + Vector2::new(400.0, 0.0);
        let w1 = world_at(&p, gone, Vector2::new(3000.0, 0.0), 0.05);
        assert!(matches!(s.tick(ctx(&w1, &tc, &p)), SkillProgress::Done(_)));
        assert_eq!(s.status(), SkillStatus::Succeeded);
    }
}
