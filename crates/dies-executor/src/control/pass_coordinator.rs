//! Joint pass coordinator.
//!
//! A [`PassCoordinator`] owns TWO robots (a passer and a receiver) through a
//! single state machine, ticked **once per frame**. It is the executor-side
//! realization of the atomic `ctx.pass(passer, receiver)` action.
//!
//! The whole point is to make the passer/receiver timing handshake *explicit*:
//! the kick is only committed once the passer is aligned AND the receiver is in
//! position (the `Commit` barrier). Everything before the kick is freely
//! abortable at zero cost; the kick is the only irreversible moment.
//!
//! Robustness here means **clean release + a typed verdict**, never recovery. On
//! any terminal outcome the coordinator stops driving both robots and reports a
//! [`PassResult`]; the joint executor then releases them back to the strategy.
//!
//! Sub-behaviours are composed from the existing single-robot skills rather than
//! reimplemented: [`HandleBallSkill`] (in `Hold` mode) for `Secure`,
//! [`ReceiveSkill`] for the receiver's positioning + interception.

use dies_core::{Angle, PlayerData, PlayerId, TeamData, Vector2, BALL_RADIUS, PLAYER_RADIUS};
use dies_strategy_protocol::{BallAction, PassBallState, PassFailure, PassResult, SkillStatus};

use super::avoidance::ObstacleSet;
use super::skill_executor::{ExecutableSkill, SkillContext, SkillProgress};
use super::team_context::TeamContext;
use crate::control::{KickerControlInput, PlayerControlInput, Velocity};
use crate::skills::executable::{HandleBallSkill, ReceiveSkill};

/// Max distance the ball may be from the passer for the weak `Secure` phase to
/// attempt a pickup. Beyond this the pass fails immediately with `BallLost` — the
/// pass never chases a loose ball.
const SECURE_DISTANCE: f64 = 600.0;
/// Heading error (rad) below which the passer is considered aligned to kick.
const ALIGNMENT_TOLERANCE: f64 = 10.0 * std::f64::consts::PI / 180.0;
/// Time budget for the `Secure` phase.
const SECURE_TIMEOUT: f64 = 3.0;
/// Time budget for the `Setup` phase (aim + receiver positioning).
const SETUP_TIMEOUT: f64 = 4.0;
/// Dribble speed (0..1) the passer holds from `Secure` through `Setup`. The sim
/// (and firmware) grip is binary — any non-zero speed pins the ball — but we
/// hold full power so the ball stays seated on the dribbler while the passer
/// rotates to aim, instead of slipping out of the dribbler cone.
const PASS_DRIBBLE_SPEED: f64 = 1.0;
/// Grace period (s) the passer may transiently read `!has_ball` during `Setup`
/// before the pass aborts with `BallLost`. A pass that rotates the ball toward
/// the receiver routinely drops the breakbeam for a frame or two as the ball
/// rides the edge of the dribbler cone; without this debounce that single-frame
/// flicker kills the pass and the passer livelocks on recapture→retry.
const SETUP_BALL_LOST_GRACE: f64 = 0.2;
/// How close (mm) the receiver must be to the intercept point to be "ready".
const RECEIVER_READY_DIST: f64 = 250.0;
/// Max perpendicular distance (mm) the receiver will travel off the pass line.
const CAPTURE_LIMIT: f64 = 1500.0;
/// Ball speed (mm/s) below which, if not at the receiver, the ball is considered
/// to have stopped short.
const STOPPED_SHORT_SPEED: f64 = 150.0;
/// Rough estimate of pass ball speed (mm/s), used only for the flight timeout.
const PASS_SPEED_ESTIMATE: f64 = 2500.0;
/// Multiplicative + additive margin on the expected flight time before timing out.
const FLIGHT_TIMEOUT_MARGIN: f64 = 2.5;
const FLIGHT_TIMEOUT_BASE: f64 = 0.6;
/// Distance (mm) within which the ball counts as "at the receiver" for the
/// stopped-short check.
const NEAR_RECEIVER_DIST: f64 = 200.0;
/// Distance (mm) at which an opponent is considered to control the ball.
const OPPONENT_POSSESSION_DIST: f64 = PLAYER_RADIUS + BALL_RADIUS + 30.0;
/// Fraction of the pass-line length past the intercept point beyond which an
/// uncaught ball is declared missed.
const OVERSHOOT_FRACTION: f64 = 1.2;

/// Map pass distance to a kick speed.
///
/// TODO(calibration): purely heuristic — no empirical distance→power mapping
/// exists yet. Tune against the simulator / real robots.
fn pass_kick_speed(distance: f64) -> f64 {
    (1500.0 + distance * 0.8).clamp(1500.0, 5000.0)
}

/// The phase of a pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassPhase {
    /// Passer confirms possession (weak: only picks up if already close).
    Secure,
    /// Passer aims; receiver moves to the intercept-ready pose. Concurrent.
    Setup,
    /// Barrier passed — issue the kick. The only irreversible state.
    Commit,
    /// Ball in flight; receiver tracks the actual ball and intercepts.
    Flight,
    /// Possession confirmed at the receiver.
    Settle,
    /// Terminal — see `result`.
    Done,
}

impl PassPhase {
    pub fn as_str(&self) -> &'static str {
        match self {
            PassPhase::Secure => "Secure",
            PassPhase::Setup => "Setup",
            PassPhase::Commit => "Commit",
            PassPhase::Flight => "Flight",
            PassPhase::Settle => "Settle",
            PassPhase::Done => "Done",
        }
    }

    pub fn index(&self) -> f64 {
        match self {
            PassPhase::Secure => 0.0,
            PassPhase::Setup => 1.0,
            PassPhase::Commit => 2.0,
            PassPhase::Flight => 3.0,
            PassPhase::Settle => 4.0,
            PassPhase::Done => 5.0,
        }
    }
}

/// Context passed to a [`PassCoordinator`] tick — the world plus the team's debug
/// context (for namespaced visualization).
pub struct PassContext<'a> {
    pub world: &'a TeamData,
    pub team_context: &'a TeamContext,
}

/// The per-frame output of a [`PassCoordinator`]: control for both robots plus
/// the joint status and (on the terminal frame) the rich result.
pub struct PassTickOutput {
    pub passer_input: PlayerControlInput,
    pub receiver_input: PlayerControlInput,
    pub status: SkillStatus,
    /// Present only on the frame the pass terminates.
    pub result: Option<PassResult>,
}

/// Joint coordinator for a single pass between two robots.
pub struct PassCoordinator {
    passer: PlayerId,
    receiver: PlayerId,
    target_hint: Option<Vector2>,

    phase: PassPhase,
    phase_elapsed: f64,
    total_elapsed: f64,
    /// Time spent reading `!has_ball` in the current `Setup` phase. Reset on any
    /// phase change and whenever possession is (re)confirmed; debounces the
    /// breakbeam so a transient drop doesn't abort the pass.
    setup_ball_lost: f64,

    /// Fixed intercept point, captured when entering `Setup`.
    intercept_point: Option<Vector2>,
    /// Sub-skill driving the passer during `Secure` (unified capture front-end,
    /// in `Hold` mode — acquire the ball and hold it facing the receiver).
    acquire: HandleBallSkill,
    /// Sub-skill driving the receiver from `Setup` onward.
    receive: ReceiveSkill,

    /// Terminal result, set once when `phase == Done`.
    result: Option<PassResult>,
}

impl PassCoordinator {
    pub fn new(passer: PlayerId, receiver: PlayerId, target_hint: Option<Vector2>) -> Self {
        Self {
            passer,
            receiver,
            target_hint,
            phase: PassPhase::Secure,
            phase_elapsed: 0.0,
            total_elapsed: 0.0,
            setup_ball_lost: 0.0,
            intercept_point: None,
            // Sub-skills are seeded with placeholder params and reconfigured on use.
            acquire: HandleBallSkill::new(
                BallAction::Hold {
                    heading: Angle::from_radians(0.0),
                },
                None,
                true,
            ),
            receive: ReceiveSkill::new(Vector2::zeros(), Vector2::zeros(), CAPTURE_LIMIT, false),
            result: None,
        }
    }

    pub fn passer(&self) -> PlayerId {
        self.passer
    }

    pub fn receiver(&self) -> PlayerId {
        self.receiver
    }

    pub fn phase(&self) -> PassPhase {
        self.phase
    }

    pub fn is_done(&self) -> bool {
        self.phase == PassPhase::Done
    }

    /// The terminal result, if the pass has ended.
    pub fn result(&self) -> Option<PassResult> {
        self.result
    }

    /// Update the optional target hint (param update from the strategy).
    pub fn update_target_hint(&mut self, target_hint: Option<Vector2>) {
        self.target_hint = target_hint;
    }

    /// Force a clean terminal failure (used by the joint executor for
    /// `PartnerLeft` / `Cancelled`). Idempotent.
    pub fn force_terminate(&mut self, reason: PassFailure, ball_state: PassBallState) {
        if self.phase != PassPhase::Done {
            self.phase = PassPhase::Done;
            self.result = Some(PassResult::Failure { reason, ball_state });
        }
    }

    fn enter(&mut self, phase: PassPhase) {
        self.phase = phase;
        self.phase_elapsed = 0.0;
        self.setup_ball_lost = 0.0;
    }

    fn finish(&mut self, result: PassResult) -> PassResult {
        self.phase = PassPhase::Done;
        self.result = Some(result);
        result
    }

    fn player<'a>(&self, world: &'a TeamData, id: PlayerId) -> Option<&'a PlayerData> {
        world.own_players.iter().find(|p| p.id == id)
    }

    /// Build a `SkillContext` for one of our robots so we can tick a sub-skill.
    fn skill_ctx<'a>(&self, ctx: &'a PassContext<'a>, player: &'a PlayerData) -> SkillContext<'a> {
        SkillContext {
            player,
            world: ctx.world,
            team_context: ctx.team_context,
            debug_prefix: ctx.team_context.key(format!("p{}", player.id)),
            // The pass FSM's secure-pickup happens with the ball already at the
            // passer; obstacle-aware approach selection isn't needed here.
            obstacles: ObstacleSet::default(),
        }
    }

    /// Advance the joint FSM one frame.
    pub fn tick(&mut self, ctx: &PassContext<'_>) -> PassTickOutput {
        let dt = ctx.world.dt;
        self.phase_elapsed += dt;
        self.total_elapsed += dt;

        // Already terminal — hold both robots, report the stored verdict.
        if self.phase == PassPhase::Done {
            return self.terminal_output();
        }

        // Both robots must be present to do anything meaningful.
        let (Some(passer), Some(receiver)) = (
            self.player(ctx.world, self.passer),
            self.player(ctx.world, self.receiver),
        ) else {
            // Robot dropped out of vision — hold and let the timeout guard catch
            // a persistent loss in the active phase.
            return PassTickOutput {
                passer_input: PlayerControlInput::default(),
                receiver_input: PlayerControlInput::default(),
                status: SkillStatus::Running,
                result: None,
            };
        };
        let passer = passer.clone();
        let receiver = receiver.clone();

        let ball = ctx.world.ball.clone();
        let ball_pos = ball.as_ref().map(|b| b.position.xy());
        let ball_vel = ball.as_ref().map(|b| b.velocity.xy());

        // Resolve / freeze the intercept point once we leave Secure.
        let target_point = self
            .intercept_point
            .or(self.target_hint)
            .unwrap_or(receiver.position);

        let mut result: Option<PassResult> = None;
        let mut passer_input = PlayerControlInput::default();
        let mut receiver_input = PlayerControlInput::default();

        match self.phase {
            PassPhase::Secure => {
                // Already have it? proceed.
                if passer.has_ball {
                    self.enter(PassPhase::Setup);
                    self.intercept_point = Some(self.target_hint.unwrap_or(receiver.position));
                    // Fall through to Setup next frame; hold position this frame
                    // but keep the dribbler engaged so the ball is not released
                    // on the Secure→Setup handoff.
                    passer_input = hold_dribbling();
                } else if let Some(bp) = ball_pos {
                    let dist = (bp - passer.position).norm();
                    if dist > SECURE_DISTANCE {
                        result = Some(self.finish(PassResult::Failure {
                            reason: PassFailure::BallLost,
                            ball_state: PassBallState::Loose { position: bp },
                        }));
                    } else if self.phase_elapsed > SECURE_TIMEOUT {
                        result = Some(self.finish(PassResult::Failure {
                            reason: PassFailure::BallLost,
                            ball_state: PassBallState::Loose { position: bp },
                        }));
                    } else {
                        // Run the unified capture front-end in Hold mode, holding
                        // the post-capture heading toward the receiver. Hold never
                        // returns Done; the Done arm is a defensive fallback.
                        let heading = Angle::between_points(bp, receiver.position);
                        self.acquire
                            .reconfigure(BallAction::Hold { heading }, Some(heading));
                        let sctx = self.skill_ctx(ctx, &passer);
                        match self.acquire.tick(sctx) {
                            SkillProgress::Continue(input) => passer_input = input,
                            SkillProgress::Done(_) => {
                                // Possession (or give-up) — re-checked next frame.
                                // Keep dribbling so a fresh grip isn't dropped.
                                passer_input = hold_dribbling();
                            }
                        }
                    }
                } else {
                    // No ball visible during secure — treat as lost.
                    result = Some(self.finish(PassResult::Failure {
                        reason: PassFailure::BallLost,
                        ball_state: PassBallState::Unknown,
                    }));
                }
            }

            PassPhase::Setup => {
                // Debounce possession: rotating the ball toward the receiver
                // rides it along the edge of the dribbler cone, so the breakbeam
                // routinely drops for a frame or two. Only abort once the ball
                // has been lost for longer than the grace window; while
                // transiently lost we keep aiming + dribbling so it re-seats.
                if passer.has_ball {
                    self.setup_ball_lost = 0.0;
                } else {
                    self.setup_ball_lost += dt;
                }

                if self.setup_ball_lost > SETUP_BALL_LOST_GRACE {
                    result = Some(
                        self.finish(PassResult::Failure {
                            reason: PassFailure::BallLost,
                            ball_state: ball_pos
                                .map(|p| PassBallState::Loose { position: p })
                                .unwrap_or(PassBallState::Unknown),
                        }),
                    );
                } else if self.phase_elapsed > SETUP_TIMEOUT {
                    result = Some(self.finish(PassResult::Failure {
                        reason: PassFailure::Timeout,
                        ball_state: PassBallState::WithPasser,
                    }));
                } else {
                    // Passer aims at the intercept point.
                    let target_heading = Angle::between_points(passer.position, target_point);
                    passer_input = aim_input(&passer, ball_pos, target_heading);

                    // Receiver gets ready (positions on the line, faces the ball).
                    receiver_input = self.tick_receive(ctx, &receiver, target_point);

                    // Commit only from a confirmed grip — never kick on a frame
                    // where the breakbeam is transiently dropped.
                    let aligned = (passer.yaw - target_heading).abs() < ALIGNMENT_TOLERANCE;
                    let ready = (receiver.position - target_point).norm() < RECEIVER_READY_DIST;
                    if passer.has_ball && aligned && ready {
                        self.enter(PassPhase::Commit);
                    }
                }
            }

            PassPhase::Commit => {
                // Issue the kick. Irreversible.
                let target_heading = Angle::between_points(passer.position, target_point);
                let dist = (target_point - passer.position).norm();
                let mut input = PlayerControlInput::new();
                input.with_yaw(target_heading);
                input.with_dribbling(0.0);
                input.with_kicker(KickerControlInput::Kick);
                input.kick_speed = Some(pass_kick_speed(dist));
                passer_input = input;

                // Keep the receiver tracking through the kick.
                receiver_input = self.tick_receive(ctx, &receiver, target_point);

                self.enter(PassPhase::Flight);
            }

            PassPhase::Flight => {
                // Passer disengages to avoid a double touch.
                passer_input = PlayerControlInput::new();
                // Receiver keeps intercepting the actual ball.
                receiver_input = self.tick_receive(ctx, &receiver, target_point);

                if let Some(r) = self.check_flight_terminal(ctx, &receiver, target_point) {
                    result = Some(self.finish(r));
                }
            }

            PassPhase::Settle => {
                result = Some(self.finish(PassResult::Success {
                    receiver: self.receiver,
                }));
            }

            PassPhase::Done => unreachable!("handled above"),
        }

        self.emit_debug(ctx, &passer, &receiver, ball_pos, ball_vel, target_point);

        let status = match (&self.phase, &result) {
            (_, Some(PassResult::Success { .. })) => SkillStatus::Succeeded,
            (_, Some(PassResult::Failure { .. })) => SkillStatus::Failed,
            _ => SkillStatus::Running,
        };
        PassTickOutput {
            passer_input,
            receiver_input,
            status,
            result,
        }
    }

    /// Drive the receiver's [`ReceiveSkill`], updating its params each frame.
    fn tick_receive(
        &mut self,
        ctx: &PassContext<'_>,
        receiver: &PlayerData,
        target_point: Vector2,
    ) -> PlayerControlInput {
        let from = self
            .player(ctx.world, self.passer)
            .map(|p| p.position)
            .unwrap_or(receiver.position);
        self.receive
            .reconfigure(from, target_point, CAPTURE_LIMIT, true);
        let sctx = self.skill_ctx(ctx, receiver);
        match self.receive.tick(sctx) {
            SkillProgress::Continue(input) => input,
            // ReceiveSkill only "completes" on possession; the coordinator owns
            // the real success decision, so just hold on completion.
            SkillProgress::Done(_) => PlayerControlInput::new(),
        }
    }

    /// Evaluate the terminal conditions during `Flight`. Returns `Some` to end.
    fn check_flight_terminal(
        &mut self,
        ctx: &PassContext<'_>,
        receiver: &PlayerData,
        target_point: Vector2,
    ) -> Option<PassResult> {
        let Some(ball) = ctx.world.ball.as_ref() else {
            // Ball lost from vision during flight — can't track; time out via guard.
            if self.phase_elapsed > self.flight_timeout(target_point) {
                return Some(PassResult::Failure {
                    reason: PassFailure::Timeout,
                    ball_state: PassBallState::Unknown,
                });
            }
            return None;
        };
        let ball_pos = ball.position.xy();
        let ball_vel = ball.velocity.xy();

        // Success: the unified possession metric credits the receiver (its
        // proximity + breakbeam fusion subsumes the old vision fallback).
        if receiver.has_ball {
            return Some(PassResult::Success {
                receiver: self.receiver,
            });
        }

        // Opponent intercepted?
        let opp_has_ball = ctx
            .world
            .opp_players
            .iter()
            .any(|o| (o.position - ball_pos).norm() < OPPONENT_POSSESSION_DIST);
        if opp_has_ball {
            return Some(PassResult::Failure {
                reason: PassFailure::Intercepted,
                ball_state: PassBallState::WithOpponent,
            });
        }

        // Stopped short: slow and not near the receiver (after a brief grace).
        if self.phase_elapsed > 0.3
            && ball_vel.norm() < STOPPED_SHORT_SPEED
            && (ball_pos - receiver.position).norm() > NEAR_RECEIVER_DIST
        {
            return Some(PassResult::Failure {
                reason: PassFailure::StoppedShort,
                ball_state: PassBallState::Loose { position: ball_pos },
            });
        }

        // Receiver missed: ball overshot the intercept point along the line, or
        // strayed beyond the capture window.
        let from = self.intercept_from();
        let line = target_point - from;
        let len = line.norm();
        if len > 1.0 {
            let dir = line / len;
            let progress = (ball_pos - from).dot(&dir) / len;
            let perp = (ball_pos - from) - dir * (ball_pos - from).dot(&dir);
            if progress > OVERSHOOT_FRACTION || perp.norm() > CAPTURE_LIMIT {
                return Some(PassResult::Failure {
                    reason: PassFailure::ReceiverMissed,
                    ball_state: PassBallState::Loose { position: ball_pos },
                });
            }
        }

        // Flight timeout.
        if self.phase_elapsed > self.flight_timeout(target_point) {
            return Some(PassResult::Failure {
                reason: PassFailure::Timeout,
                ball_state: PassBallState::Loose { position: ball_pos },
            });
        }

        None
    }

    fn intercept_from(&self) -> Vector2 {
        // The passer position is frozen into the receive sub-skill; reuse the
        // intercept point's line origin via the receive skill's from_pos.
        self.receive.from_pos()
    }

    fn flight_timeout(&self, target_point: Vector2) -> f64 {
        let dist = (target_point - self.intercept_from()).norm();
        FLIGHT_TIMEOUT_BASE + FLIGHT_TIMEOUT_MARGIN * (dist / PASS_SPEED_ESTIMATE)
    }

    fn terminal_output(&self) -> PassTickOutput {
        let status = match self.result {
            Some(PassResult::Success { .. }) => SkillStatus::Succeeded,
            Some(PassResult::Failure { .. }) => SkillStatus::Failed,
            None => SkillStatus::Failed,
        };
        PassTickOutput {
            passer_input: PlayerControlInput::default(),
            receiver_input: PlayerControlInput::default(),
            status,
            // Only report the result once (on the terminating frame); the joint
            // executor retires the coordinator after that.
            result: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_debug(
        &self,
        ctx: &PassContext<'_>,
        passer: &PlayerData,
        receiver: &PlayerData,
        ball_pos: Option<Vector2>,
        _ball_vel: Option<Vector2>,
        target_point: Vector2,
    ) {
        let tc = ctx.team_context;
        let pair = format!("pass.{}-{}", self.passer, self.receiver);

        // --- Full internal-state readout (strings/numbers). ---
        tc.debug_string(format!("{pair}.phase"), self.phase.as_str());
        tc.debug_value(format!("{pair}.phase_index"), self.phase.index());
        tc.debug_value(format!("{pair}.phase_elapsed_s"), self.phase_elapsed);
        tc.debug_value(format!("{pair}.total_elapsed_s"), self.total_elapsed);

        let target_heading = Angle::between_points(passer.position, target_point);
        let heading_error = (passer.yaw - target_heading).abs();
        let pass_distance = (target_point - passer.position).norm();
        let receiver_dist = (receiver.position - target_point).norm();
        let aligned = heading_error < ALIGNMENT_TOLERANCE;
        let ready = receiver_dist < RECEIVER_READY_DIST;
        let has = passer.has_ball;

        tc.debug_string(format!("{pair}.passer_has_ball"), bool_str(has));
        tc.debug_string(format!("{pair}.passer_aligned"), bool_str(aligned));
        tc.debug_string(format!("{pair}.receiver_ready"), bool_str(ready));
        tc.debug_string(
            format!("{pair}.barrier_satisfied"),
            bool_str(aligned && ready),
        );
        tc.debug_value(format!("{pair}.heading_error_rad"), heading_error);
        tc.debug_value(format!("{pair}.pass_distance_mm"), pass_distance);
        tc.debug_value(format!("{pair}.receiver_dist_mm"), receiver_dist);
        tc.debug_value(format!("{pair}.kick_speed"), pass_kick_speed(pass_distance));
        if let Some(PassResult::Failure { reason, .. }) = &self.result {
            tc.debug_string(format!("{pair}.result"), format!("{reason:?}"));
        } else if let Some(PassResult::Success { .. }) = &self.result {
            tc.debug_string(format!("{pair}.result"), "Success");
        }

        // Mirror the key scalars under each player so they show in the inspector.
        for id in [self.passer, self.receiver] {
            tc.debug_string(format!("p{id}.pass.phase"), self.phase.as_str());
            tc.debug_string(format!("p{id}.pass.partner"), partner_str(self, id));
        }

        // --- Field overlays (absolute coords; red→green at the barrier). ---
        let barrier = aligned && ready;
        let line_color = if barrier {
            dies_core::DebugColor::Green
        } else {
            dies_core::DebugColor::Red
        };
        dies_core::debug_line(
            tc.key(format!("{pair}.line")),
            passer.position,
            target_point,
            line_color,
        );
        dies_core::debug_cross(
            tc.key(format!("{pair}.intercept")),
            target_point,
            dies_core::DebugColor::Purple,
        );
        dies_core::debug_circle_stroke(
            tc.key(format!("{pair}.capture")),
            target_point,
            CAPTURE_LIMIT,
            dies_core::DebugColor::Gray,
        );
        if self.phase == PassPhase::Secure {
            if let Some(bp) = ball_pos {
                dies_core::debug_circle_stroke(
                    tc.key(format!("{pair}.secure")),
                    bp,
                    SECURE_DISTANCE,
                    dies_core::DebugColor::Orange,
                );
            }
        } else {
            dies_core::debug_remove(tc.key(format!("{pair}.secure")));
        }
    }
}

/// Build the passer's aim input: face the target while creeping into the ball to
/// keep contact. Mirrors `ShootSkill`'s `Facing` behaviour.
fn aim_input(
    passer: &PlayerData,
    ball_pos: Option<Vector2>,
    target_heading: Angle,
) -> PlayerControlInput {
    let mut input = PlayerControlInput::new();
    input.with_dribbling(PASS_DRIBBLE_SPEED);
    input.with_yaw(target_heading);
    input.with_care(0.7);
    if let Some(bp) = ball_pos {
        let dir = (bp - passer.position).normalize();
        input.velocity = Velocity::global(dir * 40.0);
    }
    input
}

/// A neutral hold input that keeps the dribbler at full power — used on frames
/// where the passer should not move but must not release the ball.
fn hold_dribbling() -> PlayerControlInput {
    let mut input = PlayerControlInput::new();
    input.with_dribbling(PASS_DRIBBLE_SPEED);
    input
}

fn bool_str(b: bool) -> &'static str {
    if b {
        "true"
    } else {
        "false"
    }
}

fn partner_str(c: &PassCoordinator, id: PlayerId) -> String {
    if id == c.passer {
        format!("receiver p{}", c.receiver)
    } else {
        format!("passer p{}", c.passer)
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use dies_core::{BallData, GameStateData, SideAssignment, TeamColor, Vector3};

    pub fn player(id: u32, pos: Vector2, yaw: f64) -> PlayerData {
        let mut p = PlayerData::new(PlayerId::new(id));
        p.position = pos;
        p.yaw = Angle::from_radians(yaw);
        p
    }

    pub fn world(own: Vec<PlayerData>, ball: Option<Vector2>, dt: f64) -> TeamData {
        TeamData {
            t_received: 0.0,
            t_capture: 0.0,
            dt,
            own_players: own,
            opp_players: vec![],
            ball: ball.map(|b| BallData {
                timestamp: 0.0,
                position: Vector3::new(b.x, b.y, 0.0),
                raw_position: vec![],
                velocity: Vector3::zeros(),
                detected: true,
            }),
            field_geom: None,
            current_game_state: GameStateData::default(),
            ball_on_our_side: None,
            ball_on_opp_side: None,
            kicked_ball: None,
            possession: Default::default(),
        }
    }

    pub fn team_ctx() -> TeamContext {
        TeamContext::new(TeamColor::Blue, SideAssignment::BlueOnPositive)
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;

    #[test]
    fn secure_fails_when_ball_far() {
        // Passer nowhere near the ball -> weak Secure gives up with BallLost.
        let w = world(
            vec![
                player(0, Vector2::new(0.0, 0.0), 0.0),
                player(1, Vector2::new(2000.0, 0.0), 0.0),
            ],
            Some(Vector2::new(3000.0, 0.0)),
            0.016,
        );
        let tc = team_ctx();
        let mut c = PassCoordinator::new(PlayerId::new(0), PlayerId::new(1), None);
        let out = c.tick(&PassContext {
            world: &w,
            team_context: &tc,
        });
        assert_eq!(out.status, SkillStatus::Failed);
        assert!(matches!(
            c.result(),
            Some(PassResult::Failure {
                reason: PassFailure::BallLost,
                ..
            })
        ));
    }

    #[test]
    fn secure_advances_to_setup_with_ball() {
        let mut passer = player(0, Vector2::new(0.0, 0.0), 0.0);
        passer.has_ball = true;
        let w = world(
            vec![passer, player(1, Vector2::new(2000.0, 0.0), 0.0)],
            Some(Vector2::new(60.0, 0.0)),
            0.016,
        );
        let tc = team_ctx();
        let mut c = PassCoordinator::new(PlayerId::new(0), PlayerId::new(1), None);
        let out = c.tick(&PassContext {
            world: &w,
            team_context: &tc,
        });
        assert_eq!(out.status, SkillStatus::Running);
        assert_eq!(c.phase(), PassPhase::Setup);
    }

    #[test]
    fn barrier_gates_commit() {
        // Passer has the ball, is aimed at the receiver, and the receiver is on
        // the intercept point -> the barrier passes and we reach Commit.
        let mut passer = player(0, Vector2::new(0.0, 0.0), 0.0); // facing +x toward receiver
        passer.has_ball = true;
        let receiver = player(1, Vector2::new(2000.0, 0.0), std::f64::consts::PI);
        let tc = team_ctx();
        let mut c = PassCoordinator::new(PlayerId::new(0), PlayerId::new(1), None);

        // Tick 1: Secure -> Setup.
        let w = world(
            vec![passer.clone(), receiver.clone()],
            Some(Vector2::new(60.0, 0.0)),
            0.016,
        );
        c.tick(&PassContext {
            world: &w,
            team_context: &tc,
        });
        assert_eq!(c.phase(), PassPhase::Setup);

        // Tick 2: aligned + ready -> Commit.
        let out = c.tick(&PassContext {
            world: &w,
            team_context: &tc,
        });
        assert_eq!(out.status, SkillStatus::Running);
        assert_eq!(c.phase(), PassPhase::Commit);
    }

    #[test]
    fn not_aligned_stays_in_setup() {
        let mut passer = player(0, Vector2::new(0.0, 0.0), std::f64::consts::FRAC_PI_2); // facing +y, not the receiver
        passer.has_ball = true;
        let receiver = player(1, Vector2::new(2000.0, 0.0), std::f64::consts::PI);
        let tc = team_ctx();
        let mut c = PassCoordinator::new(PlayerId::new(0), PlayerId::new(1), None);
        let w = world(vec![passer, receiver], Some(Vector2::new(60.0, 0.0)), 0.016);
        c.tick(&PassContext {
            world: &w,
            team_context: &tc,
        }); // -> Setup
        c.tick(&PassContext {
            world: &w,
            team_context: &tc,
        }); // misaligned -> stays Setup
        assert_eq!(c.phase(), PassPhase::Setup);
    }

    /// Helper: build a misaligned passer (so Setup never commits) + a receiver,
    /// drive into Setup, and return the coordinator ready for ball-loss ticks.
    fn into_setup(has_ball: bool) -> (PassCoordinator, TeamContext, PlayerData, PlayerData) {
        let mut passer = player(0, Vector2::new(0.0, 0.0), std::f64::consts::FRAC_PI_2);
        passer.has_ball = true;
        let receiver = player(1, Vector2::new(2000.0, 0.0), std::f64::consts::PI);
        let tc = team_ctx();
        let mut c = PassCoordinator::new(PlayerId::new(0), PlayerId::new(1), None);
        let w = world(
            vec![passer.clone(), receiver.clone()],
            Some(Vector2::new(60.0, 0.0)),
            0.016,
        );
        c.tick(&PassContext {
            world: &w,
            team_context: &tc,
        });
        assert_eq!(c.phase(), PassPhase::Setup);
        let mut passer = passer;
        passer.has_ball = has_ball;
        (c, tc, passer, receiver)
    }

    #[test]
    fn setup_tolerates_brief_ball_loss() {
        // A breakbeam flicker shorter than the grace window must not abort.
        let (mut c, tc, passer, receiver) = into_setup(false);
        // ~0.13s < SETUP_BALL_LOST_GRACE (0.2s) of "no ball": stay in Setup.
        for _ in 0..8 {
            let w = world(
                vec![passer.clone(), receiver.clone()],
                Some(Vector2::new(60.0, 0.0)),
                0.016,
            );
            let out = c.tick(&PassContext {
                world: &w,
                team_context: &tc,
            });
            assert_eq!(out.status, SkillStatus::Running);
            assert_eq!(c.phase(), PassPhase::Setup);
        }
        // Re-acquiring the ball clears the debounce and keeps the pass alive.
        let mut regained = passer.clone();
        regained.has_ball = true;
        let w = world(
            vec![regained, receiver.clone()],
            Some(Vector2::new(60.0, 0.0)),
            0.016,
        );
        c.tick(&PassContext {
            world: &w,
            team_context: &tc,
        });
        assert_eq!(c.phase(), PassPhase::Setup);
    }

    #[test]
    fn setup_aborts_after_sustained_ball_loss() {
        // Loss past the grace window is a real BallLost.
        let (mut c, tc, passer, receiver) = into_setup(false);
        let mut last = SkillStatus::Running;
        for _ in 0..20 {
            let w = world(
                vec![passer.clone(), receiver.clone()],
                Some(Vector2::new(60.0, 0.0)),
                0.016,
            );
            last = c
                .tick(&PassContext {
                    world: &w,
                    team_context: &tc,
                })
                .status;
            if c.is_done() {
                break;
            }
        }
        assert_eq!(last, SkillStatus::Failed);
        assert!(matches!(
            c.result(),
            Some(PassResult::Failure {
                reason: PassFailure::BallLost,
                ..
            })
        ));
    }
}
