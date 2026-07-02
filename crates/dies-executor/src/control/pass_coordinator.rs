//! Joint pass coordinator.
//!
//! A [`PassCoordinator`] owns TWO robots (a passer and a receiver) through a
//! single state machine, ticked **once per frame**. It is the executor-side
//! realization of the atomic `ctx.pass(passer, receiver)` action.
//!
//! ## The release is a drive-through reflex strike — the pass never holds
//!
//! The passer NEVER takes the ball onto its dribbler. The release reuses the
//! IRL-tested [`HandleBallSkill`] drive-through strike
//! (`BallAction::Strike { acquire_first: false }`): stage behind the free ball
//! on the pass axis, then one-motion through it with the firmware reflex kick
//! armed, so the ball is struck the instant it trips the breakbeam. Departure is
//! *verified* by the skill (`KICK_DEPART_*`) before the coordinator trusts the
//! ball to be in flight.
//!
//! The passer/receiver timing handshake is the strike **gate**
//! ([`HandleBallSkill::set_strike_gate`]): while gated the strike skill stages
//! behind the ball, lined up on the pass axis, but its commit latch is
//! suppressed — it cannot drive through or arm the kicker. Once the receiver is
//! within [`RECEIVER_READY_DIST`] of the intercept point the coordinator
//! ungates; the staging point is already inside the commit corridor, so the
//! latch engages the next tick and the strike fires.
//!
//! ## Robustness = clean joint release + a typed verdict, never recovery
//!
//! On any terminal outcome the coordinator stops driving BOTH robots on the
//! same frame and reports a [`PassResult`]; the joint executor then releases
//! them back to the strategy. In particular a whiffed strike fails jointly with
//! [`PassFailure::KickFailed`] — the receiver never chases the ball back toward
//! the passer; the planner re-decides. A pass commanded while the passer holds
//! the ball is an upstream contract violation (ball handling lives entirely in
//! one `HandleBall` episode) and fails fast, as does a ball that isn't near the
//! passer to begin with — the pass never chases a loose ball.

use dies_core::{PlayerData, PlayerId, TeamData, Vector2, BALL_RADIUS, PLAYER_RADIUS};
use dies_strategy_protocol::{
    AcquirePosition, BallAction, PassBallState, PassFailure, PassResult, SkillStatus,
};
use dies_tunables_macro::tunables;

use super::avoidance::ObstacleSet;
use super::skill_executor::{ExecutableSkill, SkillContext, SkillProgress};
use super::team_context::TeamContext;
use crate::control::PlayerControlInput;
use crate::skills::executable::{HandleBallSkill, ReceiveSkill, ReflexReceiveSkill};

tunables! {
    section "Pass";

    /// Max distance the ball may be from the passer at pass start. Beyond this
    /// the pass fails immediately with `BallLost` — a pass never chases a loose
    /// ball across the field.
    #[tunable(unit = "mm", min = 100.0, max = 2000.0, step = 50.0)]
    START_DISTANCE: f64 = 600.0;
    /// How close (mm) the receiver must be to the intercept point for the
    /// coordinator to ungate the strike (the release barrier).
    #[tunable(unit = "mm", min = 50.0, max = 1000.0, step = 25.0)]
    RECEIVER_READY_DIST: f64 = 250.0;
    /// Time budget for the `Stage` phase (passer staging + receiver positioning).
    #[tunable(unit = "s", min = 1.0, max = 10.0, step = 0.5)]
    STAGE_TIMEOUT: f64 = 4.0;
    /// Ball displacement from its pass-start position that aborts the pass while
    /// staging — the gated strike would otherwise follow a rolling ball forever.
    #[tunable(unit = "mm", min = 100.0, max = 1500.0, step = 50.0)]
    BALL_MOVED_ABORT: f64 = 500.0;
    /// Backstop time budget for the `Release` phase. Sized past the strike
    /// skill's own reflex whiff timeout (5 s) so the skill's typed `KickFailed`
    /// normally fires first; this only catches a release whose commit latch
    /// never engages at all.
    #[tunable(unit = "s", min = 1.0, max = 15.0, step = 0.5)]
    RELEASE_TIMEOUT: f64 = 7.0;
    /// Max perpendicular distance the receiver will travel off the pass line to
    /// catch a misaimed ball; beyond it the flight verdict is `ReceiverMissed`.
    #[tunable(unit = "mm", min = 200.0, max = 4000.0, step = 100.0)]
    CAPTURE_LIMIT: f64 = 1500.0;
    /// Ball speed below which, if not near the receiver, the flight verdict is
    /// `StoppedShort`.
    #[tunable(unit = "mm/s", min = 50.0, max = 600.0, step = 25.0)]
    STOPPED_SHORT_SPEED: f64 = 150.0;
    /// Distance within which the ball counts as "at the receiver" for the
    /// stopped-short check.
    #[tunable(unit = "mm", min = 50.0, max = 600.0, step = 10.0)]
    NEAR_RECEIVER_DIST: f64 = 200.0;
    /// Fraction of the pass-line length past the intercept point beyond which an
    /// uncaught ball is declared `ReceiverMissed`.
    #[tunable(unit = "x", min = 1.0, max = 2.0, step = 0.05)]
    OVERSHOOT_FRACTION: f64 = 1.2;
    /// Rough estimate of pass ball speed, used only for the flight timeout.
    #[tunable(unit = "mm/s", min = 1000.0, max = 6000.0, step = 100.0)]
    PASS_SPEED_ESTIMATE: f64 = 2500.0;
    /// Multiplicative margin on the expected flight time before timing out.
    #[tunable(unit = "x", min = 1.0, max = 5.0, step = 0.25)]
    FLIGHT_TIMEOUT_MARGIN: f64 = 2.5;
    /// Additive base on the flight timeout.
    #[tunable(unit = "s", min = 0.0, max = 2.0, step = 0.1)]
    FLIGHT_TIMEOUT_BASE: f64 = 0.6;
}

/// Distance (mm) at which an opponent is considered to control the ball.
/// Derived from physical radii — structural, not runtime-tunable (same
/// convention as `handle_ball`'s derived constants).
const OPPONENT_POSSESSION_DIST: f64 = PLAYER_RADIUS + BALL_RADIUS + 30.0;

/// The phase of a pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassPhase {
    /// Passer stages behind the ball (strike gated); receiver moves to the
    /// intercept-ready pose. Concurrent, freely abortable.
    Stage,
    /// Barrier passed — strike ungated. The skill commits, reflex-fires, and
    /// verifies departure. The only (near-)irreversible phase.
    Release,
    /// Ball in flight (verified departure); receiver tracks and intercepts.
    Flight,
    /// Terminal — see `result`.
    Done,
}

impl PassPhase {
    pub fn as_str(&self) -> &'static str {
        match self {
            PassPhase::Stage => "Stage",
            PassPhase::Release => "Release",
            PassPhase::Flight => "Flight",
            PassPhase::Done => "Done",
        }
    }

    pub fn index(&self) -> f64 {
        match self {
            PassPhase::Stage => 0.0,
            PassPhase::Release => 1.0,
            PassPhase::Flight => 2.0,
            PassPhase::Done => 3.0,
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
    /// One-timer mode: `Some(point)` = the receiver redirects the arriving ball
    /// toward this point via the pre-armed reflex kick instead of catching it.
    /// The coordinator trusts the strategy's deflection geometry and only emits
    /// the angle as a debug value.
    forward_to: Option<Vector2>,

    phase: PassPhase,
    phase_elapsed: f64,
    total_elapsed: f64,

    /// Intercept point: `target_hint` or the receiver's position, captured on the
    /// first tick. Hint updates apply only while in `Stage` (the strike re-stages
    /// naturally); frozen from `Release` on.
    intercept_point: Option<Vector2>,
    /// Ball position at pass creation — anchors the ball-moved abort guard.
    ball_start: Option<Vector2>,
    /// Ball position at verified departure — the pass-line origin for `Flight`
    /// (the live passer position drifts as it disengages).
    flight_origin: Option<Vector2>,
    /// Whether the in-flight ball ever came within [`NEAR_RECEIVER_DIST`] of the
    /// receiver. In one-timer mode this arms the endgame: once the ball has
    /// reached the receiver, the deflected (fast, off-the-pass-line) ball must
    /// not trip the miss/short verdicts — the outcome is the deflection verify,
    /// a dud-catch, or the flight timeout.
    ball_reached_receiver: bool,

    /// The passer's drive for the whole pass: the IRL-tested drive-through
    /// reflex strike, gated until the receiver is ready.
    strike: HandleBallSkill,
    /// Receiver sub-skill for a catch pass (`forward_to == None`).
    receive: ReceiveSkill,
    /// Receiver sub-skill for a one-timer (`forward_to == Some`): positions on
    /// the same intercept solve but faces the forward target with the reflex
    /// pre-armed for the whole pass, so no rotation eats into the flight time.
    /// Its standalone no-arrival timeout is suppressed — the coordinator owns
    /// every pass-level clock.
    reflex_receive: ReflexReceiveSkill,

    /// Terminal result, set once when `phase == Done`.
    result: Option<PassResult>,
}

impl PassCoordinator {
    pub fn new(
        passer: PlayerId,
        receiver: PlayerId,
        target_hint: Option<Vector2>,
        forward_to: Option<Vector2>,
    ) -> Self {
        // Sub-skills are seeded with placeholder geometry; reconfigured every tick.
        let mut reflex_receive =
            ReflexReceiveSkill::new(Vector2::zeros(), Vector2::zeros(), Vector2::zeros(), 0.0);
        // The gated Stage can outlive the skill's standalone no-arrival budget;
        // the coordinator's stage/release/flight clocks are the only authority.
        reflex_receive.set_no_arrive_timeout(true);
        Self {
            passer,
            receiver,
            target_hint,
            forward_to,
            phase: PassPhase::Stage,
            phase_elapsed: 0.0,
            total_elapsed: 0.0,
            intercept_point: None,
            ball_start: None,
            flight_origin: None,
            ball_reached_receiver: false,
            strike: HandleBallSkill::new(
                BallAction::Strike {
                    target: Vector2::zeros(),
                    acquire_first: false,
                },
                AcquirePosition::Default,
                true,
            ),
            receive: ReceiveSkill::new(Vector2::zeros(), Vector2::zeros(), CAPTURE_LIMIT(), false),
            reflex_receive,
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

    /// Update the optional hints (param update from the strategy). The target
    /// hint moves the intercept point only while still in `Stage`; the forward
    /// (one-timer) target is a facing/redirect goal and may be refreshed
    /// throughout — but never toggled between catch and one-timer mode
    /// mid-pass (the receiver skill and success semantics are fixed at start).
    pub fn update_hints(&mut self, target_hint: Option<Vector2>, forward_to: Option<Vector2>) {
        self.target_hint = target_hint;
        if self.phase == PassPhase::Stage {
            if let Some(hint) = target_hint {
                self.intercept_point = Some(hint);
            }
        }
        if self.forward_to.is_some() && forward_to.is_some() {
            self.forward_to = forward_to;
        }
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
            // The drive-through strike derives its axis from the target — no
            // obstacle-aware approach-side selection is involved.
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

        // First-tick initialization + entry guards.
        if self.ball_start.is_none() {
            if let Some(r) = self.entry_guard(&passer, ball_pos) {
                let result = self.finish(r);
                self.emit_debug(ctx, &passer, &receiver, ball_pos, receiver.position);
                return PassTickOutput {
                    passer_input: PlayerControlInput::default(),
                    receiver_input: PlayerControlInput::default(),
                    status: SkillStatus::Failed,
                    result: Some(result),
                };
            }
            self.ball_start = ball_pos;
            self.intercept_point = Some(self.target_hint.unwrap_or(receiver.position));
        }
        let target_point = self.intercept_point.unwrap_or(receiver.position);

        let mut result: Option<PassResult> = None;
        let mut passer_input = PlayerControlInput::default();
        let mut receiver_input = PlayerControlInput::default();

        match self.phase {
            PassPhase::Stage => {
                if let Some(r) = self.preflight_guard(ctx, ball_pos, STAGE_TIMEOUT()) {
                    result = Some(self.finish(r));
                } else {
                    // Barrier: ungate the strike once the receiver is ready. A
                    // one-way edge — the receiver drifting off afterwards does
                    // not re-gate a committed strike.
                    let ready = (receiver.position - target_point).norm() < RECEIVER_READY_DIST();
                    if ready {
                        self.enter(PassPhase::Release);
                        log::info!("entering release {}", receiver.position - target_point);
                    }
                    let (pi, r) = self.tick_strike(ctx, &passer, ball_pos, target_point, !ready);
                    passer_input = pi;
                    result = r.map(|r| self.finish(r));
                    let (ri, _) = self.tick_receive(ctx, &receiver, ball_pos, target_point);
                    receiver_input = ri;
                }
            }

            PassPhase::Release => {
                // Tick the (ungated) strike FIRST: a verified departure must win
                // over the guards — the kick itself moves the ball.
                let (pi, r) = self.tick_strike(ctx, &passer, ball_pos, target_point, false);
                passer_input = pi;
                if let Some(r) = r {
                    result = Some(self.finish(r));
                } else if self.phase == PassPhase::Flight {
                    // Departed this frame — fall into flight tracking below.
                } else if let Some(r) = self.release_guard(ctx, ball_pos) {
                    result = Some(self.finish(r));
                }
                let (ri, rr) = self.tick_receive(ctx, &receiver, ball_pos, target_point);
                receiver_input = ri;
                if result.is_none() {
                    // A one-timer can verify its deflection while the passer's
                    // own departure verify is still pending.
                    result = rr.map(|r| self.finish(r));
                }
            }

            PassPhase::Flight => {}

            PassPhase::Done => unreachable!("handled above"),
        }

        // Flight runs after the match so a Release→Flight transition tracks the
        // ball on the same frame it departs.
        if self.phase == PassPhase::Flight && result.is_none() {
            // Passer disengages (default input) to avoid a double touch.
            passer_input = PlayerControlInput::default();
            let from = self.flight_origin.unwrap_or(passer.position);
            // Arm the one-timer endgame: once the ball has reached the receiver,
            // the deflected ball must not trip the miss/short verdicts.
            if let Some(bp) = ball_pos {
                if (bp - receiver.position).norm() < NEAR_RECEIVER_DIST() {
                    self.ball_reached_receiver = true;
                }
            }
            let (ri, rr) = self.tick_receive_from(ctx, &receiver, from, target_point);
            receiver_input = ri;
            if let Some(r) = rr {
                result = Some(self.finish(r));
            } else if let Some(r) = self.check_flight_terminal(ctx, &receiver, from, target_point)
            {
                result = Some(self.finish(r));
            }
        }

        self.emit_debug(ctx, &passer, &receiver, ball_pos, target_point);

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

    /// First-tick contract checks. `Some` fails the pass before it starts.
    fn entry_guard(&self, passer: &PlayerData, ball_pos: Option<Vector2>) -> Option<PassResult> {
        // A pass never starts from possession: ball handling lives entirely in
        // one `HandleBall` episode, and the drive-through needs a free ball.
        if passer.has_ball {
            return Some(PassResult::Failure {
                reason: PassFailure::BallLost,
                ball_state: PassBallState::WithPasser,
            });
        }
        let Some(bp) = ball_pos else {
            return Some(PassResult::Failure {
                reason: PassFailure::BallLost,
                ball_state: PassBallState::Unknown,
            });
        };
        if (bp - passer.position).norm() > START_DISTANCE() {
            // The pass never chases a loose ball.
            return Some(PassResult::Failure {
                reason: PassFailure::BallLost,
                ball_state: PassBallState::Loose { position: bp },
            });
        }
        None
    }

    /// Per-tick guards while the ball is still with us (`Stage`): the gated
    /// strike follows a live ball with no internal timeout, so the coordinator
    /// must abort a ball that vanished, rolled away, or was taken.
    fn preflight_guard(
        &mut self,
        ctx: &PassContext<'_>,
        ball_pos: Option<Vector2>,
        timeout: f64,
    ) -> Option<PassResult> {
        let Some(bp) = ball_pos else {
            return Some(PassResult::Failure {
                reason: PassFailure::BallLost,
                ball_state: PassBallState::Unknown,
            });
        };
        if let Some(start) = self.ball_start {
            if (bp - start).norm() > BALL_MOVED_ABORT() {
                return Some(PassResult::Failure {
                    reason: PassFailure::BallLost,
                    ball_state: PassBallState::Loose { position: bp },
                });
            }
        }
        let opp_has_ball = ctx
            .world
            .opp_players
            .iter()
            .any(|o| (o.position - bp).norm() < OPPONENT_POSSESSION_DIST);
        if opp_has_ball {
            return Some(PassResult::Failure {
                reason: PassFailure::BallLost,
                ball_state: PassBallState::WithOpponent,
            });
        }
        if self.phase_elapsed > timeout {
            return Some(PassResult::Failure {
                reason: PassFailure::Timeout,
                ball_state: PassBallState::Loose { position: bp },
            });
        }
        None
    }

    /// `Release` guards: no ball-moved check (the strike itself moves the ball —
    /// a verified departure is handled before this), just theft + the backstop.
    fn release_guard(
        &mut self,
        ctx: &PassContext<'_>,
        ball_pos: Option<Vector2>,
    ) -> Option<PassResult> {
        let Some(bp) = ball_pos else {
            return Some(PassResult::Failure {
                reason: PassFailure::BallLost,
                ball_state: PassBallState::Unknown,
            });
        };
        let opp_has_ball = ctx
            .world
            .opp_players
            .iter()
            .any(|o| (o.position - bp).norm() < OPPONENT_POSSESSION_DIST);
        if opp_has_ball {
            return Some(PassResult::Failure {
                reason: PassFailure::BallLost,
                ball_state: PassBallState::WithOpponent,
            });
        }
        if self.phase_elapsed > RELEASE_TIMEOUT() {
            return Some(PassResult::Failure {
                reason: PassFailure::Timeout,
                ball_state: PassBallState::Loose { position: bp },
            });
        }
        None
    }

    /// Tick the passer's strike skill (re-aimed at the intercept point, gated or
    /// not). Returns the passer input plus, on a skill-terminal tick, either a
    /// `Flight` transition (verified departure, via side effect) or the joint
    /// `KickFailed` verdict.
    fn tick_strike(
        &mut self,
        ctx: &PassContext<'_>,
        passer: &PlayerData,
        ball_pos: Option<Vector2>,
        target_point: Vector2,
        gated: bool,
    ) -> (PlayerControlInput, Option<PassResult>) {
        self.strike.set_strike_gate(gated);
        self.strike.reconfigure(
            BallAction::Strike {
                target: target_point,
                acquire_first: false,
            },
            AcquirePosition::Default,
        );
        let sctx = self.skill_ctx(ctx, passer);
        match self.strike.tick(sctx) {
            SkillProgress::Continue(input) => (input, None),
            SkillProgress::Done(_) => match self.strike.status() {
                SkillStatus::Succeeded => {
                    // Verified departure — the ball is genuinely on its way.
                    self.flight_origin = ball_pos.or(Some(passer.position));
                    self.enter(PassPhase::Flight);
                    log::info!("entering flight");
                    (PlayerControlInput::default(), None)
                }
                _ => {
                    log::info!("strike whiffed");
                    (
                        // Whiff (reflex never connected) or a strike bail: fail BOTH
                        // robots now — the receiver must not chase toward the passer.
                        PlayerControlInput::default(),
                        Some(PassResult::Failure {
                            reason: PassFailure::KickFailed,
                            ball_state: ball_pos
                                .map(|p| PassBallState::Loose { position: p })
                                .unwrap_or(PassBallState::Unknown),
                        }),
                    )
                }
            },
        }
    }

    /// Drive the receiver's [`ReceiveSkill`] with the ball (or its stand-in) as
    /// the pass-line origin.
    fn tick_receive(
        &mut self,
        ctx: &PassContext<'_>,
        receiver: &PlayerData,
        ball_pos: Option<Vector2>,
        target_point: Vector2,
    ) -> (PlayerControlInput, Option<PassResult>) {
        // Before the kick the pass originates at the ball itself (≈200 mm off
        // the passer's center along the axis), not the passer.
        let from = ball_pos
            .or_else(|| self.player(ctx.world, self.passer).map(|p| p.position))
            .unwrap_or(receiver.position);
        self.tick_receive_from(ctx, receiver, from, target_point)
    }

    /// Drive the receiver one frame. Returns its input plus, in one-timer mode,
    /// the verified-deflection success on the tick the reflex receive confirms
    /// the ball departed onward.
    fn tick_receive_from(
        &mut self,
        ctx: &PassContext<'_>,
        receiver: &PlayerData,
        from: Vector2,
        target_point: Vector2,
    ) -> (PlayerControlInput, Option<PassResult>) {
        if let Some(forward) = self.forward_to {
            // One-timer: same intercept solve, but the receiver faces the
            // forward target with the reflex pre-armed for the whole pass.
            self.reflex_receive
                .reconfigure(from, target_point, forward, CAPTURE_LIMIT());
            let sctx = self.skill_ctx(ctx, receiver);
            return match self.reflex_receive.tick(sctx) {
                SkillProgress::Continue(input) => (input, None),
                SkillProgress::Done(_) => match self.reflex_receive.status() {
                    // The skill's success is the verified deflection (ball at
                    // the mouth, then departing fast) — the pass is complete.
                    SkillStatus::Succeeded => (
                        PlayerControlInput::default(),
                        Some(PassResult::Success {
                            receiver: self.receiver,
                            forwarded: true,
                        }),
                    ),
                    // Its no-arrival timeout is suppressed, so a failure should
                    // be unreachable; hold and let the coordinator clocks rule.
                    _ => (PlayerControlInput::new(), None),
                },
            };
        }

        self.receive
            .reconfigure(from, target_point, CAPTURE_LIMIT(), true);
        let sctx = self.skill_ctx(ctx, receiver);
        let input = match self.receive.tick(sctx) {
            SkillProgress::Continue(input) => input,
            // ReceiveSkill only "completes" on possession; the coordinator owns
            // the real success decision, so just hold on completion.
            SkillProgress::Done(_) => PlayerControlInput::new(),
        };
        (input, None)
    }

    /// Evaluate the terminal conditions during `Flight`. Returns `Some` to end.
    fn check_flight_terminal(
        &mut self,
        ctx: &PassContext<'_>,
        receiver: &PlayerData,
        from: Vector2,
        target_point: Vector2,
    ) -> Option<PassResult> {
        let Some(ball) = ctx.world.ball.as_ref() else {
            // Ball lost from vision during flight — can't track; time out via guard.
            if self.phase_elapsed > self.flight_timeout(from, target_point) {
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
        // proximity + breakbeam fusion subsumes the old vision fallback). In
        // one-timer mode this is the dud-catch fallback — the reflex didn't
        // fire but the ball is seated; still a completed pass.
        if receiver.has_ball {
            return Some(PassResult::Success {
                receiver: self.receiver,
                forwarded: false,
            });
        }

        // One-timer endgame: the ball reached the receiver, so from here the
        // ball's motion is the DEFLECTION, not the pass — a fast off-line ball
        // is the intended redirect, not a miss/steal. Only the verified
        // deflection (receiver skill), the dud-catch above, or the flight
        // timeout below may end the pass.
        let endgame = self.forward_to.is_some() && self.ball_reached_receiver;

        if !endgame {
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
            // Joint-terminal, so the receiver stops converging within ~0.3 s of the
            // ball dying — its ReceiveSkill fetch advance is capped anyway.
            if self.phase_elapsed > 0.3
                && ball_vel.norm() < STOPPED_SHORT_SPEED()
                && (ball_pos - receiver.position).norm() > NEAR_RECEIVER_DIST()
            {
                return Some(PassResult::Failure {
                    reason: PassFailure::StoppedShort,
                    ball_state: PassBallState::Loose { position: ball_pos },
                });
            }

            // Receiver missed: ball overshot the intercept point along the line, or
            // strayed beyond the capture window.
            let line = target_point - from;
            let len = line.norm();
            if len > 1.0 {
                let dir = line / len;
                let progress = (ball_pos - from).dot(&dir) / len;
                let perp = (ball_pos - from) - dir * (ball_pos - from).dot(&dir);
                if progress > OVERSHOOT_FRACTION() || perp.norm() > CAPTURE_LIMIT() {
                    return Some(PassResult::Failure {
                        reason: PassFailure::ReceiverMissed,
                        ball_state: PassBallState::Loose { position: ball_pos },
                    });
                }
            }
        }

        // Flight timeout.
        if self.phase_elapsed > self.flight_timeout(from, target_point) {
            return Some(PassResult::Failure {
                reason: PassFailure::Timeout,
                ball_state: PassBallState::Loose { position: ball_pos },
            });
        }

        None
    }

    fn flight_timeout(&self, from: Vector2, target_point: Vector2) -> f64 {
        let dist = (target_point - from).norm();
        FLIGHT_TIMEOUT_BASE() + FLIGHT_TIMEOUT_MARGIN() * (dist / PASS_SPEED_ESTIMATE())
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

    fn emit_debug(
        &self,
        ctx: &PassContext<'_>,
        passer: &PlayerData,
        receiver: &PlayerData,
        ball_pos: Option<Vector2>,
        target_point: Vector2,
    ) {
        let tc = ctx.team_context;
        let pair = format!("pass.{}-{}", self.passer, self.receiver);

        // --- Full internal-state readout (strings/numbers). ---
        tc.debug_string(format!("{pair}.phase"), self.phase.as_str());
        tc.debug_value(format!("{pair}.phase_index"), self.phase.index());
        tc.debug_value(format!("{pair}.phase_elapsed_s"), self.phase_elapsed);
        tc.debug_value(format!("{pair}.total_elapsed_s"), self.total_elapsed);

        let receiver_dist = (receiver.position - target_point).norm();
        let ready = receiver_dist < RECEIVER_READY_DIST();
        let gated = self.phase == PassPhase::Stage;
        let pass_distance = ball_pos
            .map(|bp| (target_point - bp).norm())
            .unwrap_or((target_point - passer.position).norm());

        tc.debug_string(format!("{pair}.gated"), bool_str(gated));
        tc.debug_string(format!("{pair}.receiver_ready"), bool_str(ready));
        tc.debug_value(format!("{pair}.receiver_dist_mm"), receiver_dist);
        tc.debug_value(format!("{pair}.pass_distance_mm"), pass_distance);
        tc.debug_string(
            format!("{pair}.mode"),
            if self.forward_to.is_some() {
                "one-timer"
            } else {
                "catch"
            },
        );
        if let Some(PassResult::Failure { reason, .. }) = &self.result {
            tc.debug_string(format!("{pair}.result"), format!("{reason:?}"));
        } else if let Some(PassResult::Success { forwarded, .. }) = &self.result {
            tc.debug_string(
                format!("{pair}.result"),
                if *forwarded {
                    "Success(forwarded)"
                } else {
                    "Success(caught)"
                },
            );
        }

        // Mirror the key scalars under each player so they show in the inspector.
        for id in [self.passer, self.receiver] {
            tc.debug_string(format!("p{id}.pass.phase"), self.phase.as_str());
            tc.debug_string(format!("p{id}.pass.partner"), partner_str(self, id));
        }

        // --- Field overlays (absolute coords; red→green at the barrier). ---
        let line_color = if gated {
            dies_core::DebugColor::Red
        } else {
            dies_core::DebugColor::Green
        };
        let line_from = ball_pos.unwrap_or(passer.position);
        dies_core::debug_line(
            tc.key(format!("{pair}.line")),
            line_from,
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
            CAPTURE_LIMIT(),
            dies_core::DebugColor::Gray,
        );

        // One-timer overlays: the redirect line and the deflection angle between
        // the incoming pass and the outgoing shot. The coordinator trusts the
        // strategy's geometry — this is observability only (a deep angle means
        // the ball will likely glance off the shell instead of one-timing).
        if let Some(forward) = self.forward_to {
            dies_core::debug_line(
                tc.key(format!("{pair}.forward")),
                target_point,
                forward,
                dies_core::DebugColor::Orange,
            );
            let incoming = target_point - line_from;
            let outgoing = forward - target_point;
            if incoming.norm() > 1.0 && outgoing.norm() > 1.0 {
                tc.debug_value(
                    format!("{pair}.deflection_deg"),
                    incoming.angle(&outgoing).to_degrees(),
                );
            }
        }
    }
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
    use dies_core::{Angle, BallData, GameStateData, SideAssignment, TeamColor, Vector3};

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
            sidelined_players: vec![],
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
    use crate::control::KickerControlInput;
    use dies_core::Vector3;

    fn ctx<'a>(w: &'a TeamData, tc: &'a TeamContext) -> PassContext<'a> {
        PassContext {
            world: w,
            team_context: tc,
        }
    }

    /// Ball at the origin, passer staged right behind it on the −x side of the
    /// pass axis toward a receiver on +x — the geometry that commits instantly
    /// once ungated.
    const BALL: Vector2 = Vector2::new(0.0, 0.0);
    const STAGED: Vector2 = Vector2::new(-200.0, 0.0);
    const RECEIVER_POS: Vector2 = Vector2::new(2000.0, 0.0);

    fn world_at(own: Vec<PlayerData>, ball: Option<Vector2>, t: f64) -> TeamData {
        let mut w = world(own, ball, 0.016);
        w.t_received = t;
        w
    }

    #[test]
    fn fails_fast_when_passer_holds_the_ball() {
        // A pass never starts from possession (ball handling lives entirely in
        // one HandleBall episode): fail fast, planner re-decides.
        let mut passer = player(0, STAGED, 0.0);
        passer.has_ball = true;
        let w = world(
            vec![passer, player(1, RECEIVER_POS, 0.0)],
            Some(BALL),
            0.016,
        );
        let tc = team_ctx();
        let mut c = PassCoordinator::new(PlayerId::new(0), PlayerId::new(1), None, None);
        let out = c.tick(&ctx(&w, &tc));
        assert_eq!(out.status, SkillStatus::Failed);
        assert!(matches!(
            c.result(),
            Some(PassResult::Failure {
                reason: PassFailure::BallLost,
                ball_state: PassBallState::WithPasser,
            })
        ));
    }

    #[test]
    fn fails_when_ball_far_from_passer() {
        // Passer nowhere near the ball -> the pass never chases a loose ball.
        let w = world(
            vec![
                player(0, Vector2::new(0.0, 0.0), 0.0),
                player(1, RECEIVER_POS, 0.0),
            ],
            Some(Vector2::new(3000.0, 0.0)),
            0.016,
        );
        let tc = team_ctx();
        let mut c = PassCoordinator::new(PlayerId::new(0), PlayerId::new(1), None, None);
        let out = c.tick(&ctx(&w, &tc));
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
    fn stays_gated_until_receiver_ready_then_releases() {
        // Target hint away from the receiver: the barrier is NOT satisfied, so
        // the passer must stage (no reflex armed) even though it is parked in
        // commit-ready geometry. Once the receiver reaches the intercept point,
        // the coordinator ungates and the strike arms the reflex.
        let hint = Vector2::new(2000.0, 0.0);
        let far_receiver = player(1, Vector2::new(2000.0, 1500.0), 0.0);
        let tc = team_ctx();
        let mut c = PassCoordinator::new(PlayerId::new(0), PlayerId::new(1), Some(hint), None);

        let w = world_at(
            vec![player(0, STAGED, 0.0), far_receiver.clone()],
            Some(BALL),
            0.0,
        );
        let out = c.tick(&ctx(&w, &tc));
        assert_eq!(c.phase(), PassPhase::Stage);
        assert!(
            !matches!(out.passer_input.kicker, KickerControlInput::ReflexKick),
            "gated strike must not arm the reflex"
        );

        // Receiver arrives at the intercept point -> Release, reflex armed.
        let ready_receiver = player(1, hint, 0.0);
        let w = world_at(
            vec![player(0, STAGED, 0.0), ready_receiver],
            Some(BALL),
            0.016,
        );
        let out = c.tick(&ctx(&w, &tc));
        assert_eq!(c.phase(), PassPhase::Release);
        assert!(
            matches!(out.passer_input.kicker, KickerControlInput::ReflexKick),
            "ungated strike from staging must commit and arm the reflex"
        );
        assert_eq!(out.status, SkillStatus::Running);
    }

    #[test]
    fn stage_aborts_when_ball_rolls_away() {
        let hint = Vector2::new(2000.0, 0.0);
        let tc = team_ctx();
        let mut c = PassCoordinator::new(PlayerId::new(0), PlayerId::new(1), Some(hint), None);
        let far_receiver = player(1, Vector2::new(2000.0, 1500.0), 0.0);

        let w = world_at(
            vec![player(0, STAGED, 0.0), far_receiver.clone()],
            Some(BALL),
            0.0,
        );
        c.tick(&ctx(&w, &tc));
        assert_eq!(c.phase(), PassPhase::Stage);

        // Ball knocked > BALL_MOVED_ABORT from its start -> joint BallLost.
        let w = world_at(
            vec![player(0, STAGED, 0.0), far_receiver],
            Some(Vector2::new(0.0, BALL_MOVED_ABORT() + 100.0)),
            0.016,
        );
        let out = c.tick(&ctx(&w, &tc));
        assert_eq!(out.status, SkillStatus::Failed);
        assert!(matches!(
            c.result(),
            Some(PassResult::Failure {
                reason: PassFailure::BallLost,
                ball_state: PassBallState::Loose { .. },
            })
        ));
    }

    #[test]
    fn stage_aborts_when_opponent_takes_the_ball() {
        let hint = Vector2::new(2000.0, 0.0);
        let tc = team_ctx();
        let mut c = PassCoordinator::new(PlayerId::new(0), PlayerId::new(1), Some(hint), None);
        let far_receiver = player(1, Vector2::new(2000.0, 1500.0), 0.0);

        let w = world_at(
            vec![player(0, STAGED, 0.0), far_receiver.clone()],
            Some(BALL),
            0.0,
        );
        c.tick(&ctx(&w, &tc));

        let mut w = world_at(
            vec![player(0, STAGED, 0.0), far_receiver],
            Some(BALL),
            0.016,
        );
        w.opp_players.push(player(9, Vector2::new(80.0, 0.0), 0.0));
        let out = c.tick(&ctx(&w, &tc));
        assert_eq!(out.status, SkillStatus::Failed);
        assert!(matches!(
            c.result(),
            Some(PassResult::Failure {
                reason: PassFailure::BallLost,
                ball_state: PassBallState::WithOpponent,
            })
        ));
    }

    #[test]
    fn stage_times_out() {
        let hint = Vector2::new(2000.0, 0.0);
        let tc = team_ctx();
        let mut c = PassCoordinator::new(PlayerId::new(0), PlayerId::new(1), Some(hint), None);
        let far_receiver = player(1, Vector2::new(2000.0, 1500.0), 0.0);

        let mk = |t: f64, dt: f64| {
            let mut w = world(
                vec![player(0, STAGED, 0.0), far_receiver.clone()],
                Some(BALL),
                dt,
            );
            w.t_received = t;
            w
        };
        c.tick(&ctx(&mk(0.0, 0.016), &tc));
        assert_eq!(c.phase(), PassPhase::Stage);
        // One big frame past the stage budget (receiver never ready).
        let out = c.tick(&ctx(&mk(0.016 + STAGE_TIMEOUT(), STAGE_TIMEOUT()), &tc));
        assert_eq!(out.status, SkillStatus::Failed);
        assert!(matches!(
            c.result(),
            Some(PassResult::Failure {
                reason: PassFailure::Timeout,
                ..
            })
        ));
    }

    #[test]
    fn whiff_fails_both_robots_with_kick_failed() {
        // No hint: the receiver is trivially "ready" where it stands, so the
        // pass releases immediately; the passer commits (armed) but the ball
        // never departs -> the strike's reflex timeout fails the WHOLE pass
        // jointly on one frame with KickFailed. The receiver is released the
        // same frame — it never chases the dead ball toward the passer.
        let tc = team_ctx();
        let mut c = PassCoordinator::new(PlayerId::new(0), PlayerId::new(1), None, None);
        let receiver = player(1, RECEIVER_POS, 0.0);

        // Tick 1 (t=0): enters Release and commits (reflex armed at t=0).
        let w = world_at(
            vec![player(0, STAGED, 0.0), receiver.clone()],
            Some(BALL),
            0.0,
        );
        let out = c.tick(&ctx(&w, &tc));
        assert_eq!(c.phase(), PassPhase::Release);
        assert!(matches!(
            out.passer_input.kicker,
            KickerControlInput::ReflexKick
        ));

        // Tick 2 well past the strike's REFLEX_TIMEOUT (5 s), ball never moved.
        let mut w = world(vec![player(0, STAGED, 0.0), receiver], Some(BALL), 0.016);
        w.t_received = 6.0;
        let out = c.tick(&ctx(&w, &tc));
        assert_eq!(out.status, SkillStatus::Failed);
        assert!(matches!(
            c.result(),
            Some(PassResult::Failure {
                reason: PassFailure::KickFailed,
                ball_state: PassBallState::Loose { .. },
            })
        ));
        // Terminal: both robots get default (released) inputs from now on.
        let out = c.tick(&ctx(&w, &tc));
        assert!(out.result.is_none());
        assert_eq!(out.status, SkillStatus::Failed);
    }

    /// Drive a coordinator through a real verified departure into `Flight`:
    /// commit at t=0, then present a ball that has departed along the axis.
    fn into_flight(tc: &TeamContext) -> PassCoordinator {
        let mut c = PassCoordinator::new(PlayerId::new(0), PlayerId::new(1), None, None);
        let receiver = player(1, RECEIVER_POS, 0.0);
        let w = world_at(
            vec![player(0, STAGED, 0.0), receiver.clone()],
            Some(BALL),
            0.0,
        );
        c.tick(&ctx(&w, tc));
        assert_eq!(c.phase(), PassPhase::Release);

        // Ball departed along +x at kick speed. It must still be inside the
        // commit-latch release band (within ~140 mm of the staged robot's
        // corridor) when the departure check runs — as it is one frame after a
        // real kick — so departure verifies via KICK_DEPART_SPEED and the
        // coordinator enters Flight on the same frame.
        let mut w = world(
            vec![player(0, STAGED, 0.0), receiver],
            Some(Vector2::new(60.0, 0.0)),
            0.016,
        );
        w.t_received = 0.032;
        if let Some(b) = w.ball.as_mut() {
            b.velocity = Vector3::new(2500.0, 0.0, 0.0);
        }
        c.tick(&ctx(&w, tc));
        assert_eq!(c.phase(), PassPhase::Flight);
        c
    }

    #[test]
    fn flight_succeeds_when_receiver_gets_the_ball() {
        let tc = team_ctx();
        let mut c = into_flight(&tc);
        let mut receiver = player(1, RECEIVER_POS, 0.0);
        receiver.has_ball = true;
        let mut w = world(
            vec![player(0, STAGED, 0.0), receiver],
            Some(RECEIVER_POS),
            0.016,
        );
        w.t_received = 0.8;
        let out = c.tick(&ctx(&w, &tc));
        assert_eq!(out.status, SkillStatus::Succeeded);
        assert!(matches!(c.result(), Some(PassResult::Success { .. })));
    }

    #[test]
    fn flight_fails_stopped_short() {
        let tc = team_ctx();
        let mut c = into_flight(&tc);
        let receiver = player(1, RECEIVER_POS, 0.0);
        // Ball dead at midfield, well short of the receiver, past the grace.
        let mk = |t: f64, dt: f64| {
            let mut w = world(
                vec![player(0, STAGED, 0.0), receiver.clone()],
                Some(Vector2::new(900.0, 0.0)),
                dt,
            );
            w.t_received = t;
            w
        };
        // Accumulate past the 0.3 s stopped-short grace.
        c.tick(&ctx(&mk(0.24, 0.2), &tc));
        let out = c.tick(&ctx(&mk(0.44, 0.2), &tc));
        assert_eq!(out.status, SkillStatus::Failed);
        assert!(matches!(
            c.result(),
            Some(PassResult::Failure {
                reason: PassFailure::StoppedShort,
                ..
            })
        ));
    }

    // ── one-timer (forward_to) mode ──────────────────────────────────────────

    /// Down-axis forward target: the deflection continues along +x.
    const FORWARD: Vector2 = Vector2::new(4000.0, 0.0);

    #[test]
    fn one_timer_receiver_waits_armed_facing_forward() {
        // A hinted one-timer whose receiver is still traveling: the pass stays
        // gated in Stage, and the receiver must already hold the reflex armed
        // and face the FORWARD target (not the passer), so no rotation eats
        // into the flight later.
        let hint = Vector2::new(2000.0, 0.0);
        let tc = team_ctx();
        let mut c =
            PassCoordinator::new(PlayerId::new(0), PlayerId::new(1), Some(hint), Some(FORWARD));
        let far_receiver = player(1, Vector2::new(2000.0, 1500.0), 0.0);
        let w = world_at(
            vec![player(0, STAGED, 0.0), far_receiver],
            Some(BALL),
            0.0,
        );
        let out = c.tick(&ctx(&w, &tc));
        assert_eq!(c.phase(), PassPhase::Stage);
        assert!(
            matches!(out.receiver_input.kicker, KickerControlInput::ReflexKick),
            "one-timer receiver must hold the reflex armed while staging"
        );
        // Facing the forward target: receiver at (2000,1500), FORWARD at
        // (4000,0) → yaw well below zero (down-right), never toward the passer
        // (which would be ~180°).
        let yaw = out.receiver_input.yaw.expect("receiver yaw commanded");
        assert!(yaw.radians() < 0.0 && yaw.radians() > -std::f64::consts::FRAC_PI_2);
    }

    /// Drive a one-timer pass through a real departure into `Flight`.
    fn into_flight_forward(tc: &TeamContext) -> PassCoordinator {
        let mut c =
            PassCoordinator::new(PlayerId::new(0), PlayerId::new(1), None, Some(FORWARD));
        let receiver = player(1, RECEIVER_POS, 0.0);
        let w = world_at(
            vec![player(0, STAGED, 0.0), receiver.clone()],
            Some(BALL),
            0.0,
        );
        c.tick(&ctx(&w, tc));
        assert_eq!(c.phase(), PassPhase::Release);
        let mut w = world(
            vec![player(0, STAGED, 0.0), receiver],
            Some(Vector2::new(60.0, 0.0)),
            0.016,
        );
        w.t_received = 0.032;
        if let Some(b) = w.ball.as_mut() {
            b.velocity = Vector3::new(2500.0, 0.0, 0.0);
        }
        c.tick(&ctx(&w, tc));
        assert_eq!(c.phase(), PassPhase::Flight);
        c
    }

    #[test]
    fn one_timer_deflection_completes_forwarded() {
        let tc = team_ctx();
        let mut c = into_flight_forward(&tc);
        let receiver = player(1, RECEIVER_POS, 0.0);
        let mk = |ball: Vector2, vel: Vector2, t: f64| {
            let mut w = world(
                vec![player(0, STAGED, 0.0), receiver.clone()],
                Some(ball),
                0.016,
            );
            w.t_received = t;
            if let Some(b) = w.ball.as_mut() {
                b.velocity = Vector3::new(vel.x, vel.y, 0.0);
            }
            w
        };
        // Ball arrives at the mouth (arms the was-near edge + the endgame gate).
        let out = c.tick(&ctx(&mk(Vector2::new(1950.0, 0.0), Vector2::new(2500.0, 0.0), 0.8), &tc));
        assert_eq!(out.status, SkillStatus::Running);
        // Reflex fired: ball speeding away past the receiver — this is 1.2×
        // past the intercept along the pass line, which without the endgame
        // gate would have been declared ReceiverMissed. It must be the
        // verified deflection instead.
        let out = c.tick(&ctx(&mk(Vector2::new(2400.0, 0.0), Vector2::new(2500.0, 0.0), 0.85), &tc));
        assert_eq!(out.status, SkillStatus::Succeeded);
        assert!(matches!(
            c.result(),
            Some(PassResult::Success {
                forwarded: true,
                ..
            })
        ));
    }

    #[test]
    fn one_timer_dud_catch_succeeds_unforwarded() {
        // The reflex duds but the ball seats on the receiver's dribbler: still a
        // completed pass, reported as a catch (forwarded: false).
        let tc = team_ctx();
        let mut c = into_flight_forward(&tc);
        let mut receiver = player(1, RECEIVER_POS, 0.0);
        receiver.has_ball = true;
        let mut w = world(
            vec![player(0, STAGED, 0.0), receiver],
            Some(RECEIVER_POS),
            0.016,
        );
        w.t_received = 0.8;
        let out = c.tick(&ctx(&w, &tc));
        assert_eq!(out.status, SkillStatus::Succeeded);
        assert!(matches!(
            c.result(),
            Some(PassResult::Success {
                forwarded: false,
                ..
            })
        ));
    }
}
