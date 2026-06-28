//! Driver — realizes the current waypoint via skill commands and reports status.
//!
//! The planner decides *what* should happen to the ball; the Driver makes it
//! happen for the active robot and reports Ongoing / Succeeded / Failed(reason)
//! so the orchestrator can replan. Failure reasons are rich enough for the
//! planner to react differently (e.g. NoProgress → try another robot).

use dies_strategy_api::prelude::*;
use dies_strategy_api::{PassFailure, PassResult, World};

use crate::config;
use crate::geometry;
use crate::planner::{Engagement, PlanContext, Waypoint};

/// Outcome of the current waypoint this tick.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WaypointStatus {
    Ongoing,
    Succeeded,
    Failed(FailReason),
}

/// Why a waypoint failed. Distinct reasons let the planner respond differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailReason {
    Timeout,
    BallMoved,
    PossessionLost,
    /// Approach made no headway (unlocks the planner's anti-loop). Emitted in M3.
    NoProgress,
    SkillFailed,
    /// Pass had no viable receiver. Pass seam — emitted in the passing milestone.
    #[allow(dead_code)]
    NoReceiver,
}

/// Internal phase of the active robot's execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Idle,
    /// Unified acquire + carry + shoot/strike via `player.handle_ball`. One phase
    /// for the whole ball-handling episode — the acquire→act transition is a live
    /// param swap inside the skill, so the driver never tears it down.
    Handle,
    /// Strip a held ball off an opponent — press the ball and rotate to peel it
    /// loose (the `snatch` skill). Distinct from `Handle` because it has a simpler
    /// completion contract (succeed when the opponent loses the ball) and its own
    /// timeout, not the pickup tiered-commit timer.
    Snatch,
    /// A coordinated pass is in flight; the framework's `PassCoordinator` owns
    /// both robots. The driver just keeps the pass alive and watches the result.
    PassExec,
}

pub struct Driver {
    active_robot: Option<PlayerId>,
    waypoint: Option<Waypoint>,
    phase: Phase,
    phase_entered: f64,
    last_status: WaypointStatus,
    /// Ball position when the current waypoint started (for ball-moved detection).
    engage_ball_pos: Option<Vector2>,
    /// When the active robot first entered the ball's committing range during a
    /// capture (`< CAPTURE_PICKUP_DIST`). Drives the tighter pickup timeout;
    /// cleared whenever it backs out of range so the timer only runs while close.
    pickup_near_since: Option<f64>,
    /// When the active robot last secured the ball during a `Handle` waypoint
    /// (the `!has_ball` → `has_ball` edge). Arms the per-action timeout; cleared
    /// while we don't hold the ball so it re-arms on the next acquisition.
    act_started: Option<f64>,
    /// Whether the post-kick latch reset has been handled for the current `Handle`
    /// waypoint. A successful `HandleBall` kick is one-shot and latches the
    /// executor slot; a fresh episode clears it for one frame before re-commanding.
    handle_reset_done: bool,
    /// Next active robot hint (set by a completed pass; consumed by orchestrator).
    new_active: Option<PlayerId>,
}

impl Default for Driver {
    fn default() -> Self {
        Self::new()
    }
}

impl Driver {
    pub fn new() -> Self {
        Self {
            active_robot: None,
            waypoint: None,
            phase: Phase::Idle,
            phase_entered: 0.0,
            last_status: WaypointStatus::Ongoing,
            engage_ball_pos: None,
            pickup_near_since: None,
            act_started: None,
            handle_reset_done: false,
            new_active: None,
        }
    }

    pub fn status(&self) -> &WaypointStatus {
        &self.last_status
    }

    pub fn active_robot_id(&self) -> Option<PlayerId> {
        self.active_robot
    }

    /// Robots the plan currently controls (excluded from Formation). One for most
    /// waypoints; both passer and receiver for a `Pass` — the latter is a hard
    /// requirement so Formation never issues a `go_to` that would cancel the pass.
    pub fn plan_slots(&self) -> Vec<PlayerId> {
        match &self.waypoint {
            Some(Waypoint::Pass {
                passer, receiver, ..
            }) => vec![*passer, *receiver],
            _ => self.active_robot.into_iter().collect(),
        }
    }

    /// True while a pass is mid-flight. The orchestrator uses this to suppress
    /// soft replans: possession flits We→Loose→We during a pass, and the
    /// coordinator (not the planner) owns abort/failure.
    pub fn is_passing(&self) -> bool {
        matches!(self.waypoint, Some(Waypoint::Pass { .. }))
            && self.last_status == WaypointStatus::Ongoing
    }

    /// Coverage the plan-controlled robots provide this tick, for Formation's
    /// coverage accounting. Empty when no waypoint is active.
    pub fn plan_context(&self) -> PlanContext {
        let wp = match &self.waypoint {
            Some(wp) => wp,
            None => return PlanContext::default(),
        };
        let mut engagements = Vec::new();
        // Any ball-holding/contesting waypoint occupies the ball area.
        if let Some(at) = self.engage_ball_pos {
            engagements.push(Engagement::BallContest { at });
        }
        match wp {
            Waypoint::Steal { from } => engagements.push(Engagement::Engaging { opponent: *from }),
            Waypoint::Pass { target_area, .. } => {
                engagements.push(Engagement::ReceiveArea { at: *target_area })
            }
            _ => {}
        }
        PlanContext { engagements }
    }

    /// Consume the next-active hint (set when a pass succeeds). `None` in v1.
    pub fn take_new_active(&mut self) -> Option<PlayerId> {
        self.new_active.take()
    }

    /// Reset to idle, dropping any active waypoint.
    pub fn clear(&mut self) {
        self.active_robot = None;
        self.waypoint = None;
        self.phase = Phase::Idle;
        self.last_status = WaypointStatus::Ongoing;
        self.engage_ball_pos = None;
        self.pickup_near_since = None;
        self.act_started = None;
        self.handle_reset_done = false;
        self.new_active = None;
    }

    /// Begin executing a waypoint with the given active robot.
    pub fn set_waypoint(&mut self, waypoint: Waypoint, active_robot: PlayerId, now: f64) {
        self.phase = match &waypoint {
            // Stripping a held ball is a snatch; everything else (acquire + carry +
            // shoot/strike) is one unified ball-handling episode.
            Waypoint::Steal { .. } => Phase::Snatch,
            Waypoint::Handle { .. } => Phase::Handle,
            Waypoint::Pass { .. } => Phase::PassExec,
        };
        self.waypoint = Some(waypoint);
        self.active_robot = Some(active_robot);
        self.phase_entered = now;
        self.last_status = WaypointStatus::Ongoing;
        self.engage_ball_pos = None;
        self.pickup_near_since = None;
        self.act_started = None;
        self.handle_reset_done = false;
        self.new_active = None;
    }

    /// Tick the active robot's skill; returns the waypoint status.
    pub fn update(&mut self, world: &World, ctx: &mut TeamContext) -> WaypointStatus {
        let waypoint = match self.waypoint.clone() {
            Some(w) => w,
            None => return self.last_status.clone(),
        };
        let active_id = match self.active_robot {
            Some(id) => id,
            None => return self.last_status.clone(),
        };

        let ball_pos = world
            .ball_position()
            .or(self.engage_ball_pos)
            .unwrap_or_else(|| Vector2::new(0.0, 0.0));
        if self.engage_ball_pos.is_none() {
            self.engage_ball_pos = Some(ball_pos);
        }
        let now = world.timestamp();
        let opp_goal = world.opp_goal_center();

        // ── Pass: delegate to the joint coordinator ─────────────────────────
        // A pass is a degenerate waypoint: keep it alive (idempotent) and watch
        // the typed result. The framework's PassCoordinator owns both robots and
        // every phase (Secure→Setup→Commit→Flight→Settle). We deliberately skip
        // the ball-moved guard below — during a pass the ball is *meant* to move.
        if let Waypoint::Pass {
            passer,
            receiver,
            target_area,
        } = waypoint
        {
            ctx.pass(passer, receiver).target_hint(target_area);
            if let Some(p) = ctx.player(passer) {
                p.set_role("passing");
            }
            if let Some(r) = ctx.player(receiver) {
                r.set_role("receiving");
            }
            let status = match ctx.pass_result(passer) {
                Some(PassResult::Success { .. }) => {
                    self.new_active = Some(receiver);
                    WaypointStatus::Succeeded
                }
                Some(PassResult::Failure { reason, .. }) => {
                    WaypointStatus::Failed(map_pass_failure(*reason))
                }
                None => WaypointStatus::Ongoing,
            };
            if matches!(
                status,
                WaypointStatus::Succeeded | WaypointStatus::Failed(_)
            ) {
                return self.finish(status);
            }
            self.last_status = status.clone();
            return status;
        }

        // ── Global failure: ball left the engagement area, and not ours ─────
        if let (Some(engage), Some(bp)) = (self.engage_ball_pos, world.ball_position()) {
            let ours = world
                .own_player(active_id)
                .map(|p| p.has_ball)
                .unwrap_or(false);
            if !ours && (bp - engage).norm() > config::BALL_MOVED_DIST {
                return self.finish(WaypointStatus::Failed(FailReason::BallMoved));
            }
        }

        let player = match ctx.player(active_id) {
            Some(p) => p,
            None => return self.finish(WaypointStatus::Failed(FailReason::SkillFailed)),
        };

        // Snapshot immutable state before issuing mutable commands.
        let player_pos = player.position();
        let skill_status = player.skill_status();
        let has_ball = player.has_ball();
        let elapsed = now - self.phase_entered;

        // Post-kick latch reset: a successful one-shot `HandleBall` kick latches the
        // executor slot, so a *fresh* `Handle` episode must clear it for one frame
        // before re-commanding (else the new command is ignored). This only fires
        // when the previous skill actually succeeded — a `Hold → Shoot` live swap
        // stays `Running` and updates in place, which is the seam fix.
        if self.phase == Phase::Handle && !self.handle_reset_done {
            self.handle_reset_done = true;
            if skill_status == SkillStatus::Succeeded {
                player.stop();
                self.last_status = WaypointStatus::Ongoing;
                return WaypointStatus::Ongoing;
            }
        }

        let status = match (&waypoint, self.phase) {
            // ── Handle (acquire + carry + shoot/strike) ─────────────────
            // One unified skill owns the whole episode: it stages behind the ball,
            // commits the capture, then (for Shoot) orbits to aim and kicks — all
            // without a teardown, so the capture→aim seam keeps its state. The
            // action is recommanded every frame; swapping it (e.g. once we secure
            // the ball) is a live param update on the same skill instance.
            (Waypoint::Handle { action, rescue }, Phase::Handle) => {
                // Acquire exit-bias, computed live so a contested/boundary capture
                // scoops the ball to the open side. Only used while `!has_ball`.
                let approach = matches!(action, BallAction::Hold { .. })
                    .then(|| pickup_heading(world, active_id, ball_pos, opp_goal, *rescue));
                player.handle_ball(*action, approach);
                player.set_role(match action {
                    BallAction::Shoot { .. } => "shooting",
                    BallAction::Strike { .. } => "kicking",
                    BallAction::Carry { .. } => "carrying",
                    BallAction::Hold { .. } if *rescue => "rescuing",
                    BallAction::Hold { .. } => "capturing",
                });

                // Timeout: while we don't hold the ball, the tiered approach/pickup
                // timer (generous traversing, tight once committing); once we hold
                // it, the per-action timer armed at the acquisition edge.
                let timed_out = if has_ball {
                    self.pickup_near_since = None;
                    let started = *self.act_started.get_or_insert(now);
                    let budget = match action {
                        BallAction::Shoot { .. } | BallAction::Strike { .. } => {
                            config::SHOOT_TIMEOUT
                        }
                        BallAction::Carry { .. } => config::DRIBBLE_TIMEOUT,
                        BallAction::Hold { .. } => f64::INFINITY,
                    };
                    now - started > budget
                } else {
                    self.act_started = None;
                    let near = (player_pos - ball_pos).norm() < config::CAPTURE_PICKUP_DIST;
                    if near {
                        self.pickup_near_since.get_or_insert(now);
                    } else {
                        self.pickup_near_since = None;
                    }
                    match self.pickup_near_since {
                        Some(since) => now - since > config::PICKUP_TIMEOUT,
                        None => elapsed > config::APPROACH_TIMEOUT,
                    }
                };

                // Completion is action-specific: Shoot/Strike succeed on the kick
                // (the skill verifies the ball left); Carry succeeds on arrival
                // (driver-owned, as the old Dribble waypoint did); Hold never
                // self-completes — the planner replans it away on the possession flip.
                match action {
                    BallAction::Carry { to, .. } => {
                        if skill_status == SkillStatus::Failed {
                            WaypointStatus::Failed(FailReason::PossessionLost)
                        } else if has_ball
                            && (player_pos - *to).norm() < config::DRIBBLE_ARRIVE_DIST
                        {
                            WaypointStatus::Succeeded
                        } else if timed_out {
                            WaypointStatus::Failed(FailReason::Timeout)
                        } else {
                            WaypointStatus::Ongoing
                        }
                    }
                    _ => match skill_status {
                        SkillStatus::Succeeded => WaypointStatus::Succeeded,
                        SkillStatus::Failed => WaypointStatus::Failed(if has_ball {
                            FailReason::SkillFailed
                        } else {
                            FailReason::PossessionLost
                        }),
                        _ if timed_out => WaypointStatus::Failed(FailReason::Timeout),
                        _ => WaypointStatus::Ongoing,
                    },
                }
            }

            // ── Snatch (strip a held ball) ──────────────────────────────
            // The opponent is holding the ball on its dribbler. Press our dribbler
            // against the ball and rotate to peel it loose, aimed to a safe clear
            // away from the holder. Succeeds when the opponent loses the ball — the
            // ball then goes loose and the next replan captures it normally.
            (Waypoint::Steal { from }, Phase::Snatch) => {
                let release = snatch_release_hint(world, *from, ball_pos);
                player.snatch(release);
                player.set_role("snatching");
                match skill_status {
                    SkillStatus::Succeeded => WaypointStatus::Succeeded,
                    SkillStatus::Failed => WaypointStatus::Failed(FailReason::SkillFailed),
                    _ if elapsed > config::SNATCH_TIMEOUT => {
                        WaypointStatus::Failed(FailReason::Timeout)
                    }
                    _ => WaypointStatus::Ongoing,
                }
            }

            // Invalid combination — stay ongoing (defensive; should not happen).
            _ => WaypointStatus::Ongoing,
        };

        if matches!(
            status,
            WaypointStatus::Succeeded | WaypointStatus::Failed(_)
        ) {
            return self.finish(status);
        }
        self.last_status = status.clone();
        status
    }

    fn finish(&mut self, status: WaypointStatus) -> WaypointStatus {
        self.last_status = status.clone();
        self.waypoint = None;
        self.phase = Phase::Idle;
        status
    }
}

/// Heading pointing from `from` toward `to`.
fn heading_toward(from: Vector2, to: Vector2) -> Angle {
    let d = to - from;
    Angle::from_radians(d.y.atan2(d.x))
}

/// How far ahead (mm) the snatch release point is placed. Only its *direction*
/// from the ball matters (the skill picks which way to peel from it), so the
/// magnitude is arbitrary — just large enough to be a stable direction.
const SNATCH_RELEASE_AIM: f64 = 1000.0;

/// Where to knock a stripped ball: a safe clear *away from the holder*. The peel
/// moves the ball perpendicular to the holder's facing, so the only real choice
/// is which lateral side — pick the one pointing away from our own goal (upfield)
/// so a strip never squirts the ball back toward danger. Returns `None` (skill's
/// own safe default) if the holder can't be located.
fn snatch_release_hint(world: &World, holder: PlayerId, ball: Vector2) -> Option<Vector2> {
    let holder_pos = world.opp_player(holder)?.position;
    let facing = ball - holder_pos;
    if facing.norm() < 1e-3 {
        return None;
    }
    let facing = facing.normalize();
    // The two lateral (perpendicular) escape directions.
    let lateral = Vector2::new(-facing.y, facing.x);
    let away_from_goal = (ball - world.own_goal_center()).normalize();
    let side = if lateral.dot(&away_from_goal) >= 0.0 {
        lateral
    } else {
        -lateral
    };
    Some(ball + side * SNATCH_RELEASE_AIM)
}

/// How strongly the off-axis escape direction bends the contested pickup heading
/// away from the straight-at-goal aim (0 = aim at goal, larger = more sideways).
const CONTEST_HEADING_BIAS: f64 = 0.7;

/// Capture heading for the pickup phase. With no contest, aim straight at the
/// opponent goal (unchanged). Under contest, blend the goal direction with the
/// off-axis [`geometry::escape_direction`] so we scoop the loose ball to our open
/// side instead of head-butting the opponent straight over it.
fn pickup_heading(
    world: &World,
    robot: PlayerId,
    ball: Vector2,
    opp_goal: Vector2,
    rescue: bool,
) -> Angle {
    // Boundary rescue takes priority: dribble the ball back into the field rather
    // than along/over the line. Falls through to the normal heading if the ball
    // has since drifted off the boundary.
    if rescue {
        if let Some(h) = geometry::boundary_rescue_heading(
            ball,
            opp_goal,
            world.field_length() / 2.0,
            world.field_width() / 2.0,
            config::BOUNDARY_RESCUE_MARGIN,
            config::RESCUE_GOAL_BIAS,
        ) {
            return h;
        }
    }
    if world.ball_contest().is_none() {
        return heading_toward(ball, opp_goal);
    }
    let robot_pos = world.own_player(robot).map(|p| p.position).unwrap_or(ball);
    let presser = world
        .principal_presser()
        .and_then(|id| world.opp_player(id))
        .map(|p| p.position)
        .unwrap_or(ball);
    let escape = geometry::escape_direction(
        robot_pos,
        presser,
        ball,
        world.opp_players(),
        world.field_width() / 2.0,
    );
    let to_goal = opp_goal - ball;
    let to_goal = if to_goal.norm() > 1e-6 {
        to_goal / to_goal.norm()
    } else {
        Vector2::new(1.0, 0.0)
    };
    let blended = to_goal + escape * CONTEST_HEADING_BIAS;
    Angle::from_radians(blended.y.atan2(blended.x))
}

/// Map a typed pass failure onto the driver's coarse `FailReason`. The planner
/// only needs enough granularity to drive its anti-loop and next-action choice.
fn map_pass_failure(reason: PassFailure) -> FailReason {
    match reason {
        PassFailure::Timeout => FailReason::Timeout,
        // Ball ended up loose / on an opponent — possession is gone.
        PassFailure::ReceiverMissed | PassFailure::Intercepted | PassFailure::StoppedShort => {
            FailReason::PossessionLost
        }
        // The pass never really got going (passer lost it, or a side dropped out).
        PassFailure::BallLost | PassFailure::PartnerLeft | PassFailure::Cancelled => {
            FailReason::SkillFailed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::Waypoint;
    use dies_strategy_api::PassBallState;
    use dies_strategy_protocol::{BallState, GameState, WorldSnapshot};
    use std::collections::HashMap;

    fn snap() -> WorldSnapshot {
        WorldSnapshot {
            timestamp: 1.0,
            dt: 0.016,
            field_geom: Some(FieldGeometry::default()),
            ball: Some(BallState {
                position: Vector2::new(0.0, 0.0),
                velocity: Vector2::new(0.0, 0.0),
                detected: true,
            }),
            own_players: vec![
                PlayerState::new(
                    PlayerId::new(2),
                    Vector2::new(0.0, 0.0),
                    Vector2::new(0.0, 0.0),
                    Angle::from_radians(0.0),
                ),
                PlayerState::new(
                    PlayerId::new(3),
                    Vector2::new(2000.0, 0.0),
                    Vector2::new(0.0, 0.0),
                    Angle::from_radians(0.0),
                ),
            ],
            opp_players: vec![],
            game_state: GameState::Run,
            us_operating: true,
            our_keeper_id: None,
            freekick_kicker: None,
            possession: Possession::We(PlayerId::new(2)),
            possession_stale: false,
            ball_contest: None,
        }
    }

    fn ctx_with(result: Option<PassResult>) -> TeamContext {
        let mut pr = HashMap::new();
        if let Some(r) = result {
            pr.insert(PlayerId::new(2), r);
        }
        TeamContext::new(snap(), HashMap::new(), pr, HashMap::new())
    }

    fn pass_waypoint() -> Waypoint {
        Waypoint::Pass {
            passer: PlayerId::new(2),
            receiver: PlayerId::new(3),
            target_area: Vector2::new(2500.0, 0.0),
        }
    }

    #[test]
    fn pass_succeeds_and_hands_off_to_receiver() {
        let world = World::new(snap());
        let mut ctx = ctx_with(Some(PassResult::Success {
            receiver: PlayerId::new(3),
        }));
        let mut d = Driver::new();
        d.set_waypoint(pass_waypoint(), PlayerId::new(2), 0.0);

        assert_eq!(d.update(&world, &mut ctx), WaypointStatus::Succeeded);
        assert_eq!(d.take_new_active(), Some(PlayerId::new(3)));
    }

    #[test]
    fn pass_failure_maps_to_fail_reason() {
        let world = World::new(snap());
        let mut ctx = ctx_with(Some(PassResult::Failure {
            reason: PassFailure::Timeout,
            ball_state: PassBallState::Unknown,
        }));
        let mut d = Driver::new();
        d.set_waypoint(pass_waypoint(), PlayerId::new(2), 0.0);

        assert_eq!(
            d.update(&world, &mut ctx),
            WaypointStatus::Failed(FailReason::Timeout)
        );
    }

    #[test]
    fn pass_in_progress_reserves_both_robots() {
        let world = World::new(snap());
        let mut ctx = ctx_with(None);
        let mut d = Driver::new();
        d.set_waypoint(pass_waypoint(), PlayerId::new(2), 0.0);

        assert_eq!(d.update(&world, &mut ctx), WaypointStatus::Ongoing);
        assert!(d.is_passing());
        assert_eq!(d.plan_slots(), vec![PlayerId::new(2), PlayerId::new(3)]);
    }
}
