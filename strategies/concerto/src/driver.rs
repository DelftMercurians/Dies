//! Driver — realizes the current waypoint via skill commands and reports status.
//!
//! The planner decides *where the ball should go*; the Driver makes it happen
//! for the active robot and reports Ongoing / Succeeded / Failed(reason) so the
//! orchestrator can replan. The reason is informational (debug/logging); any
//! failure simply triggers a fresh replan for the same most-eligible robot.

use std::sync::atomic::{AtomicU64, Ordering};

use dies_strategy_api::prelude::*;
use dies_strategy_api::{PassFailure, PassResult, World};

use crate::config;
use crate::planner::{Engagement, PlanContext, Waypoint};

/// Outcome of the current waypoint this tick.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WaypointStatus {
    Ongoing,
    Succeeded,
    Failed(FailReason),
}

/// Why a waypoint failed. Informational only (debug/logging) — every failure drives
/// the same response: re-derive a fresh plan for the same most-eligible robot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailReason {
    Timeout,
    BallMoved,
    PossessionLost,
    SkillFailed,
}

/// Internal phase of the active robot's execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Idle,
    /// One strike-through ball episode via `player.handle_ball(Strike)`: the
    /// skill stages behind the ball, drives through, and reflex-kicks. The
    /// target is recommanded every frame, so a replan that only moves the aim is
    /// a live param swap inside the running skill — never a teardown.
    Strike,
    /// Block an opponent holder: track a point directly in front of its dribbler
    /// each tick. Never terminal — ends via replan edges (possession change).
    Harass,
    /// A coordinated pass is in flight; the framework's `PassCoordinator` owns
    /// both robots. The driver just keeps the pass alive and watches the result.
    PassExec,
}

pub struct Driver {
    active_robot: Option<PlayerId>,
    waypoint: Option<Waypoint>,
    phase: Phase,
    last_status: WaypointStatus,
    /// Ball position when the current waypoint started (for ball-moved detection).
    engage_ball_pos: Option<Vector2>,
    /// Whether the post-kick latch reset has been handled for the current `Strike`
    /// waypoint. A successful `HandleBall` kick is one-shot and latches the
    /// executor slot; a fresh episode clears it for one frame before re-commanding.
    handle_reset_done: bool,
    /// Next active robot hint (set by a completed pass; consumed by orchestrator).
    new_active: Option<PlayerId>,
    /// Sequence number of the current `Pass` waypoint (stable across its frames,
    /// fresh per pass decision) — lets the executor tell a stale re-send of a
    /// finished pass from a new pass between the same pair.
    pass_seq: u64,
}

/// Global per-process pass counter. A plain counter (not time-based) so
/// faster-than-real self-play stays deterministic. Starts at 1: seq 0 is the
/// protocol's "no nonce" sentinel.
static NEXT_PASS_SEQ: AtomicU64 = AtomicU64::new(1);

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
            last_status: WaypointStatus::Ongoing,
            engage_ball_pos: None,
            handle_reset_done: false,
            new_active: None,
            pass_seq: 0,
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

    /// The live pass being executed, if any: `(passer, receiver, target_area)`.
    /// The orchestrator's pass follow-through pin is armed from this.
    pub fn current_pass(&self) -> Option<(PlayerId, PlayerId, Vector2)> {
        match &self.waypoint {
            Some(Waypoint::Pass {
                passer,
                receiver,
                target_area,
                ..
            }) => Some((*passer, *receiver, *target_area)),
            _ => None,
        }
    }

    /// Coverage the plan-controlled robots provide this tick, for Formation's
    /// coverage accounting. Empty when no waypoint is active.
    pub fn plan_context(&self) -> PlanContext {
        let wp = match &self.waypoint {
            Some(wp) => wp,
            None => return PlanContext::default(),
        };
        let mut engagements = Vec::new();
        // Any ball-working waypoint occupies the ball area.
        if let Some(at) = self.engage_ball_pos {
            engagements.push(Engagement::BallContest { at });
        }
        match wp {
            Waypoint::Harass { opponent } => {
                engagements.push(Engagement::Engaging {
                    opponent: *opponent,
                });
            }
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
        self.handle_reset_done = false;
        self.new_active = None;
    }

    /// Begin executing a waypoint with the given active robot.
    pub fn set_waypoint(&mut self, waypoint: Waypoint, active_robot: PlayerId) {
        self.phase = match &waypoint {
            Waypoint::Strike { .. } => Phase::Strike,
            Waypoint::Harass { .. } => Phase::Harass,
            Waypoint::Pass { .. } => Phase::PassExec,
        };
        if matches!(waypoint, Waypoint::Pass { .. }) {
            // Every Pass waypoint is a fresh pass decision — even an immediate
            // same-pair retry gets a new seq.
            self.pass_seq = NEXT_PASS_SEQ.fetch_add(1, Ordering::Relaxed);
        }
        self.waypoint = Some(waypoint);
        self.active_robot = Some(active_robot);
        self.last_status = WaypointStatus::Ongoing;
        self.engage_ball_pos = None;
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

        // ── Pass: delegate to the joint coordinator ─────────────────────────
        // A pass is a degenerate waypoint: keep it alive (idempotent) and watch
        // the typed result. The framework's PassCoordinator owns both robots and
        // every phase (Stage→Release→Flight→Settle). We deliberately skip the
        // ball-moved guard below — during a pass the ball is *meant* to move.
        if let Waypoint::Pass {
            passer,
            receiver,
            target_area,
            forward_to,
        } = waypoint
        {
            let mut pass = ctx
                .pass(passer, receiver)
                .target_hint(target_area)
                .seq(self.pass_seq);
            if let Some(aim) = forward_to {
                pass = pass.forward_to(aim);
            }
            drop(pass);
            if let Some(p) = ctx.player(passer) {
                p.set_role("passing");
            }
            if let Some(r) = ctx.player(receiver) {
                r.set_role(if forward_to.is_some() {
                    "one_timer"
                } else {
                    "receiving"
                });
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

        // ── Harass: track a block point in front of the holder's dribbler ───
        // Positional, never terminal (the possession edge replans it away), and
        // exempt from the ball-moved guard — the ball moves with the holder.
        if let Waypoint::Harass { opponent } = waypoint {
            let Some(holder) = world.opp_player(opponent) else {
                // Holder vanished (sidelined / tracking lost): replan.
                return self.finish(WaypointStatus::Failed(FailReason::SkillFailed));
            };
            let Some(player) = ctx.player(active_id) else {
                return self.finish(WaypointStatus::Failed(FailReason::SkillFailed));
            };
            // Directly in front of the dribbler: on the holder's facing axis, a
            // contact-distance standoff. ORCA off — it would refuse to press
            // this close; the block point itself keeps us just off the bumper.
            let facing = holder.heading.to_vector();
            let block = holder.position + facing * config::HARASS_STANDOFF;
            player
                .go_to(block)
                .facing(holder.position)
                .avoid_robots(false);
            player.set_role("harassing");
            self.last_status = WaypointStatus::Ongoing;
            return WaypointStatus::Ongoing;
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
        let skill_status = player.skill_status();
        let has_ball = player.has_ball();

        // Post-kick latch reset: a successful one-shot `HandleBall` kick latches the
        // executor slot, so a *fresh* `Strike` episode must clear it for one frame
        // before re-commanding (else the new command is ignored). This only fires
        // when the previous skill actually succeeded — a live target update on a
        // Running strike stays in place, which is the seam fix.
        if self.phase == Phase::Strike && !self.handle_reset_done {
            self.handle_reset_done = true;
            if skill_status == SkillStatus::Succeeded {
                player.stop();
                self.last_status = WaypointStatus::Ongoing;
                return WaypointStatus::Ongoing;
            }
        }

        let status = match (&waypoint, self.phase) {
            // ── Strike-through kick ─────────────────────────────────────
            // The skill owns the whole episode (stage behind the ball, commit,
            // drive through, reflex kick, verify) and all execution timing; the
            // driver just maps its terminal status.
            (Waypoint::Strike { target }, Phase::Strike) => {
                player.handle_ball(
                    BallAction::Strike {
                        target: *target,
                        acquire_first: false,
                    },
                    AcquirePosition::Default,
                );
                player.set_role("kicking");
                match skill_status {
                    SkillStatus::Succeeded => WaypointStatus::Succeeded,
                    SkillStatus::Failed => WaypointStatus::Failed(if has_ball {
                        FailReason::SkillFailed
                    } else {
                        FailReason::PossessionLost
                    }),
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

/// Map a typed pass failure onto the driver's coarse `FailReason` (informational).
fn map_pass_failure(reason: PassFailure) -> FailReason {
    match reason {
        PassFailure::Timeout => FailReason::Timeout,
        // Ball ended up loose / on an opponent — possession is gone.
        PassFailure::ReceiverMissed | PassFailure::Intercepted | PassFailure::StoppedShort => {
            FailReason::PossessionLost
        }
        // The pass never really got going (passer lost it, a side dropped out,
        // or the one-timer geometry was rejected pre-commit).
        PassFailure::BallLost
        | PassFailure::PartnerLeft
        | PassFailure::Cancelled
        | PassFailure::KickFailed
        | PassFailure::InfeasibleGeometry => FailReason::SkillFailed,
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
            pre_stage: false,
            our_keeper_id: None,
            freekick_kicker: None,
            double_touch_barred: None,
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
            forward_to: None,
        }
    }

    #[test]
    fn pass_succeeds_and_hands_off_to_receiver() {
        let world = World::new(snap());
        let mut ctx = ctx_with(Some(PassResult::Success {
            receiver: PlayerId::new(3),
            forwarded: false,
        }));
        let mut d = Driver::new();
        d.set_waypoint(pass_waypoint(), PlayerId::new(2));

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
        d.set_waypoint(pass_waypoint(), PlayerId::new(2));

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
        d.set_waypoint(pass_waypoint(), PlayerId::new(2));

        assert_eq!(d.update(&world, &mut ctx), WaypointStatus::Ongoing);
        assert!(d.is_passing());
        assert_eq!(d.plan_slots(), vec![PlayerId::new(2), PlayerId::new(3)]);
        assert_eq!(
            d.current_pass(),
            Some((
                PlayerId::new(2),
                PlayerId::new(3),
                Vector2::new(2500.0, 0.0)
            ))
        );
    }
}
