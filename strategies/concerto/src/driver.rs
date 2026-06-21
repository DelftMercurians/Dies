//! Driver — realizes the current waypoint via skill commands and reports status.
//!
//! The planner decides *what* should happen to the ball; the Driver makes it
//! happen for the active robot and reports Ongoing / Succeeded / Failed(reason)
//! so the orchestrator can replan. Failure reasons are rich enough for the
//! planner to react differently (e.g. NoProgress → try another robot).

use dies_strategy_api::prelude::*;
use dies_strategy_api::World;

use crate::config;
use crate::planner::Waypoint;

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
    Approach,
    Pickup,
    Dribble,
    Shoot,
    // PassExec / PassSettle added in the passing milestone.
}

pub struct Driver {
    active_robot: Option<PlayerId>,
    waypoint: Option<Waypoint>,
    phase: Phase,
    phase_entered: f64,
    last_status: WaypointStatus,
    /// Ball position when the current waypoint started (for ball-moved detection).
    engage_ball_pos: Option<Vector2>,
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
            new_active: None,
        }
    }

    pub fn status(&self) -> &WaypointStatus {
        &self.last_status
    }

    pub fn active_robot_id(&self) -> Option<PlayerId> {
        self.active_robot
    }

    /// The area Formation should staff with a receiver (Pass waypoints only).
    /// Always `None` in v1; the formation reads this to build the receiver role.
    pub fn plan_context_area(&self) -> Option<Vector2> {
        match &self.waypoint {
            Some(Waypoint::Pass { target_area, .. }) => Some(*target_area),
            _ => None,
        }
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
        self.new_active = None;
    }

    /// Begin executing a waypoint with the given active robot.
    pub fn set_waypoint(&mut self, waypoint: Waypoint, active_robot: PlayerId, now: f64) {
        self.phase = match &waypoint {
            Waypoint::Capture { .. } => Phase::Approach,
            Waypoint::Dribble { .. } => Phase::Dribble,
            Waypoint::Shoot { .. } => Phase::Shoot,
            Waypoint::Pass { .. } => Phase::Idle, // passing milestone wires this
        };
        self.waypoint = Some(waypoint);
        self.active_robot = Some(active_robot);
        self.phase_entered = now;
        self.last_status = WaypointStatus::Ongoing;
        self.engage_ball_pos = None;
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

        let status = match (&waypoint, self.phase) {
            // ── Capture: Approach ───────────────────────────────────────
            (Waypoint::Capture { .. }, Phase::Approach) => {
                player.go_to(ball_pos).facing(ball_pos);
                player.set_role("capturing");
                if (player_pos - ball_pos).norm() < config::CAPTURE_PICKUP_DIST {
                    self.enter(Phase::Pickup, now);
                    WaypointStatus::Ongoing
                } else if elapsed > config::APPROACH_TIMEOUT {
                    WaypointStatus::Failed(FailReason::Timeout)
                } else {
                    WaypointStatus::Ongoing
                }
            }

            // ── Capture: Pickup ─────────────────────────────────────────
            (Waypoint::Capture { .. }, Phase::Pickup) => {
                let heading = heading_toward(ball_pos, opp_goal);
                player.pickup_ball(heading);
                player.set_role("picking");
                match skill_status {
                    SkillStatus::Succeeded => WaypointStatus::Succeeded,
                    SkillStatus::Failed => WaypointStatus::Failed(FailReason::SkillFailed),
                    _ if elapsed > config::PICKUP_TIMEOUT => {
                        WaypointStatus::Failed(FailReason::Timeout)
                    }
                    _ => WaypointStatus::Ongoing,
                }
            }

            // ── Dribble ─────────────────────────────────────────────────
            (Waypoint::Dribble { target_area }, Phase::Dribble) => {
                let heading = heading_toward(*target_area, opp_goal);
                player.dribble_to(*target_area, heading);
                player.set_role("dribbling");
                if has_ball && (player_pos - *target_area).norm() < config::DRIBBLE_ARRIVE_DIST {
                    WaypointStatus::Succeeded
                } else if skill_status == SkillStatus::Failed {
                    WaypointStatus::Failed(FailReason::PossessionLost)
                } else if elapsed > config::DRIBBLE_TIMEOUT {
                    WaypointStatus::Failed(FailReason::Timeout)
                } else {
                    WaypointStatus::Ongoing
                }
            }

            // ── Shoot ───────────────────────────────────────────────────
            (Waypoint::Shoot { target }, Phase::Shoot) => {
                player.reflex_shoot(*target);
                player.set_role("shooting");
                match skill_status {
                    SkillStatus::Succeeded => WaypointStatus::Succeeded,
                    SkillStatus::Failed => WaypointStatus::Failed(FailReason::SkillFailed),
                    _ if elapsed > config::SHOOT_TIMEOUT => {
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

    fn enter(&mut self, phase: Phase, now: f64) {
        self.phase = phase;
        self.phase_entered = now;
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
