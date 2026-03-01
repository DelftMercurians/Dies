use dies_strategy_api::prelude::*;

use crate::planner::Waypoint;

#[derive(Debug, Clone, PartialEq)]
pub enum WaypointStatus {
    Ongoing,
    Succeeded,
    Failed(FailReason),
}

#[derive(Debug, Clone, PartialEq)]
pub enum FailReason {
    Timeout,
    BallMoved,
    PossessionLost,
    SkillFailed,
}

#[derive(Debug, Clone, PartialEq)]
enum DriverState {
    Idle,
    Approaching,
    Picking,
    Dribbling,
    Kicking,
    WaitingForBallSettle,
}

pub struct Driver {
    active_robot: Option<PlayerId>,
    current_waypoint: Option<Waypoint>,
    state: DriverState,
    state_entered_at: f64,
    last_status: WaypointStatus,
    initial_ball_pos: Option<Vector2>,
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
            current_waypoint: None,
            state: DriverState::Idle,
            state_entered_at: 0.0,
            last_status: WaypointStatus::Ongoing,
            initial_ball_pos: None,
            new_active: None,
        }
    }

    pub fn status(&self) -> &WaypointStatus {
        &self.last_status
    }

    /// Reset the driver to idle state, clearing any active waypoint.
    pub fn clear(&mut self) {
        self.active_robot = None;
        self.current_waypoint = None;
        self.state = DriverState::Idle;
        self.last_status = WaypointStatus::Ongoing;
        self.initial_ball_pos = None;
        self.new_active = None;
    }

    pub fn active_robot_id(&self) -> Option<PlayerId> {
        self.active_robot
    }

    pub fn plan_context_area(&self) -> Option<Vector2> {
        match &self.current_waypoint {
            Some(Waypoint::Pass { target_area }) => Some(*target_area),
            _ => None,
        }
    }

    pub fn new_active_robot(&mut self) -> Option<PlayerId> {
        self.new_active.take()
    }

    pub fn set_waypoint(
        &mut self,
        waypoint: Waypoint,
        active_robot: PlayerId,
        timestamp: f64,
    ) {
        self.state = match &waypoint {
            Waypoint::Capture { .. } => DriverState::Approaching,
            Waypoint::Dribble { .. } => DriverState::Dribbling,
            Waypoint::Shoot { .. } => DriverState::Kicking,
            Waypoint::Pass { .. } => DriverState::Kicking,
        };
        self.current_waypoint = Some(waypoint);
        self.active_robot = Some(active_robot);
        self.state_entered_at = timestamp;
        self.last_status = WaypointStatus::Ongoing;
        self.new_active = None;
        self.initial_ball_pos = None;
    }

    pub fn update(&mut self, world: &World, ctx: &mut TeamContext) -> WaypointStatus {
        let waypoint = match self.current_waypoint.clone() {
            Some(w) => w,
            None => return self.last_status.clone(),
        };

        let active_id = match self.active_robot {
            Some(id) => id,
            None => return self.last_status.clone(),
        };

        // Get ball position; fall back to last known if unavailable
        let ball_pos = world
            .ball_position()
            .or(self.initial_ball_pos)
            .unwrap_or(Vector2::new(0.0, 0.0));

        // Store initial ball position on first update
        if self.initial_ball_pos.is_none() {
            self.initial_ball_pos = Some(ball_pos);
        }

        let timestamp = world.timestamp();
        let opp_goal = world.opp_goal_center();

        // Get the player handle; fail if it doesn't exist
        let player = match ctx.player(active_id) {
            Some(p) => p,
            None => {
                self.last_status = WaypointStatus::Failed(FailReason::SkillFailed);
                self.current_waypoint = None;
                self.state = DriverState::Idle;
                return self.last_status.clone();
            }
        };

        // Read all immutable player state before issuing mutable commands
        let player_pos = player.position();
        let skill_status = player.skill_status();
        let has_ball = player.has_ball();

        let status = match (&waypoint, &self.state) {
            // ── Capture: Approaching ────────────────────────────────────
            (Waypoint::Capture { .. }, DriverState::Approaching) => {
                player.go_to(ball_pos).facing(ball_pos);
                player.set_role("Capturing");

                let dist_to_ball = (player_pos - ball_pos).norm();
                if dist_to_ball < 500.0 {
                    self.state = DriverState::Picking;
                    self.state_entered_at = timestamp;
                    WaypointStatus::Ongoing
                } else if timestamp - self.state_entered_at > 3.0 {
                    WaypointStatus::Failed(FailReason::Timeout)
                } else if let Some(initial) = self.initial_ball_pos {
                    if (ball_pos - initial).norm() > 2000.0 {
                        WaypointStatus::Failed(FailReason::BallMoved)
                    } else {
                        WaypointStatus::Ongoing
                    }
                } else {
                    WaypointStatus::Ongoing
                }
            }

            // ── Capture: Picking ────────────────────────────────────────
            (Waypoint::Capture { .. }, DriverState::Picking) => {
                let dir = opp_goal - ball_pos;
                let heading = Angle::from_radians(dir.y.atan2(dir.x));
                player.pickup_ball(heading);
                player.set_role("Picking");

                if skill_status == SkillStatus::Succeeded {
                    WaypointStatus::Succeeded
                } else if skill_status == SkillStatus::Failed {
                    WaypointStatus::Failed(FailReason::SkillFailed)
                } else if timestamp - self.state_entered_at > 2.0 {
                    WaypointStatus::Failed(FailReason::Timeout)
                } else {
                    WaypointStatus::Ongoing
                }
            }

            // ── Dribble ─────────────────────────────────────────────────
            (Waypoint::Dribble { target_area }, DriverState::Dribbling) => {
                let dir = opp_goal - *target_area;
                let heading = Angle::from_radians(dir.y.atan2(dir.x));
                player.dribble_to(*target_area, heading);
                player.set_role("Dribbling");

                let dist_to_target = (player_pos - *target_area).norm();
                if has_ball && dist_to_target < 500.0 {
                    WaypointStatus::Succeeded
                } else if skill_status == SkillStatus::Failed {
                    WaypointStatus::Failed(FailReason::PossessionLost)
                } else if timestamp - self.state_entered_at > 4.0 {
                    WaypointStatus::Failed(FailReason::Timeout)
                } else {
                    WaypointStatus::Ongoing
                }
            }

            // ── Shoot ───────────────────────────────────────────────────
            (Waypoint::Shoot { target }, DriverState::Kicking) => {
                player.reflex_shoot(*target);
                player.set_role("Shooting");

                if skill_status == SkillStatus::Succeeded {
                    WaypointStatus::Succeeded
                } else if skill_status == SkillStatus::Failed {
                    WaypointStatus::Failed(FailReason::SkillFailed)
                } else if timestamp - self.state_entered_at > 2.0 {
                    WaypointStatus::Failed(FailReason::Timeout)
                } else {
                    WaypointStatus::Ongoing
                }
            }

            // ── Pass: Kicking ───────────────────────────────────────────
            (Waypoint::Pass { target_area }, DriverState::Kicking) => {
                player.reflex_shoot(*target_area);
                player.set_role("Passing");

                if skill_status == SkillStatus::Succeeded {
                    self.state = DriverState::WaitingForBallSettle;
                    self.state_entered_at = timestamp;
                    WaypointStatus::Ongoing
                } else if skill_status == SkillStatus::Failed {
                    WaypointStatus::Failed(FailReason::SkillFailed)
                } else if timestamp - self.state_entered_at > 2.0 {
                    WaypointStatus::Failed(FailReason::Timeout)
                } else {
                    WaypointStatus::Ongoing
                }
            }

            // ── Pass: WaitingForBallSettle ──────────────────────────────
            (Waypoint::Pass { target_area }, DriverState::WaitingForBallSettle) => {
                player.set_role("Waiting");

                // Check if any own player received the ball near the target area
                let mut pass_succeeded = false;
                for p in world.own_players() {
                    if p.has_ball && (p.position - *target_area).norm() < 1000.0 {
                        self.new_active = Some(p.id);
                        pass_succeeded = true;
                        break;
                    }
                }

                if pass_succeeded {
                    WaypointStatus::Succeeded
                } else if timestamp - self.state_entered_at > 1.5 {
                    WaypointStatus::Failed(FailReason::PossessionLost)
                } else {
                    WaypointStatus::Ongoing
                }
            }

            // Invalid waypoint/state combination — stay ongoing
            _ => WaypointStatus::Ongoing,
        };

        self.last_status = status;

        // Terminal status: clear waypoint and go idle
        if matches!(
            self.last_status,
            WaypointStatus::Succeeded | WaypointStatus::Failed(_)
        ) {
            self.current_waypoint = None;
            self.state = DriverState::Idle;
        }

        self.last_status.clone()
    }
}
