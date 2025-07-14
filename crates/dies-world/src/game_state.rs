use std::time::Instant;

use dies_core::{
    GameState, PlayerData, TeamColor, TeamPlayerId, Vector2, Vector3, WorldData, BALL_RADIUS,
    PLAYER_RADIUS,
};
use dies_protos::ssl_gc_referee_message::{referee::Command, Referee};

use crate::BallData;

const FREEKICK_TIMEOUT_SECS: u64 = 10;

#[derive(Debug, Clone, Copy)]
pub struct GameStateTracker {
    /// Game state in vision coordinates
    game_state: GameState,
    /// Previous game state in vision coordinates  
    prev_state: GameState,
    /// New state on ball movement in vision coordinates
    new_state_movement: GameState,
    /// New state on timeout in vision coordinates
    new_state_timeout: GameState,
    init_ball_pos: Vector3,
    start: Instant,
    timeout: u64,
    is_outdated: bool,
    operator_is_blue: Option<bool>,
    last_cmd: Option<Command>,
    freekick_kicker: Option<TeamPlayerId>,
    freekick_start_time: Option<Instant>,
}

impl GameStateTracker {
    pub fn new() -> GameStateTracker {
        GameStateTracker {
            game_state: GameState::Halt,
            prev_state: GameState::Unknown,
            new_state_movement: GameState::Unknown,
            new_state_timeout: GameState::Unknown,
            init_ball_pos: Vector3::new(0.0, 0.0, 0.0),
            start: Instant::now(),
            timeout: 0,
            is_outdated: true,
            operator_is_blue: None,
            last_cmd: None,
            freekick_kicker: None,
            freekick_start_time: None,
        }
    }

    pub fn update(&mut self, data: &Referee) -> GameState {
        let command = data.command();

        if self.last_cmd == Some(command) {
            return self.game_state;
        }
        self.last_cmd = Some(command);

        let last_game_state = self.game_state;

        self.game_state = match command {
            Command::HALT => GameState::Halt,
            Command::STOP => GameState::Stop,
            Command::NORMAL_START => {
                if self.game_state == GameState::PrepareKickoff {
                    GameState::Kickoff
                } else if self.game_state == GameState::PreparePenalty {
                    GameState::Penalty
                } else {
                    GameState::Run
                }
            }
            Command::FORCE_START => GameState::Run,
            Command::PREPARE_KICKOFF_YELLOW => GameState::PrepareKickoff,
            Command::PREPARE_KICKOFF_BLUE => GameState::PrepareKickoff,
            Command::PREPARE_PENALTY_YELLOW => GameState::PreparePenalty,
            Command::PREPARE_PENALTY_BLUE => GameState::PreparePenalty,
            Command::DIRECT_FREE_YELLOW
            | Command::DIRECT_FREE_BLUE
            | Command::INDIRECT_FREE_BLUE
            | Command::INDIRECT_FREE_YELLOW => {
                self.freekick_kicker = None;
                self.freekick_start_time = Some(Instant::now());
                GameState::FreeKick
            }
            Command::TIMEOUT_YELLOW => GameState::Timeout,
            Command::TIMEOUT_BLUE => GameState::Timeout,
            Command::BALL_PLACEMENT_YELLOW | Command::BALL_PLACEMENT_BLUE => {
                if let Some(pos) = data.designated_position.as_ref() {
                    GameState::BallReplacement(Vector2::new(pos.x() as f64, pos.y() as f64))
                } else {
                    log::error!("No position for ball placement");
                    self.game_state
                }
            }
            Command::GOAL_YELLOW | Command::GOAL_BLUE => self.game_state,
        };
        dies_core::debug_string("game_state", format!("{:?}", self.game_state));

        dies_core::debug_string("last_cmd", format!("{:?}", command));

        self.operator_is_blue = match command {
            Command::PREPARE_KICKOFF_BLUE
            | Command::PREPARE_PENALTY_BLUE
            | Command::DIRECT_FREE_BLUE
            | Command::BALL_PLACEMENT_BLUE => Some(true),
            Command::PREPARE_KICKOFF_YELLOW
            | Command::PREPARE_PENALTY_YELLOW
            | Command::DIRECT_FREE_YELLOW
            | Command::BALL_PLACEMENT_YELLOW => Some(false),
            _ => self.operator_is_blue,
        };
        dies_core::debug_string(
            "operating_team",
            format!(
                "{:?}",
                if self.operator_is_blue.unwrap_or(true) {
                    "Blue"
                } else {
                    "Yellow"
                }
            ),
        );

        // Reset
        match self.game_state {
            GameState::Halt | GameState::Stop | GameState::Timeout => {
                self.operator_is_blue = None;
                self.freekick_kicker = None;
                self.freekick_start_time = None;
            }
            GameState::Run => {
                self.operator_is_blue = None;
                // Keep freekick_kicker and freekick_start_time for double touch tracking
            }
            _ => (),
        }

        self.game_state
    }

    pub fn get_operator_is_blue(&self) -> Option<bool> {
        self.operator_is_blue
    }

    pub fn start_ball_movement_check(&mut self, ball_pos: Vector3, timeout: u64) {
        if !self.is_outdated {
            return;
        }
        self.init_ball_pos = ball_pos;
        self.start = Instant::now();
        self.timeout = timeout;
        self.is_outdated = false;
        self.prev_state = self.game_state;
        self.new_state_movement = match self.game_state {
            GameState::Kickoff | GameState::FreeKick => GameState::Run,
            GameState::Penalty => GameState::PenaltyRun,
            _ => self.game_state,
        };
        self.new_state_timeout = match self.game_state {
            GameState::Kickoff | GameState::FreeKick => GameState::Run,
            GameState::Penalty => GameState::Stop,
            _ => self.game_state,
        };
    }

    pub fn update_ball_movement_check(&mut self, ball_data: Option<&BallData>) -> GameState {
        let p = self.init_ball_pos;
        if self.is_outdated || ball_data.is_none() {
            return self.game_state;
        } else if self.prev_state != self.game_state {
            self.is_outdated = true;
        } else if self.start.elapsed().as_secs() >= self.timeout {
            self.game_state = self.new_state_timeout;
            dies_core::debug_string("game_state", format!("{}", self.game_state));
            self.is_outdated = true;
        } else {
            let distance = (ball_data.unwrap().position - p).norm();
            let velocity = ball_data.unwrap().velocity.norm();
            if distance > 500.0 || velocity > 5000.0 {
                println!(
                    "ball movement detected: distance: {}, velocity: {}\ngame state {} -> {}",
                    distance, velocity, self.game_state, self.new_state_movement
                );
                self.game_state = self.new_state_movement;
                dies_core::debug_string("game_state", format!("{}", self.game_state));
                self.is_outdated = true;
            }
            return self.game_state;
        }
        self.game_state
    }

    pub fn get(&self) -> GameState {
        // Return raw game state without coordinate transformation
        self.game_state
    }

    fn is_robot_touching_ball(&self, player: &PlayerData, ball_pos: Vector2) -> bool {
        let distance = (player.position - ball_pos).norm();
        distance <= (PLAYER_RADIUS + BALL_RADIUS + 50.0)
    }

    fn find_closest_robot_to_ball(
        &self,
        world_data: &WorldData,
        ball_pos: Vector2,
    ) -> Option<TeamPlayerId> {
        let mut closest_robot: Option<TeamPlayerId> = None;
        let mut closest_distance = f64::INFINITY;

        for player in &world_data.blue_team {
            let distance = (player.position - ball_pos).norm();
            if distance < closest_distance {
                closest_distance = distance;
                closest_robot = Some(TeamPlayerId {
                    team_color: TeamColor::Blue,
                    player_id: player.id,
                });
            }
        }

        for player in &world_data.yellow_team {
            let distance = (player.position - ball_pos).norm();
            if distance < closest_distance {
                closest_distance = distance;
                closest_robot = Some(TeamPlayerId {
                    team_color: TeamColor::Yellow,
                    player_id: player.id,
                });
            }
        }

        closest_robot
    }

    pub fn get_freekick_kicker(&self) -> Option<TeamPlayerId> {
        self.freekick_kicker
    }

    pub fn update_freekick_double_touch(
        &mut self,
        world_data: Option<&WorldData>,
        ball_data: Option<&BallData>,
    ) {
        // Only track during FreeKick state or Run state with an active kicker
        if self.game_state != GameState::FreeKick
            && (self.game_state != GameState::Run || self.freekick_kicker.is_none())
        {
            return;
        }

        let (world_data, ball_data) = match (world_data, ball_data) {
            (Some(w), Some(b)) => (w, b),
            _ => return,
        };

        let ball_pos = ball_data.position.xy();

        if let Some(start_time) = self.freekick_start_time {
            if start_time.elapsed().as_secs() >= FREEKICK_TIMEOUT_SECS {
                self.freekick_kicker = None;
                self.freekick_start_time = None;
                return;
            }
        }

        if self.freekick_kicker.is_none() {
            let closest_robot = self.find_closest_robot_to_ball(world_data, ball_pos);
            if let Some(robot) = closest_robot {
                let is_touching = match robot.team_color {
                    TeamColor::Blue => world_data
                        .blue_team
                        .iter()
                        .find(|p| p.id == robot.player_id)
                        .map(|p| self.is_robot_touching_ball(p, ball_pos))
                        .unwrap_or(false),
                    TeamColor::Yellow => world_data
                        .yellow_team
                        .iter()
                        .find(|p| p.id == robot.player_id)
                        .map(|p| self.is_robot_touching_ball(p, ball_pos))
                        .unwrap_or(false),
                };

                if is_touching {
                    self.freekick_kicker = Some(robot);
                }
            }
        } else if let Some(kicker) = self.freekick_kicker {
            for player in &world_data.blue_team {
                if self.is_robot_touching_ball(player, ball_pos) {
                    let touching_robot = TeamPlayerId {
                        team_color: TeamColor::Blue,
                        player_id: player.id,
                    };

                    if touching_robot.team_color != kicker.team_color
                        || touching_robot.player_id != kicker.player_id
                    {
                        self.freekick_kicker = None;
                        self.freekick_start_time = None;
                        return;
                    }
                }
            }

            for player in &world_data.yellow_team {
                if self.is_robot_touching_ball(player, ball_pos) {
                    let touching_robot = TeamPlayerId {
                        team_color: TeamColor::Yellow,
                        player_id: player.id,
                    };

                    if touching_robot.team_color != kicker.team_color
                        || touching_robot.player_id != kicker.player_id
                    {
                        self.freekick_kicker = None;
                        self.freekick_start_time = None;
                        return;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use dies_protos::{
        ssl_gc_referee_message::referee::{Command::FORCE_START, Point},
        MessageField,
    };

    use super::*;
    use crate::game_state::GameState::Stop;

    fn referee_msg(command: Command) -> Referee {
        let mut msg = Referee::new();
        msg.set_command(command);
        msg
    }

    fn referee_msg_with_pos(command: Command, pos: Vector2) -> Referee {
        let mut msg = Referee::new();
        msg.set_command(command);
        msg.designated_position = MessageField::some(Point::new());
        msg.designated_position
            .as_mut()
            .unwrap()
            .set_x(pos.x as f32);
        msg.designated_position
            .as_mut()
            .unwrap()
            .set_y(pos.y as f32);
        msg
    }

    #[test]
    fn test_new_game_state_tracker() {
        let tracker = GameStateTracker::new();
        assert_eq!(tracker.get(), GameState::Unknown);
    }

    #[test]
    fn test_normal_update() {
        let mut tracker = GameStateTracker::new();
        tracker.update(&referee_msg(Command::HALT));
        assert_eq!(tracker.get(), GameState::Halt);
        tracker.update(&referee_msg(Command::STOP));
        assert_eq!(tracker.get(), Stop);
        tracker.update(&referee_msg(FORCE_START));
        assert_eq!(tracker.get(), GameState::Run);
        tracker.update(&referee_msg(Command::PREPARE_KICKOFF_BLUE));
        assert_eq!(tracker.get(), GameState::PrepareKickoff);
        tracker.update(&referee_msg(Command::PREPARE_PENALTY_BLUE));
        assert_eq!(tracker.get(), GameState::PreparePenalty);
        tracker.update(&referee_msg(Command::TIMEOUT_BLUE));
        assert_eq!(tracker.get(), GameState::Timeout);
        tracker.update(&referee_msg_with_pos(
            Command::BALL_PLACEMENT_BLUE,
            Vector2::new(0.0, 0.0),
        ));
        assert_eq!(
            tracker.get(),
            GameState::BallReplacement(Vector2::new(0.0, 0.0))
        );
    }

    #[test]
    fn test_only_once_update() {
        let mut tracker = GameStateTracker::new();
        tracker.update(&referee_msg(Command::PREPARE_KICKOFF_YELLOW));
        assert_eq!(tracker.get(), GameState::PrepareKickoff);
        tracker.update(&referee_msg(Command::NORMAL_START));
        assert_eq!(tracker.get(), GameState::Kickoff);
        tracker.update(&referee_msg(Command::NORMAL_START));
        assert_eq!(tracker.get(), GameState::Kickoff);

        tracker.update(&referee_msg(Command::STOP));
        assert_eq!(tracker.get(), Stop);
        tracker.update(&referee_msg(Command::DIRECT_FREE_BLUE));
        assert_eq!(tracker.get(), GameState::FreeKick);
        tracker.update(&referee_msg(Command::DIRECT_FREE_BLUE));
        assert_eq!(tracker.get(), GameState::FreeKick);
    }

    #[test]
    fn test_x_flip() {
        let mut tracker = GameStateTracker::new();
        tracker.update(&referee_msg(Command::HALT));
        assert_eq!(tracker.get(), GameState::Halt);
        tracker.update(&referee_msg(Command::STOP));
        assert_eq!(tracker.get(), Stop);
        tracker.update(&referee_msg(FORCE_START));
        assert_eq!(tracker.get(), GameState::Run);
        tracker.update(&referee_msg(Command::PREPARE_KICKOFF_BLUE));
        assert_eq!(tracker.get(), GameState::PrepareKickoff);
        tracker.update(&referee_msg(Command::PREPARE_PENALTY_BLUE));
        assert_eq!(tracker.get(), GameState::PreparePenalty);
        tracker.update(&referee_msg(Command::TIMEOUT_BLUE));
        assert_eq!(tracker.get(), GameState::Timeout);
        tracker.update(&referee_msg_with_pos(
            Command::BALL_PLACEMENT_BLUE,
            Vector2::new(1.0, 0.0),
        ));

        assert_eq!(
            tracker.get(),
            GameState::BallReplacement(Vector2::new(1.0, 0.0))
        );
    }
}
