use std::time::Instant;

use dies_core::{
    GameState, PlayerData, PlayerId, TeamColor, TeamPlayerId, Vector2, Vector3, WorldData,
    BALL_RADIUS, PLAYER_RADIUS,
};
use dies_protos::ssl_gc_referee_message::{
    referee::{Command, Stage},
    Referee,
};

use crate::BallData;

const FREEKICK_TIMEOUT_SECS: u64 = 10;

#[derive(Debug, Clone)]
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
    blue_team_keeper_id: Option<PlayerId>,
    yellow_team_keeper_id: Option<PlayerId>,

    // --- Enriched referee detail, captured every referee message ---
    stage: Option<Stage>,
    /// Seconds left in the stage at the moment it was captured.
    stage_time_left_s: Option<f64>,
    /// Seconds remaining for the current action at the moment it was captured.
    action_time_remaining_s: Option<f64>,
    /// When the time-remaining values above were captured, for interpolation.
    referee_capture: Option<Instant>,
    next_command: Option<Command>,
    status_message: Option<String>,
    /// GC-reported team names (empty in the packet ⇒ `None`).
    blue_team_name: Option<String>,
    yellow_team_name: Option<String>,
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
            blue_team_keeper_id: None,
            yellow_team_keeper_id: None,
            stage: None,
            stage_time_left_s: None,
            action_time_remaining_s: None,
            referee_capture: None,
            next_command: None,
            status_message: None,
            blue_team_name: None,
            yellow_team_name: None,
        }
    }

    pub fn update(&mut self, data: &Referee) -> GameState {
        let command = data.command();

        // Capture the per-message referee detail (these change on every packet,
        // independent of command transitions, so do it before the early return).
        self.stage = Some(data.stage());
        self.stage_time_left_s = data.stage_time_left.map(|us| us as f64 / 1_000_000.0);
        self.action_time_remaining_s = data
            .current_action_time_remaining
            .map(|us| us as f64 / 1_000_000.0);
        self.next_command = if data.has_next_command() {
            Some(data.next_command())
        } else {
            None
        };
        self.status_message = if data.has_status_message() {
            Some(data.status_message().to_string())
        } else {
            None
        };
        self.referee_capture = Some(Instant::now());

        // GC team names — non-empty only once the operator has typed them.
        let non_empty = |s: &str| (!s.is_empty()).then(|| s.to_string());
        self.blue_team_name = non_empty(data.blue.name());
        self.yellow_team_name = non_empty(data.yellow.name());

        if self.last_cmd == Some(command) {
            return self.game_state;
        }
        self.last_cmd = Some(command);

        self.game_state;

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

        self.blue_team_keeper_id = Some(PlayerId::new(data.blue.goalkeeper()));
        self.yellow_team_keeper_id = Some(PlayerId::new(data.yellow.goalkeeper()));

        self.game_state
    }

    pub fn get_blue_team_keeper_id(&self) -> Option<PlayerId> {
        self.blue_team_keeper_id
    }
    pub fn get_yellow_team_keeper_id(&self) -> Option<PlayerId> {
        self.yellow_team_keeper_id
    }

    pub fn get_operator_is_blue(&self) -> Option<bool> {
        self.operator_is_blue
    }

    /// The coarse game stage as a display string, if known.
    pub fn get_stage_display(&self) -> Option<String> {
        self.stage.map(|s| stage_display(s).to_string())
    }

    /// Seconds left in the stage, interpolated to the current instant.
    pub fn get_stage_time_left(&self) -> Option<f64> {
        self.interpolate(self.stage_time_left_s)
    }

    /// Seconds remaining for the current action (free kick / placement /
    /// kickoff), interpolated to the current instant. Can be negative.
    pub fn get_action_time_remaining(&self) -> Option<f64> {
        self.interpolate(self.action_time_remaining_s)
    }

    pub fn get_next_command_display(&self) -> Option<String> {
        self.next_command.map(command_display)
    }

    /// The `GameState` that the GC's `next_command` hint predicts will resume
    /// play after the current stoppage, for the meaningful set-piece restarts
    /// (free kick / kickoff / penalty). `None` for force/normal starts, ball
    /// placement, or when no hint is present. See [`predict_from_command`].
    pub fn get_predicted_next_game_state(&self) -> Option<GameState> {
        self.next_command
            .and_then(predict_from_command)
            .map(|(state, _)| state)
    }

    /// The team the predicted restart ([`Self::get_predicted_next_game_state`])
    /// is for, derived from the color suffix of the `next_command` hint.
    pub fn get_predicted_operating_team(&self) -> Option<TeamColor> {
        self.next_command
            .and_then(predict_from_command)
            .map(|(_, is_blue)| {
                if is_blue {
                    TeamColor::Blue
                } else {
                    TeamColor::Yellow
                }
            })
    }

    pub fn get_status_message(&self) -> Option<String> {
        self.status_message.clone()
    }

    /// GC-reported blue team name, if the operator has entered one.
    pub fn get_blue_team_name(&self) -> Option<String> {
        self.blue_team_name.clone()
    }

    /// GC-reported yellow team name, if the operator has entered one.
    pub fn get_yellow_team_name(&self) -> Option<String> {
        self.yellow_team_name.clone()
    }

    /// Subtract the time elapsed since the last referee capture from a captured
    /// countdown value, so the UI shows a smooth tick between referee packets.
    fn interpolate(&self, captured: Option<f64>) -> Option<f64> {
        match (captured, self.referee_capture) {
            (Some(v), Some(t)) => Some(v - t.elapsed().as_secs_f64()),
            (Some(v), None) => Some(v),
            _ => None,
        }
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
        distance <= (PLAYER_RADIUS + BALL_RADIUS + 10.0)
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
        if (self.game_state != GameState::FreeKick && self.game_state != GameState::Kickoff)
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

/// Human-readable label for a referee stage.
fn stage_display(stage: Stage) -> &'static str {
    match stage {
        Stage::NORMAL_FIRST_HALF_PRE => "Pre first half",
        Stage::NORMAL_FIRST_HALF => "First half",
        Stage::NORMAL_HALF_TIME => "Half time",
        Stage::NORMAL_SECOND_HALF_PRE => "Pre second half",
        Stage::NORMAL_SECOND_HALF => "Second half",
        Stage::EXTRA_TIME_BREAK => "Extra time break",
        Stage::EXTRA_FIRST_HALF_PRE => "Pre extra first half",
        Stage::EXTRA_FIRST_HALF => "Extra first half",
        Stage::EXTRA_HALF_TIME => "Extra half time",
        Stage::EXTRA_SECOND_HALF_PRE => "Pre extra second half",
        Stage::EXTRA_SECOND_HALF => "Extra second half",
        Stage::PENALTY_SHOOTOUT_BREAK => "Shootout break",
        Stage::PENALTY_SHOOTOUT => "Penalty shootout",
        Stage::POST_GAME => "Post game",
    }
}

/// Human-readable label for a referee command ("what's next").
pub fn command_display(command: Command) -> String {
    match command {
        Command::HALT => "Halt",
        Command::STOP => "Stop",
        Command::NORMAL_START => "Normal start",
        Command::FORCE_START => "Force start",
        Command::PREPARE_KICKOFF_YELLOW => "Kick-off (Yellow)",
        Command::PREPARE_KICKOFF_BLUE => "Kick-off (Blue)",
        Command::PREPARE_PENALTY_YELLOW => "Penalty (Yellow)",
        Command::PREPARE_PENALTY_BLUE => "Penalty (Blue)",
        Command::DIRECT_FREE_YELLOW => "Free kick (Yellow)",
        Command::DIRECT_FREE_BLUE => "Free kick (Blue)",
        Command::INDIRECT_FREE_YELLOW => "Indirect free kick (Yellow)",
        Command::INDIRECT_FREE_BLUE => "Indirect free kick (Blue)",
        Command::TIMEOUT_YELLOW => "Timeout (Yellow)",
        Command::TIMEOUT_BLUE => "Timeout (Blue)",
        Command::GOAL_YELLOW => "Goal (Yellow)",
        Command::GOAL_BLUE => "Goal (Blue)",
        Command::BALL_PLACEMENT_YELLOW => "Ball placement (Yellow)",
        Command::BALL_PLACEMENT_BLUE => "Ball placement (Blue)",
    }
    .to_string()
}

/// Map a GC `next_command` hint to the `GameState` that will resume play and the
/// team that operates it (`true` = blue), for the meaningful set-piece restarts
/// only. Kickoff/penalty map to their *prepare* variants (so downstream logic
/// stages without treating the ball as in play); free kicks map to `FreeKick`.
/// Returns `None` for force/normal starts, ball placement, timeouts, and the
/// deprecated indirect free kicks — i.e. cases we don't pre-stage for.
pub fn predict_from_command(command: Command) -> Option<(GameState, bool)> {
    match command {
        Command::DIRECT_FREE_BLUE => Some((GameState::FreeKick, true)),
        Command::DIRECT_FREE_YELLOW => Some((GameState::FreeKick, false)),
        Command::PREPARE_KICKOFF_BLUE => Some((GameState::PrepareKickoff, true)),
        Command::PREPARE_KICKOFF_YELLOW => Some((GameState::PrepareKickoff, false)),
        Command::PREPARE_PENALTY_BLUE => Some((GameState::PreparePenalty, true)),
        Command::PREPARE_PENALTY_YELLOW => Some((GameState::PreparePenalty, false)),
        _ => None,
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
        assert_eq!(tracker.get(), GameState::Halt);
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
    fn test_predict_from_command() {
        assert_eq!(
            predict_from_command(Command::DIRECT_FREE_BLUE),
            Some((GameState::FreeKick, true))
        );
        assert_eq!(
            predict_from_command(Command::DIRECT_FREE_YELLOW),
            Some((GameState::FreeKick, false))
        );
        assert_eq!(
            predict_from_command(Command::PREPARE_KICKOFF_YELLOW),
            Some((GameState::PrepareKickoff, false))
        );
        assert_eq!(
            predict_from_command(Command::PREPARE_PENALTY_BLUE),
            Some((GameState::PreparePenalty, true))
        );
        assert_eq!(predict_from_command(FORCE_START), None);
        assert_eq!(predict_from_command(Command::NORMAL_START), None);
        assert_eq!(predict_from_command(Command::BALL_PLACEMENT_BLUE), None);
    }

    #[test]
    fn test_predicted_getters_track_next_command() {
        let mut tracker = GameStateTracker::new();
        tracker.update(&referee_msg(Command::STOP));
        // No next_command on the plain STOP message → no prediction.
        assert_eq!(tracker.get_predicted_next_game_state(), None);
        assert_eq!(tracker.get_predicted_operating_team(), None);

        let mut msg = referee_msg(Command::STOP);
        msg.set_next_command(Command::DIRECT_FREE_YELLOW);
        tracker.update(&msg);
        assert_eq!(
            tracker.get_predicted_next_game_state(),
            Some(GameState::FreeKick)
        );
        assert_eq!(
            tracker.get_predicted_operating_team(),
            Some(TeamColor::Yellow)
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
