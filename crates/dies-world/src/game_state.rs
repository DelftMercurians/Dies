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
/// The ball is in play once it moved this far from the restart spot (mm) — the
/// same criterion the GC uses.
const BALL_IN_PLAY_DIST: f64 = 50.0;
/// Vision distance at or below which a robot counts as touching the ball (mm).
const TOUCH_DIST: f64 = PLAYER_RADIUS + BALL_RADIUS + 10.0;
/// Schmitt release margin above `TOUCH_DIST` for ending the kicker's touch
/// episode — the gap debounces vision jitter without timers, so a one-frame
/// bounce mid-dribble doesn't count as a release.
const KICKER_RELEASE_HYST: f64 = 60.0;

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
    /// World-time (`t_received`) when the free kick started. The referee update
    /// that sets the state has no world clock, so the command only marks
    /// `freekick_timer_pending` and the next tracking frame stamps this. Kept in
    /// world time (not `Instant`) so headless/FTR runs stay deterministic.
    freekick_start_t: Option<f64>,
    freekick_timer_pending: bool,
    /// Ball position on the first tracked frame of the restart (free kick /
    /// kickoff) — the reference for the ball-in-play check.
    restart_spot: Option<Vector2>,
    /// Ball has moved ≥ `BALL_IN_PLAY_DIST` from `restart_spot` (one-way latch
    /// per restart).
    restart_ball_in_play: bool,
    /// Kicker↔ball contact state (Schmitt: enters at `TOUCH_DIST`, exits
    /// `KICKER_RELEASE_HYST` above it) and whether a contact episode has ended
    /// since the kicker was latched. The double-touch bar requires a release:
    /// latching at first contact alone would bar the taker mid-take.
    kicker_in_contact: bool,
    kicker_released: bool,
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
            freekick_start_t: None,
            freekick_timer_pending: false,
            restart_spot: None,
            restart_ball_in_play: false,
            kicker_in_contact: false,
            kicker_released: false,
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
                    // Fresh double-touch tracking for the kickoff taker (no
                    // tracking timeout — that backstop is free-kick-only).
                    self.reset_freekick_tracking();
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
                self.reset_freekick_tracking();
                self.freekick_timer_pending = true;
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
                self.reset_freekick_tracking();
            }
            GameState::Run => {
                self.operator_is_blue = None;
                // Keep the freekick/double-touch tracking state — the bar must
                // survive the FreeKick→Run transition.
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
        distance <= TOUCH_DIST
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

    /// The robot barred from touching the ball by the double-touch rule: the
    /// restart taker, once the ball is in play (moved ≥50mm from the restart
    /// spot) AND its touch episode has ended. Deliberately later than
    /// [`Self::get_freekick_kicker`], which latches identity at first contact —
    /// barring on identity alone evicts the taker from its own kick.
    pub fn get_double_touch_barred(&self) -> Option<TeamPlayerId> {
        if self.restart_ball_in_play && self.kicker_released {
            self.freekick_kicker
        } else {
            None
        }
    }

    /// Clear all restart/double-touch tracking (kicker identity, timers,
    /// restart spot, ball-in-play and contact latches).
    fn reset_freekick_tracking(&mut self) {
        self.freekick_kicker = None;
        self.freekick_start_t = None;
        self.freekick_timer_pending = false;
        self.restart_spot = None;
        self.restart_ball_in_play = false;
        self.kicker_in_contact = false;
        self.kicker_released = false;
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
        let now = world_data.t_received;

        // Stamp the world-time start of the free kick (the referee update that
        // set the state has no world clock) and the restart spot on the first
        // tracked frame of the restart.
        if self.freekick_timer_pending {
            self.freekick_timer_pending = false;
            self.freekick_start_t = Some(now);
        }
        if self.restart_spot.is_none() {
            self.restart_spot = Some(ball_pos);
        }
        if let Some(spot) = self.restart_spot {
            if (ball_pos - spot).norm() >= BALL_IN_PLAY_DIST {
                self.restart_ball_in_play = true;
            }
        }

        if let Some(start_t) = self.freekick_start_t {
            if now - start_t >= FREEKICK_TIMEOUT_SECS as f64 {
                self.reset_freekick_tracking();
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
                    self.kicker_in_contact = true;
                    self.kicker_released = false;
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
                        self.reset_freekick_tracking();
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
                        self.reset_freekick_tracking();
                        return;
                    }
                }
            }

            // Schmitt contact tracking for the kicker's touch episode: contact
            // begins at TOUCH_DIST, ends only past TOUCH_DIST + hysteresis. A
            // completed episode ("released") plus ball-in-play raises the
            // double-touch bar (see `get_double_touch_barred`).
            let kicker_player = match kicker.team_color {
                TeamColor::Blue => world_data
                    .blue_team
                    .iter()
                    .find(|p| p.id == kicker.player_id),
                TeamColor::Yellow => world_data
                    .yellow_team
                    .iter()
                    .find(|p| p.id == kicker.player_id),
            };
            if let Some(p) = kicker_player {
                let dist = (p.position - ball_pos).norm();
                if dist <= TOUCH_DIST {
                    self.kicker_in_contact = true;
                } else if dist > TOUCH_DIST + KICKER_RELEASE_HYST {
                    if self.kicker_in_contact {
                        self.kicker_released = true;
                    }
                    self.kicker_in_contact = false;
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

    /// World with our blue robot 0 and yellow robot 1 at the given positions.
    fn world_at(blue_pos: Vector2, yellow_pos: Vector2, t: f64) -> dies_core::WorldData {
        let mut w = dies_core::mock_world_data();
        w.t_received = t;
        w.blue_team[0].position = blue_pos;
        w.yellow_team[0].position = yellow_pos;
        w
    }

    fn ball_at(pos: Vector2) -> BallData {
        BallData {
            timestamp: 0.0,
            position: Vector3::new(pos.x, pos.y, 0.0),
            raw_position: vec![],
            velocity: Vector3::zeros(),
            detected: true,
        }
    }

    const FAR: Vector2 = Vector2::new(-4000.0, -3000.0);

    /// Start a blue free kick and run one tracking frame with the taker not yet
    /// at the ball; returns the tracker with the restart spot stamped at `spot`.
    fn free_kick_at(spot: Vector2) -> GameStateTracker {
        let mut tracker = GameStateTracker::new();
        tracker.update(&referee_msg(Command::STOP));
        tracker.update(&referee_msg(Command::DIRECT_FREE_BLUE));
        assert_eq!(tracker.get(), GameState::FreeKick);
        let w = world_at(spot + Vector2::new(500.0, 0.0), FAR, 0.0);
        tracker.update_freekick_double_touch(Some(&w), Some(&ball_at(spot)));
        assert_eq!(tracker.get_freekick_kicker(), None);
        tracker
    }

    #[test]
    fn test_new_game_state_tracker() {
        let tracker = GameStateTracker::new();
        assert_eq!(tracker.get(), GameState::Halt);
    }

    #[test]
    fn test_double_touch_bar_full_lifecycle() {
        let spot = Vector2::new(1000.0, 500.0);
        let mut tracker = free_kick_at(spot);
        let kicker = TeamPlayerId {
            team_color: TeamColor::Blue,
            player_id: PlayerId::new(0),
        };

        // Taker reaches the ball: identity latches, bar stays down — barring at
        // first contact would evict the taker from its own kick.
        let w = world_at(spot + Vector2::new(100.0, 0.0), FAR, 0.5);
        tracker.update_freekick_double_touch(Some(&w), Some(&ball_at(spot)));
        assert_eq!(tracker.get_freekick_kicker(), Some(kicker));
        assert_eq!(tracker.get_double_touch_barred(), None);

        // Dribble-into-play: ball moved >50mm but still in contact → no bar.
        let ball = spot + Vector2::new(100.0, 0.0);
        let w = world_at(ball + Vector2::new(100.0, 0.0), FAR, 1.0);
        tracker.update_freekick_double_touch(Some(&w), Some(&ball_at(ball)));
        assert_eq!(tracker.get_double_touch_barred(), None);

        // Inside the hysteresis band (released < 60mm past touch) → still no bar.
        let w = world_at(ball + Vector2::new(TOUCH_DIST + 30.0, 0.0), FAR, 1.5);
        tracker.update_freekick_double_touch(Some(&w), Some(&ball_at(ball)));
        assert_eq!(tracker.get_double_touch_barred(), None);

        // Ball kicked clear of the hysteresis ring → episode over → bar rises.
        let ball = spot + Vector2::new(800.0, 0.0);
        let w = world_at(spot + Vector2::new(200.0, 0.0), FAR, 2.0);
        tracker.update_freekick_double_touch(Some(&w), Some(&ball_at(ball)));
        assert_eq!(tracker.get_double_touch_barred(), Some(kicker));

        // Another robot touches → tracking clears entirely.
        let w = world_at(FAR, ball + Vector2::new(50.0, 0.0), 2.5);
        tracker.update_freekick_double_touch(Some(&w), Some(&ball_at(ball)));
        assert_eq!(tracker.get_freekick_kicker(), None);
        assert_eq!(tracker.get_double_touch_barred(), None);
    }

    #[test]
    fn test_double_touch_bar_requires_ball_in_play() {
        // Taker touches, then backs off without the ball having moved 50mm:
        // released, but the kick wasn't taken → no bar.
        let spot = Vector2::new(0.0, 0.0);
        let mut tracker = free_kick_at(spot);
        let w = world_at(spot + Vector2::new(100.0, 0.0), FAR, 0.5);
        tracker.update_freekick_double_touch(Some(&w), Some(&ball_at(spot)));
        assert!(tracker.get_freekick_kicker().is_some());

        let w = world_at(spot + Vector2::new(400.0, 0.0), FAR, 1.0);
        tracker.update_freekick_double_touch(Some(&w), Some(&ball_at(spot)));
        assert_eq!(tracker.get_double_touch_barred(), None);
    }

    #[test]
    fn test_double_touch_bar_survives_run_and_clears_on_stop() {
        let spot = Vector2::new(0.0, 0.0);
        let mut tracker = free_kick_at(spot);
        let w = world_at(spot + Vector2::new(100.0, 0.0), FAR, 0.5);
        tracker.update_freekick_double_touch(Some(&w), Some(&ball_at(spot)));

        // Kick released into play → bar up.
        let ball = spot + Vector2::new(800.0, 0.0);
        let w = world_at(spot + Vector2::new(100.0, 0.0), FAR, 1.0);
        tracker.update_freekick_double_touch(Some(&w), Some(&ball_at(ball)));
        assert!(tracker.get_double_touch_barred().is_some());

        // FreeKick → Run keeps the bar (the rule outlives the restart state)…
        tracker.update(&referee_msg(FORCE_START));
        assert!(tracker.get_double_touch_barred().is_some());
        // …and Stop clears everything.
        tracker.update(&referee_msg(Command::STOP));
        assert_eq!(tracker.get_double_touch_barred(), None);
        assert_eq!(tracker.get_freekick_kicker(), None);
    }

    #[test]
    fn test_kickoff_double_touch_tracked() {
        let mut tracker = GameStateTracker::new();
        tracker.update(&referee_msg(Command::PREPARE_KICKOFF_BLUE));
        tracker.update(&referee_msg(Command::NORMAL_START));
        assert_eq!(tracker.get(), GameState::Kickoff);

        let spot = Vector2::new(0.0, 0.0);
        let w = world_at(spot + Vector2::new(100.0, 0.0), FAR, 0.0);
        tracker.update_freekick_double_touch(Some(&w), Some(&ball_at(spot)));
        assert!(tracker.get_freekick_kicker().is_some());
        assert_eq!(tracker.get_double_touch_barred(), None);

        let ball = spot + Vector2::new(600.0, 0.0);
        let w = world_at(spot + Vector2::new(100.0, 0.0), FAR, 0.5);
        tracker.update_freekick_double_touch(Some(&w), Some(&ball_at(ball)));
        assert!(tracker.get_double_touch_barred().is_some());
    }

    #[test]
    fn test_freekick_tracking_timeout_uses_world_time() {
        let spot = Vector2::new(0.0, 0.0);
        let mut tracker = free_kick_at(spot); // stamps t=0 as the start
        let w = world_at(spot + Vector2::new(100.0, 0.0), FAR, 0.5);
        tracker.update_freekick_double_touch(Some(&w), Some(&ball_at(spot)));
        assert!(tracker.get_freekick_kicker().is_some());

        // Past the tracking timeout in *world* time (no wall clock involved).
        let w = world_at(spot + Vector2::new(100.0, 0.0), FAR, 10.5);
        tracker.update_freekick_double_touch(Some(&w), Some(&ball_at(spot)));
        assert_eq!(tracker.get_freekick_kicker(), None);
        assert_eq!(tracker.get_double_touch_barred(), None);
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
