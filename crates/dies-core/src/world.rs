use std::{
    collections::HashSet,
    fmt::Display,
    hash::Hash,
    time::{Duration, Instant},
};

use dies_protos::{ssl_gc_common::Team, ssl_vision_detection_tracked::TrackedFrame};
use serde::{Deserialize, Serialize};
use typeshare::typeshare;

use crate::{
    distance_to_line, player::PlayerId, Angle, FieldGeometry, Handicap, Possession,
    ReflexKickState, RoleType, SideAssignment, SysStatus, TeamColor, TeamPlayerId, Vector2,
    Vector3,
};

const STOP_BALL_AVOIDANCE_RADIUS: f64 = 800.0;
pub const PLAYER_RADIUS: f64 = 90.0;
pub const BALL_RADIUS: f64 = 21.5;
const MAX_SPEED: f64 = 10000.0;
const MAX_ACCELERATION: f64 = 125000.0;

// Enum to represent different obstacle types
#[derive(Debug, Clone, Serialize)]
pub enum Obstacle {
    Circle { center: Vector2, radius: f64 },
    Rectangle { min: Vector2, max: Vector2 },
    Line { start: Vector2, end: Vector2 },
}

#[derive(Debug, Clone, Copy)]
pub enum WorldInstant {
    Real(Instant),
    Simulated(f64),
}

impl WorldInstant {
    pub fn now_real() -> Self {
        WorldInstant::Real(Instant::now())
    }

    pub fn simulated(t: f64) -> Self {
        WorldInstant::Simulated(t)
    }

    pub fn duration_since(&self, earlier: &Self) -> f64 {
        match (earlier, self) {
            (WorldInstant::Real(earlier), WorldInstant::Real(later)) => {
                later.duration_since(*earlier).as_secs_f64()
            }
            (WorldInstant::Simulated(earlier), WorldInstant::Simulated(later)) => later - earlier,
            _ => panic!("Cannot compare real and simulated instants"),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[typeshare]
pub struct WorldUpdate {
    pub world_data: WorldData,
    /// Monotonic id of this world frame (minted by the executor). Exposed to the
    /// UI for the frame counter and used as the join key in recorded logs.
    #[typeshare(serialized_as = "number")]
    pub frame_id: u64,
    /// Announcer feed items minted since the previous broadcast. The UI
    /// accumulates these into a scrolling commentary log (deduped by `id`).
    /// Empty on most frames; only populated when something happens.
    #[serde(default)]
    pub announcements: Vec<Announcement>,
}

/// A single line in the announcer commentary feed, formatted backend-side from
/// referee commands, game events, and (in sim) the simulator's own events so it
/// reads identically for sim and live matches.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[typeshare]
pub struct Announcement {
    /// Monotonic id, unique within a session. The UI uses it to dedupe and to
    /// detect freshly-arrived lines (which animate in at full opacity).
    #[typeshare(serialized_as = "number")]
    pub id: u64,
    /// World time (seconds) when the line was minted.
    pub timestamp: f64,
    pub category: AnnouncementCategory,
    /// The team the line concerns, if any — used to tint the line.
    pub team: Option<TeamColor>,
    /// Human-readable commentary, e.g. "Free kick — Blue" or
    /// "Throw-in: ball left field, last touched by Yellow".
    pub text: String,
}

/// Coarse classification of an announcer line, used by the UI for icon/color.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[typeshare]
pub enum AnnouncementCategory {
    /// Generic state/info transition (Halt, Stop, Run, Force start, Timeout).
    Info,
    /// A stoppage reason (ball out of field, no progress).
    Stoppage,
    FreeKick,
    Kickoff,
    Penalty,
    /// Ball placement related.
    Placement,
    /// A foul or rule violation.
    Foul,
    /// A card was issued.
    Card,
    Goal,
}

/// The game state, as reported by the referee.
#[derive(Serialize, Deserialize, Clone, Debug, Copy, Default)]
#[serde(tag = "type", content = "data")]
#[typeshare]
pub enum GameState {
    #[default]
    Unknown,
    Halt,
    Timeout,
    Stop,
    PrepareKickoff,
    BallReplacement(Vector2),
    PreparePenalty,
    Kickoff,
    FreeKick,
    Penalty,
    PenaltyRun,
    Run,
}

impl GameState {
    pub fn is_ball_in_play(&self) -> bool {
        matches!(
            self,
            GameState::Kickoff | GameState::PenaltyRun | GameState::Run | GameState::FreeKick
        )
    }
}

impl Display for GameState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            GameState::Unknown => "Unknown".to_string(),
            GameState::Halt => "Halt".to_string(),
            GameState::Timeout => "Timeout".to_string(),
            GameState::Stop => "Stop".to_string(),
            GameState::PrepareKickoff => "PrepareKickoff".to_string(),
            GameState::BallReplacement(_) => "BallReplacement".to_string(),
            GameState::PreparePenalty => "PreparePenalty".to_string(),
            GameState::Kickoff => "Kickoff".to_string(),
            GameState::FreeKick => "FreeKick".to_string(),
            GameState::Penalty => "Penalty".to_string(),
            GameState::PenaltyRun => "PenaltyRun".to_string(),
            GameState::Run => "Run".to_string(),
        };
        write!(f, "{}", str)
    }
}
impl Hash for GameState {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.to_string().hash(state);
    }
}

impl PartialEq for GameState {
    fn eq(&self, other: &Self) -> bool {
        self.to_string() == other.to_string()
    }
}

impl Eq for GameState {}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum StrategyGameStateMacther {
    #[default]
    Any,
    Specific(GameState),
    AnyOf(HashSet<GameState>),
}

impl StrategyGameStateMacther {
    pub fn matches(&self, state: &GameState) -> bool {
        match self {
            StrategyGameStateMacther::Any => true,
            StrategyGameStateMacther::Specific(s) => s == state,
            StrategyGameStateMacther::AnyOf(states) => states.contains(state),
        }
    }

    pub fn any() -> Self {
        StrategyGameStateMacther::Any
    }

    pub fn specific(state: GameState) -> Self {
        StrategyGameStateMacther::Specific(state)
    }

    pub fn any_of(states: &[GameState]) -> Self {
        StrategyGameStateMacther::AnyOf(states.iter().cloned().collect())
    }
}

/// A struct to store the ball state from a single frame.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[typeshare]
pub struct BallData {
    /// Unix timestamp of the recorded frame from which this data was extracted (in
    /// seconds). This is the time that ssl-vision received the frame.
    pub timestamp: f64,
    /// Position of the ball filtered by us, in mm, in dies coordinates
    pub position: Vector3,
    /// Raw position as reported by vision
    pub raw_position: Vec<Vector3>,
    /// Velocity of the ball in mm/s, in dies coordinates
    pub velocity: Vector3,
    /// Whether the ball is being detected
    pub detected: bool,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default)]
#[typeshare]
pub struct GameStateData {
    /// The state of current game
    pub game_state: GameState,
    /// If we are the main party currently performing tasks in the state.
    /// true for symmetric states(halt stop run timout)
    pub us_operating: bool,
    #[typeshare(serialized_as = "u32")]
    pub yellow_cards: usize,
    /// The player who performed the freekick, if it was us (for double touch tracking)
    /// Only Some until another player touches the ball
    pub freekick_kicker: Option<PlayerId>,
    /// Our robot barred from touching the ball by the double-touch rule (the
    /// restart taker, once the ball is in play and released). `None` while the
    /// taker is still legally taking the kick, and when the taker is theirs.
    pub double_touch_barred: Option<PlayerId>,
    pub max_allowed_bots: u32,
    pub our_keeper_id: Option<PlayerId>,
    /// The restart the GC's `next_command` hint predicts will resume play after
    /// the current stoppage (free kick / kickoff / penalty), if any. Used to
    /// pre-stage during Stop/BallReplacement. `None` when there is no meaningful
    /// hint. This never affects rule compliance — that keys off `game_state`.
    pub predicted_next_game_state: Option<GameState>,
    /// Whether the predicted restart ([`Self::predicted_next_game_state`]) is
    /// ours to take, in team-relative terms. `None` when there is no prediction.
    pub predicted_us_operating: Option<bool>,
}

/// Effective keeper id for a team. The GC's designation is ground truth and
/// always wins — the rules only recognize the robot registered with the referee
/// as the keeper, so we must never second-guess it (even if that robot is off
/// the field: substituting another robot into the box would itself be a foul).
/// Only when no GC has told us anything (no referee connected — e.g. field tests
/// or GC-less sim runs) do we fall back deterministically: id 4 if present, then
/// 0, then the lowest id on the roster.
pub fn effective_keeper_id<I: IntoIterator<Item = PlayerId>>(
    gc_keeper: Option<PlayerId>,
    roster: I,
) -> Option<PlayerId> {
    if gc_keeper.is_some() {
        return gc_keeper;
    }
    let mut ids: Vec<PlayerId> = roster.into_iter().collect();
    ids.sort_unstable();
    let preferred = PlayerId::new(4);
    if ids.contains(&preferred) {
        return Some(preferred);
    }
    ids.first().copied()
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[typeshare]
pub struct RawGameStateData {
    /// The state of current game
    pub game_state: GameState,
    /// The team that is currently performing tasks in the state.
    pub operating_team: TeamColor,
    /// The player who performed the freekick (for double touch tracking)
    pub freekick_kicker: Option<TeamPlayerId>,
    /// The robot barred from touching the ball by the double-touch rule: the
    /// restart taker, once the ball is in play AND the taker released it.
    /// Rises later than `freekick_kicker` (which latches at first contact) —
    /// while the taker is still legally taking the kick this is `None`.
    pub double_touch_barred: Option<TeamPlayerId>,
    pub blue_team_max_allowed_bots: u32,
    pub yellow_team_max_allowed_bots: u32,
    #[typeshare(serialized_as = "u32")]
    pub blue_team_yellow_cards: usize,
    #[typeshare(serialized_as = "u32")]
    pub yellow_team_yellow_cards: usize,
    /// Goals scored, from the referee `TeamInfo` (tracked in sim, real in live).
    pub blue_team_score: u32,
    pub yellow_team_score: u32,

    pub blue_team_keeper_id: Option<PlayerId>,
    pub yellow_team_keeper_id: Option<PlayerId>,

    // --- Enriched referee detail (for the game-state overlay panel) ---
    /// The coarse game stage, e.g. "First half" (display string).
    pub stage: Option<String>,
    /// Seconds left in the current stage (can be negative). Interpolated.
    pub stage_time_left: Option<f64>,
    /// Seconds remaining for the current action — free kick / placement /
    /// kickoff countdown (can be negative). Interpolated frame-by-frame.
    pub action_time_remaining: Option<f64>,
    /// The command that will resume play after the current stoppage ("what's
    /// next"), as a display string.
    pub next_command: Option<String>,
    /// Machine-usable prediction derived from the `next_command` hint: the
    /// `GameState` that will resume play, for the meaningful set-piece restarts
    /// (free kick / kickoff / penalty). `None` otherwise.
    pub predicted_next_game_state: Option<GameState>,
    /// The team the predicted restart is for (from the `next_command` color).
    pub predicted_operating_team: Option<TeamColor>,
    /// A human-readable reason for the current stoppage, if provided.
    pub status_message: Option<String>,
    /// GC-reported team names, if the operator has entered them.
    pub blue_team_name: Option<String>,
    pub yellow_team_name: Option<String>,
}

/// Execution state of a player's active skill, mirroring the executor's
/// `SkillStatus` without coupling `dies-core` to the strategy protocol.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[typeshare]
pub enum SkillState {
    /// No skill has been commanded for this player.
    #[default]
    Idle,
    /// The skill is currently executing.
    Running,
    /// The skill completed successfully.
    Succeeded,
    /// The skill failed.
    Failed,
}

/// A snapshot of the skill a (own-team) player is currently executing, surfaced
/// to the UI. Populated by the executor from the active skill's own report.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[typeshare]
pub struct PlayerSkillInfo {
    /// Short skill type name, e.g. `"GoToPos"`, `"Shoot"`, `"Pass"`.
    pub skill_type: String,
    /// Execution state.
    pub state: SkillState,
    /// Human-readable description of the skill's internal state, e.g.
    /// `"approaching ball"` or `"aiming"`.
    pub description: String,
}

/// Why a robot is present on the field but not an assignable roster member.
///
/// Both cases are physically on the field (still avoided as obstacles) but must
/// never be handed to the strategy as a controllable player:
/// - `RadioLost`: seen by vision but the RF/basestation link is gone, so we
///   cannot command it (non-cooperative — teammates take full avoidance).
/// - `CardRemoved`: benched by the team controller for a yellow card and driven
///   off-field (still cooperative — it moves under our ORCA control).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[typeshare]
pub enum SidelineReason {
    RadioLost,
    CardRemoved,
}

/// A struct to store the player state from a single frame.
/// Live per-robot open-loop delay estimate (command → first visible motion on
/// vision), computed online from stop→go events via the same √displacement
/// x-intercept extrapolation used offline. Only populated for own players, and
/// only once at least one clean event has been measured. The rolling window is
/// anchored to the most recent event, so a robot that sits still does not lose
/// its last estimate — `age_s` tells you how stale it is.
#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
#[typeshare]
pub struct OpenLoopDelayStats {
    /// Median measured delay (ms) over the rolling window.
    pub median_ms: f64,
    /// Maximum measured delay (ms) over the rolling window.
    pub max_ms: f64,
    /// Seconds since the most recent measured event (staleness of the estimate).
    pub age_s: f64,
    /// Number of events in the rolling window.
    pub sample_count: u32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[typeshare]
pub struct PlayerData {
    /// Unix timestamp of the recorded frame from which this data was extracted (in
    /// seconds). This is the time that ssl-vision received the frame.
    pub timestamp: f64,
    /// The player's unique id
    pub id: PlayerId,
    /// Unfiltered position as reported by vision
    pub raw_position: Vector2,
    /// Position of the player filtered by us in mm, in dies coordinates
    pub position: Vector2,
    /// Velocity of the player in mm/s, in dies coordinates
    pub velocity: Vector2,
    /// Yaw of the player, in radians, (`-pi`, `pi`), where `0` is the positive
    /// x direction, and `pi/2` is the positive y direction.
    pub yaw: Angle,
    /// Unfiltered yaw as reported by vision
    pub raw_yaw: Angle,
    /// Angular speed of the player (in rad/s)
    pub angular_speed: f64,
    /// EWMA-smoothed position noise floor in mm — RMS of the Kalman innovation
    /// (raw vision vs. constant-velocity prediction). Useful for diagnosing
    /// whether downstream controllers are tracking sensor noise.
    pub position_noise: f64,

    /// The overall status of the robot. Only available for own players.
    pub primary_status: Option<SysStatus>,
    /// The voltage of the kicker capacitor (in V). Only available for own players.
    pub kicker_cap_voltage: Option<f32>,
    /// The temperature of the kicker. Only available for own players.
    pub kicker_temp: Option<f32>,
    /// The voltages of the battery packs. Only available for own players.
    pub pack_voltages: Option<[f32; 2]>,
    /// Raw breakbeam sensor reading (only meaningful for own players). An *input*
    /// to the possession metric and kept for logging — not a consumer-facing
    /// signal. Use `has_ball` (or `WorldData::possession`) instead.
    pub breakbeam_ball_detected: bool,
    /// Unified "this player has the ball" signal, derived from
    /// `WorldData::possession` (true iff this player is the confident owner).
    pub has_ball: bool,
    /// Whether the onboard ToF ball sensor is reporting `Ok` (only meaningful for
    /// own players). Used as the magnet-mode capability gate — `Standby`/etc. and
    /// robots without the sensor read `false`.
    pub tof_ok: bool,
    /// Whether the onboard ToF sensor currently sees the ball in its ~64 mm front
    /// window (only meaningful for own players). The magnet-mode engage trigger.
    pub tof_ball_detected: bool,
    /// Raw ToF ball-position estimate `[x, y]` in sensor units (`x` signed
    /// −left/+right, `y` unsigned +forward/−back). Only meaningful for own
    /// players; `None` when the sensor reports nothing. An *input* to the
    /// ToF-backup breakbeam substitute — use `has_ball`, not this.
    pub tof_xy: Option<[i32; 2]>,
    /// Raw ToF detection confidence byte (`0..=255`); `None` when the robot
    /// reports no reading. Input to the ToF-backup breakbeam substitute.
    pub tof_confidence: Option<u8>,
    /// Whether this robot's per-robot ToF-backup breakbeam toggle is on. When
    /// true, the hardware breakbeam bit is ignored for possession and the
    /// Schmitt-triggered ToF signal (`tof_backup_ball_detected`) stands in.
    pub tof_backup_enabled: bool,
    /// Latched output of the ToF-backup Schmitt trigger (only meaningful when
    /// `tof_backup_enabled`). Stamped from the possession tracker, mirrors how
    /// `has_ball` is filled — for logging/tuning, not a consumer signal.
    pub tof_backup_ball_detected: bool,
    pub imu_status: Option<SysStatus>,
    pub imu_readings: Option<[f32; 6]>,
    pub kicker_status: Option<SysStatus>,
    /// State of the firmware reflex-kick state machine, as reported over feedback
    /// (only meaningful for own players; `None` when the robot reports nothing).
    /// `Armed` means the kicker will fire the instant the ball trips the beam.
    /// Surfaced for logging/UI; skills may read it to confirm the firmware armed.
    pub reflex_kick_state: Option<ReflexKickState>,

    /// The skill this player is currently executing, if known. Only populated
    /// for own players (the team we control); `None` for opponents and when no
    /// skill is active.
    pub skill: Option<PlayerSkillInfo>,

    pub handicaps: HashSet<Handicap>,

    /// Set when this robot is on the field but not an assignable roster member
    /// (radio lost, or card-removed). `None` for normal players and opponents.
    /// Robots with this set are forked out of `TeamData::own_players` into
    /// `TeamData::sidelined_players` but remain in the color lists.
    #[serde(default)]
    pub sideline: Option<SidelineReason>,

    /// Live open-loop delay estimate (own players only). `None` until the first
    /// clean stop→go event has been measured. Stamped by the executor.
    #[serde(default)]
    pub open_loop_delay: Option<OpenLoopDelayStats>,
}

impl PlayerData {
    pub fn new(id: PlayerId) -> Self {
        Self {
            timestamp: 0.0,
            id,
            raw_position: Vector2::zeros(),
            position: Vector2::zeros(),
            velocity: Vector2::zeros(),
            yaw: Angle::default(),
            raw_yaw: Angle::default(),
            angular_speed: 0.0,
            position_noise: 0.0,
            primary_status: None,
            kicker_cap_voltage: None,
            kicker_temp: None,
            pack_voltages: None,
            imu_status: None,
            breakbeam_ball_detected: false,
            has_ball: false,
            tof_ok: false,
            tof_ball_detected: false,
            tof_xy: None,
            tof_confidence: None,
            tof_backup_enabled: false,
            tof_backup_ball_detected: false,
            kicker_status: None,
            imu_readings: None,
            reflex_kick_state: None,
            skill: None,
            handicaps: HashSet::new(),
            sideline: None,
            open_loop_delay: None,
        }
    }

    /// On the field but not an assignable roster member (radio-lost or
    /// card-removed). Such robots are forked out of `TeamData::own_players`.
    pub fn is_sidelined(&self) -> bool {
        self.sideline.is_some()
    }

    /// Benched for a yellow card and driven off-field. We still command these
    /// (unlike radio-lost robots), so they stay in the commandable set.
    pub fn is_card_removed(&self) -> bool {
        self.sideline == Some(SidelineReason::CardRemoved)
    }
}

pub enum BallPrediction {
    Linear(Vector2),
    Collision(Vector2),
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[typeshare]
pub struct AutorefKickedBall {
    /// The initial position [m] from which the ball was kicked
    pub pos: Vector2,
    /// The initial velocity [m/s] with which the ball was kicked
    pub vel: Vector3,
    /// The unix timestamp [s] when the kick was performed
    pub start_timestamp: f64,

    /// The predicted unix timestamp [s] when the ball comes to a stop
    pub stop_timestamp: Option<f64>,
    /// The predicted position [m] at which the ball will come to a stop
    pub stop_pos: Option<Vector2>,

    /// The robot that kicked the ball
    #[typeshare(skip)]
    pub robot_id: Option<(TeamColor, PlayerId)>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[typeshare]
pub struct AutorefInfo {
    pub kicked_ball: Option<AutorefKickedBall>,
}

impl From<TrackedFrame> for AutorefInfo {
    fn from(frame: TrackedFrame) -> Self {
        Self {
            kicked_ball: frame.kicked_ball.as_ref().map(|k| AutorefKickedBall {
                pos: Vector2::new(k.pos.x() as f64, k.pos.y() as f64),
                vel: Vector3::new(k.vel.x() as f64, k.vel.y() as f64, k.vel.z() as f64),
                start_timestamp: k.start_timestamp.unwrap_or(0.0),
                stop_timestamp: k.stop_timestamp,
                stop_pos: k
                    .stop_pos
                    .as_ref()
                    .map(|p| Vector2::new(p.x() as f64, p.y() as f64)),
                robot_id: if let Some(r) = k.robot_id.as_ref() {
                    let team = match r.team() {
                        Team::BLUE => Some(TeamColor::Blue),
                        Team::YELLOW => Some(TeamColor::Yellow),
                        Team::UNKNOWN => None,
                    };
                    if let (Some(team), Some(id)) = (team, r.id) {
                        Some((team, PlayerId::new(id)))
                    } else {
                        None
                    }
                } else {
                    None
                },
            }),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[typeshare]
pub struct AutorefKickedBallTeam {
    /// The initial position [m] from which the ball was kicked
    pub pos: Vector2,
    /// The initial velocity [m/s] with which the ball was kicked
    pub vel: Vector3,
    /// The unix timestamp [s] when the kick was performed
    pub start_timestamp: f64,

    /// The predicted unix timestamp [s] when the ball comes to a stop
    pub stop_timestamp: Option<f64>,
    /// The predicted position [m] at which the ball will come to a stop
    pub stop_pos: Option<Vector2>,

    /// The robot that kicked the ball
    pub robot_id: PlayerId,
    pub we_kicked: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[typeshare]
pub struct WorldData {
    /// Timestamp of the frame, in seconds. This timestamp is relative to the time the
    /// world tracking was started.
    pub t_received: f64,
    /// Recording timestamp of the frame, in seconds, as reported by vision. This
    /// timestamp is relative to the time the first image was captured.
    pub t_capture: f64,
    /// The time since the last frame was received, in seconds
    pub dt: f64,
    pub blue_team: Vec<PlayerData>,
    pub yellow_team: Vec<PlayerData>,
    pub ball: Option<BallData>,
    pub field_geom: Option<FieldGeometry>,
    pub game_state: RawGameStateData,
    pub side_assignment: SideAssignment,
    pub ball_on_blue_side: Option<Duration>,
    pub ball_on_yellow_side: Option<Duration>,
    pub autoref_info: Option<AutorefInfo>,
    /// Unified ball-possession metric (absolute / team-tagged). Computed once in
    /// the world tracker; the single source of truth for who has the ball.
    pub possession: Possession,
}

impl WorldData {
    pub fn get_team_players(&self, color: TeamColor) -> &Vec<PlayerData> {
        match color {
            TeamColor::Blue => &self.blue_team,
            TeamColor::Yellow => &self.yellow_team,
        }
    }

    pub fn get_opp_team_players(&self, color: TeamColor) -> &Vec<PlayerData> {
        match color {
            TeamColor::Blue => &self.yellow_team,
            TeamColor::Yellow => &self.blue_team,
        }
    }

    pub fn get_team_data(&self, color: TeamColor, ignore_gc: bool) -> TeamData {
        self.side_assignment
            .transform_to_team_coords(color, self, ignore_gc)
    }
}

/// A struct to store the world state from a single frame.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[typeshare]
pub struct TeamData {
    /// Timestamp of the frame, in seconds. This timestamp is relative to the time the
    /// world tracking was started.
    pub t_received: f64,
    /// Recording timestamp of the frame, in seconds, as reported by vision. This
    /// timestamp is relative to the time the first image was captured.
    pub t_capture: f64,
    /// The time since the last frame was received, in seconds
    pub dt: f64,
    pub own_players: Vec<PlayerData>,
    pub opp_players: Vec<PlayerData>,
    /// Own-color robots on the field that are not assignable roster members
    /// (radio lost, or card-removed — see [`SidelineReason`]). Excluded from
    /// `own_players` so the strategy never roles them, but still avoided as
    /// obstacles by the executor and blocking for ray/radius queries.
    #[serde(default)]
    pub sidelined_players: Vec<PlayerData>,
    pub ball: Option<BallData>,
    pub field_geom: Option<FieldGeometry>,
    pub current_game_state: GameStateData,
    pub ball_on_our_side: Option<Duration>,
    pub ball_on_opp_side: Option<Duration>,
    pub kicked_ball: Option<AutorefKickedBallTeam>,
    /// Unified possession metric (absolute / team-tagged — the same value as on
    /// `WorldData`). Converted to a team-relative view at the strategy boundary.
    pub possession: Possession,
}

impl TeamData {
    pub fn get_player(&self, id: PlayerId) -> &PlayerData {
        self.own_players.iter().find(|p| p.id == id).unwrap()
    }

    pub fn players_within_radius(&self, pos: Vector2, radius: f64) -> Vec<&PlayerData> {
        self.own_players
            .iter()
            .chain(self.opp_players.iter())
            .chain(self.sidelined_players.iter())
            .filter(|p| (p.position.xy() - pos).norm() < radius)
            .collect()
    }

    /// Compute the approximate time it will take for the player to reach the target point
    pub fn time_to_reach_point(&self, player: &PlayerData, target: Vector2) -> f64 {
        let max_speed = MAX_SPEED;
        let max_acceleration = MAX_ACCELERATION;
        let current_speed = player.velocity.norm();
        let dist = (target - player.position.xy()).norm();

        // Acceleration phase
        let t1 = (max_speed - current_speed) / max_acceleration;
        let d1 = current_speed * t1 + 0.5 * max_acceleration * t1.powi(2);

        // Deceleration phase
        let t3 = max_speed / max_acceleration;
        let d3 = 0.5 * max_speed * t3;

        // Constant speed phase
        let d2 = dist - d1 - d3;

        if d2 > 0.0 {
            // The robot reaches max speed
            let t2 = d2 / max_speed;
            t1 + t2 + t3
        } else {
            // The robot doesn't reach max speed
            // Calculate time considering both acceleration and deceleration
            let v_peak = (max_acceleration * dist).sqrt();
            let t_acc = (v_peak - current_speed) / max_acceleration;
            let t_dec = v_peak / max_acceleration;
            t_acc + t_dec
        }
    }

    /// Cast a ray from the start point in the given direction and return the intersection point with the first
    /// player or wall that it.
    pub fn cast_ray(&self, start: Vector2, direction: Vector2) -> Option<Vector2> {
        let normalized_direction = direction.normalize();
        let mut closest_intersection: Option<(f64, Vector2)> = None;

        let update_closest = |closest: &mut Option<(f64, Vector2)>, t: f64, point: Vector2| {
            if t > 0.0 && (closest.is_none() || t < closest.unwrap().0) {
                *closest = Some((t, point));
            }
        };

        // Check intersections with players
        for player in self
            .own_players
            .iter()
            .chain(self.opp_players.iter())
            .chain(self.sidelined_players.iter())
        {
            let oc = start - player.position.xy();
            let a = normalized_direction.dot(&normalized_direction);
            let b = 2.0 * oc.dot(&normalized_direction);
            let c = oc.dot(&oc) - PLAYER_RADIUS * PLAYER_RADIUS;
            let discriminant = b * b - 4.0 * a * c;

            if discriminant >= 0.0 {
                let t = (-b - discriminant.sqrt()) / (2.0 * a);
                if t > 0.0 {
                    let intersection_point = start + normalized_direction * t;
                    update_closest(&mut closest_intersection, t, intersection_point);
                }
            }
        }

        // Check intersections with field boundaries
        // if let Some(field_geom) = &self.field_geom {
        //     let half_length = field_geom.field_length / 2.0;
        //     let half_width = field_geom.field_width / 2.0;

        //     let check_boundary = |p1: Vector2, p2: Vector2| {
        //         let v1 = p2 - p1;
        //         if let Some(intersection) = find_intersection(p1, v1, start, direction) {
        //             let t = (intersection - start).dot(&normalized_direction);
        //             update_closest(&mut closest_intersection, t, intersection);
        //         }
        //     };

        // Check all four boundaries
        /*
        check_boundary(
            Vector2::new(-half_length, half_width),
            Vector2::new(half_length, half_width),
        );
        check_boundary(
            Vector2::new(-half_length, -half_width),
            Vector2::new(half_length, -half_width),
        );
        check_boundary(
            Vector2::new(-half_length, -half_width),
            Vector2::new(-half_length, half_width),
        );
        check_boundary(
            Vector2::new(half_length, -half_width),
            Vector2::new(half_length, half_width),
        );
        */
        // }

        closest_intersection.map(|(_, point)| point)
    }

    /// Predict the position of the ball at time `t` seconds in the future assuming it
    /// moves in a straight line.
    pub fn predict_ball_position(&self, t: f64) -> Option<BallPrediction> {
        if let Some(ball) = &self.ball {
            let current_pos = ball.position.xy();
            let dp = ball.velocity.xy() * t;
            let distance = dp.norm();
            let pred_pos = ball.position.xy() + dp;

            if let Some(intersection) = self.cast_ray(current_pos, dp) {
                if (intersection - current_pos).norm() < distance {
                    Some(BallPrediction::Collision(intersection))
                } else {
                    Some(BallPrediction::Linear(pred_pos))
                }
            } else {
                Some(BallPrediction::Linear(pred_pos))
            }
        } else {
            None
        }
    }

    pub fn get_best_kick_direction(&self, player_id: PlayerId) -> Angle {
        let player = self.get_player(player_id);
        let mut score = 0.0;
        let mut best_angle: Angle = Angle::default();
        for theta in (-90..90).step_by(20) {
            let angle = Angle::from_degrees(theta as f64);
            let direction = player.position + angle.to_vector() * 1000.0;

            let mut min_dist: f64 = 0.0;
            for p in self.opp_players.iter() {
                let dist = distance_to_line(player.position, direction, p.position);
                if dist < min_dist {
                    min_dist = dist;
                }
            }

            if min_dist > score {
                score = min_dist;
                best_angle = angle;
            }
        }
        best_angle
    }

    pub fn get_obstacles_for_player(&self, role: RoleType) -> Vec<Obstacle> {
        if let Some(field_geom) = &self.field_geom {
            let mut obstacles = vec![];

            match self.current_game_state.game_state {
                GameState::Stop => {
                    if let Some(ball) = &self.ball {
                        obstacles.push(Obstacle::Circle {
                            center: ball.position.xy(),
                            radius: STOP_BALL_AVOIDANCE_RADIUS,
                        });
                    }
                }
                GameState::BallReplacement(target_ball_pos) => {
                    if let Some(ball) = &self.ball {
                        obstacles.push(Obstacle::Circle {
                            center: ball.position.xy(),
                            radius: STOP_BALL_AVOIDANCE_RADIUS,
                        });
                        obstacles.push(Obstacle::Line {
                            start: ball.position.xy(),
                            end: target_ball_pos,
                        });
                    }
                }
                GameState::Kickoff | GameState::PrepareKickoff => match role {
                    RoleType::KickoffKicker => {}
                    _ => {
                        // Add center circle for non kicker robots
                        // TODO: fix
                        obstacles.push(Obstacle::Circle {
                            center: Vector2::zeros(),
                            radius: field_geom.center_circle_radius,
                        });
                    }
                },
                GameState::PreparePenalty => {}
                GameState::FreeKick => {}
                GameState::Penalty => {}
                GameState::PenaltyRun => {}
                GameState::Run | GameState::Halt | GameState::Timeout | GameState::Unknown => {
                    // Nothing to do
                }
            };

            obstacles
        } else {
            vec![]
        }
    }
}

#[derive(Debug, Clone)]
pub enum Avoid {
    Line { start: Vector2, end: Vector2 },
    Circle { center: Vector2 },
    Rectangle { min: Vector2, max: Vector2 },
}

impl Avoid {
    fn distance_to(&self, pos: Vector2) -> f64 {
        match self {
            Avoid::Line { start, end } => distance_to_line(*start, *end, pos),
            Avoid::Circle { center } => (center - pos).norm(),
            Avoid::Rectangle { min, max } => {
                // Calculate distance from point to rectangle
                let dx = if pos.x < min.x {
                    min.x - pos.x
                } else if pos.x > max.x {
                    pos.x - max.x
                } else {
                    0.0
                };
                let dy = if pos.y < min.y {
                    min.y - pos.y
                } else if pos.y > max.y {
                    pos.y - max.y
                } else {
                    0.0
                };
                (dx * dx + dy * dy).sqrt()
            }
        }
    }

    fn intersects_line(&self, start: Vector2, end: Vector2) -> bool {
        match self {
            Avoid::Line {
                start: line_start,
                end: line_end,
            } => {
                let line_seg = *line_end - *line_start;
                let test_seg = end - start;

                let denom = line_seg.x * test_seg.y - line_seg.y * test_seg.x;
                if denom.abs() < 1e-10 {
                    return false; // Lines are parallel
                }

                let t = ((start.x - line_start.x) * test_seg.y
                    - (start.y - line_start.y) * test_seg.x)
                    / denom;
                let u = ((start.x - line_start.x) * line_seg.y
                    - (start.y - line_start.y) * line_seg.x)
                    / denom;

                (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u)
            }
            Avoid::Circle { center } => {
                // Calculate minimum distance from circle center to line segment
                let line_vec = end - start;
                let line_len_sq = line_vec.norm_squared();

                if line_len_sq == 0.0 {
                    return false;
                }

                let t = ((center - start).dot(&line_vec) / line_len_sq).clamp(0.0, 1.0);
                let closest_point = start + line_vec * t;
                let dist_to_line = (center - closest_point).norm();

                dist_to_line < PLAYER_RADIUS
            }
            Avoid::Rectangle { min, max } => {
                // Check if line segment intersects with rectangle using line-rectangle intersection
                let line_dir = end - start;

                // Check intersection with each edge of the rectangle
                let edges = [
                    (*min, Vector2::new(max.x, min.y)), // bottom edge
                    (Vector2::new(max.x, min.y), *max), // right edge
                    (*max, Vector2::new(min.x, max.y)), // top edge
                    (Vector2::new(min.x, max.y), *min), // left edge
                ];

                for (edge_start, edge_end) in edges.iter() {
                    let edge_dir = *edge_end - *edge_start;
                    let denom = line_dir.x * edge_dir.y - line_dir.y * edge_dir.x;

                    if denom.abs() > 1e-10 {
                        let t = ((edge_start.x - start.x) * edge_dir.y
                            - (edge_start.y - start.y) * edge_dir.x)
                            / denom;
                        let u = ((edge_start.x - start.x) * line_dir.y
                            - (edge_start.y - start.y) * line_dir.x)
                            / denom;

                        if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
                            return true;
                        }
                    }
                }

                // Check if either endpoint is inside the rectangle
                let start_inside =
                    start.x >= min.x && start.x <= max.x && start.y >= min.y && start.y <= max.y;
                let end_inside =
                    end.x >= min.x && end.x <= max.x && end.y >= min.y && end.y <= max.y;

                start_inside || end_inside
            }
        }
    }
}

pub fn nearest_safe_pos(
    entity_to_avoid: Avoid,
    min_distance: f64,
    initial_pos: Vector2,
    target_pos: Vector2,
    max_radius: i32,
    field: &FieldGeometry,
) -> Vector2 {
    // Fast path: if the target is already clear of the avoided region (and on the
    // field) and the straight line to it doesn't cross the region, keep it
    // unchanged. The polar grid below samples on a 50 mm radial step with a
    // closeness penalty, so for any target within ~one grid step of the robot it
    // otherwise snaps back to `initial_pos` (the robot's *current* position) —
    // freezing every fine final approach (e.g. the acquire staging point) ~30 mm
    // short of its goal, even when no keep-out is anywhere near.
    if is_pos_in_field(target_pos, field)
        && entity_to_avoid.distance_to(target_pos) > min_distance
        && !entity_to_avoid.intersects_line(initial_pos, target_pos)
    {
        return target_pos;
    }

    let mut best_pos = target_pos;
    let mut best_loss = f64::INFINITY;
    let min_theta = 0;
    let max_theta = 360;

    for theta in (min_theta..max_theta).step_by(10) {
        let theta = Angle::from_degrees(theta as f64);
        for radius in (0..max_radius).step_by(50) {
            let position = initial_pos + theta.to_vector() * (radius as f64);
            if is_pos_in_field(position, field)
                && entity_to_avoid.distance_to(position) > min_distance
            {
                let distnorm = (initial_pos - target_pos).norm() + 1e-6;
                let dist_score = (position - target_pos).norm() / distnorm;
                let closeness_score = (position - initial_pos).norm() / distnorm;
                let not_gay_score =
                    entity_to_avoid.intersects_line(initial_pos, position) as i32 as f64;
                let loss = dist_score + closeness_score * 0.5 + not_gay_score * 0.0;

                if loss < best_loss {
                    best_loss = loss;
                    best_pos = position;
                }
            }
        }
    }
    if best_loss == f64::INFINITY {
        log::warn!("Could not find a safe position from {initial_pos}, avoiding {entity_to_avoid:?}, {best_loss}");
    }

    best_pos
}

pub fn is_pos_in_field(pos: Vector2, field: &FieldGeometry) -> bool {
    const MARGIN: f64 = 0.0;
    // check if pos outside field
    if pos.x.abs() > field.boundary_width + field.field_length / 2.0 - MARGIN
        || pos.y.abs() > field.boundary_width + field.field_width / 2.0 - MARGIN
    {
        return false;
    }

    true
}

pub fn mock_world_data() -> WorldData {
    WorldData {
        t_received: 0.0,
        t_capture: 0.0,
        dt: 1.0,
        blue_team: vec![PlayerData {
            id: PlayerId::new(0),
            position: Vector2::new(1000.0, 1000.0),
            timestamp: 0.0,
            raw_position: Vector2::new(1000.0, 1000.0),
            velocity: Vector2::zeros(),
            yaw: Angle::default(),
            raw_yaw: Angle::default(),
            angular_speed: 0.0,
            position_noise: 0.0,
            primary_status: Some(SysStatus::Ready),
            kicker_cap_voltage: Some(0.0),
            kicker_temp: Some(0.0),
            pack_voltages: Some([0.0, 0.0]),
            breakbeam_ball_detected: false,
            has_ball: false,
            tof_ok: false,
            tof_ball_detected: false,
            tof_xy: None,
            tof_confidence: None,
            tof_backup_enabled: false,
            tof_backup_ball_detected: false,
            imu_status: Some(SysStatus::Ready),
            imu_readings: Some([0.0; 6]),
            kicker_status: Some(SysStatus::Standby),
            reflex_kick_state: None,
            skill: None,
            handicaps: HashSet::new(),
            sideline: None,
            open_loop_delay: None,
        }],
        yellow_team: vec![PlayerData {
            id: PlayerId::new(1),
            position: Vector2::new(-1000.0, -1000.0),
            timestamp: 0.0,
            raw_position: Vector2::new(-1000.0, -1000.0),
            velocity: Vector2::zeros(),
            yaw: Angle::default(),
            raw_yaw: Angle::default(),
            angular_speed: 0.0,
            position_noise: 0.0,
            primary_status: Some(SysStatus::Ready),
            kicker_cap_voltage: Some(0.0),
            kicker_temp: Some(0.0),
            pack_voltages: Some([0.0, 0.0]),
            breakbeam_ball_detected: false,
            has_ball: false,
            tof_ok: false,
            tof_ball_detected: false,
            tof_xy: None,
            tof_confidence: None,
            tof_backup_enabled: false,
            tof_backup_ball_detected: false,
            imu_status: Some(SysStatus::Ready),
            imu_readings: Some([0.0; 6]),
            kicker_status: Some(SysStatus::Standby),
            reflex_kick_state: None,
            skill: None,
            handicaps: HashSet::new(),
            sideline: None,
            open_loop_delay: None,
        }],
        field_geom: Default::default(),
        ball: None,
        game_state: RawGameStateData {
            game_state: GameState::Run,
            operating_team: TeamColor::Blue,
            freekick_kicker: None,
            double_touch_barred: None,
            blue_team_yellow_cards: 0,
            yellow_team_yellow_cards: 0,
            blue_team_score: 0,
            yellow_team_score: 0,
            blue_team_max_allowed_bots: 6,
            yellow_team_max_allowed_bots: 6,
            blue_team_keeper_id: None,
            yellow_team_keeper_id: None,
            stage: None,
            stage_time_left: None,
            action_time_remaining: None,
            next_command: None,
            predicted_next_game_state: None,
            predicted_operating_team: None,
            status_message: None,
            blue_team_name: None,
            yellow_team_name: None,
        },
        side_assignment: SideAssignment::YellowOnPositive,
        ball_on_blue_side: None,
        ball_on_yellow_side: None,
        autoref_info: None,
        possession: Possession::default(),
    }
}

pub fn mock_team_data() -> TeamData {
    TeamData {
        sidelined_players: Vec::new(),
        own_players: vec![PlayerData {
            id: PlayerId::new(0),
            position: Vector2::new(1000.0, 1000.0),
            timestamp: 0.0,
            raw_position: Vector2::new(1000.0, 1000.0),
            velocity: Vector2::zeros(),
            yaw: Angle::default(),
            raw_yaw: Angle::default(),
            angular_speed: 0.0,
            position_noise: 0.0,
            primary_status: Some(SysStatus::Ready),
            kicker_cap_voltage: Some(0.0),
            kicker_temp: Some(0.0),
            pack_voltages: Some([0.0, 0.0]),
            breakbeam_ball_detected: false,
            has_ball: false,
            tof_ok: false,
            tof_ball_detected: false,
            tof_xy: None,
            tof_confidence: None,
            tof_backup_enabled: false,
            tof_backup_ball_detected: false,
            imu_status: Some(SysStatus::Ready),
            imu_readings: Some([0.0; 6]),
            kicker_status: Some(SysStatus::Standby),
            reflex_kick_state: None,
            skill: None,
            handicaps: HashSet::new(),
            sideline: None,
            open_loop_delay: None,
        }],
        opp_players: vec![PlayerData {
            id: PlayerId::new(1),
            position: Vector2::new(-1000.0, -1000.0),
            timestamp: 0.0,
            raw_position: Vector2::new(-1000.0, -1000.0),
            velocity: Vector2::zeros(),
            yaw: Angle::default(),
            raw_yaw: Angle::default(),
            angular_speed: 0.0,
            position_noise: 0.0,
            primary_status: Some(SysStatus::Ready),
            kicker_cap_voltage: Some(0.0),
            kicker_temp: Some(0.0),
            pack_voltages: Some([0.0, 0.0]),
            breakbeam_ball_detected: false,
            has_ball: false,
            tof_ok: false,
            tof_ball_detected: false,
            tof_xy: None,
            tof_confidence: None,
            tof_backup_enabled: false,
            tof_backup_ball_detected: false,
            imu_status: Some(SysStatus::Ready),
            imu_readings: Some([0.0; 6]),
            kicker_status: Some(SysStatus::Standby),
            reflex_kick_state: None,
            skill: None,
            handicaps: HashSet::new(),
            sideline: None,
            open_loop_delay: None,
        }],
        field_geom: Some(FieldGeometry {
            field_length: 9000.0,
            field_width: 6000.0,
            ..Default::default()
        }),
        t_received: 0.0,
        t_capture: 0.0,
        dt: 1.0,
        ball: None,
        current_game_state: GameStateData {
            game_state: GameState::Run,
            us_operating: true,
            yellow_cards: 0,
            freekick_kicker: None,
            double_touch_barred: None,
            max_allowed_bots: 6,
            our_keeper_id: None,
            predicted_next_game_state: None,
            predicted_us_operating: None,
        },
        ball_on_our_side: None,
        ball_on_opp_side: None,
        kicked_ball: None,
        possession: Possession::default(),
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn sidelined_robot_forks_out_of_own_players() {
        let mut wd = mock_world_data();
        wd.blue_team[0].sideline = Some(SidelineReason::RadioLost);
        let td = wd.get_team_data(TeamColor::Blue, false);
        // Radio-lost robot leaves the roster but stays present as sidelined.
        assert!(td.own_players.is_empty());
        assert_eq!(td.sidelined_players.len(), 1);
        assert_eq!(
            td.sidelined_players[0].sideline,
            Some(SidelineReason::RadioLost)
        );
        // Opponent list is unaffected by our sideline marks.
        assert_eq!(td.opp_players.len(), 1);
        // Still present for physical-presence queries (raycast/radius).
        assert_eq!(
            td.players_within_radius(Vector2::new(1000.0, 1000.0), 50.0)
                .len(),
            1
        );
    }

    #[test]
    fn test_ball_placement_equals() {
        assert!(
            GameState::BallReplacement(Vector2::zeros())
                == GameState::BallReplacement(Vector2::x())
        )
    }

    #[test]
    fn test_ball_placement_matches() {
        assert!(StrategyGameStateMacther::any_of(
            vec![GameState::BallReplacement(Vector2::zeros())].as_slice()
        )
        .matches(&GameState::BallReplacement(Vector2::x())))
    }

    #[test]
    fn test_rectangle_distance_to() {
        let rect = Avoid::Rectangle {
            min: Vector2::new(0.0, 0.0),
            max: Vector2::new(100.0, 100.0),
        };

        // Point inside rectangle
        assert_relative_eq!(
            rect.distance_to(Vector2::new(50.0, 50.0)),
            0.0,
            epsilon = 1e-6
        );

        // Point outside rectangle
        assert_relative_eq!(
            rect.distance_to(Vector2::new(150.0, 150.0)),
            70.71,
            epsilon = 0.1
        );

        // Point directly to the right
        assert_relative_eq!(
            rect.distance_to(Vector2::new(150.0, 50.0)),
            50.0,
            epsilon = 1e-6
        );

        // Point directly above
        assert_relative_eq!(
            rect.distance_to(Vector2::new(50.0, 150.0)),
            50.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_rectangle_intersects_line() {
        let rect = Avoid::Rectangle {
            min: Vector2::new(0.0, 0.0),
            max: Vector2::new(100.0, 100.0),
        };

        // Line crossing through rectangle
        assert!(rect.intersects_line(Vector2::new(-50.0, 50.0), Vector2::new(150.0, 50.0)));

        // Line completely outside rectangle
        assert!(!rect.intersects_line(Vector2::new(-50.0, -50.0), Vector2::new(-10.0, -10.0)));

        // Line with one endpoint inside rectangle
        assert!(rect.intersects_line(Vector2::new(50.0, 50.0), Vector2::new(150.0, 150.0)));

        // Line touching rectangle edge
        assert!(rect.intersects_line(Vector2::new(0.0, 0.0), Vector2::new(100.0, 0.0)));

        // Line parallel to rectangle edge but not touching
        assert!(!rect.intersects_line(Vector2::new(-50.0, -50.0), Vector2::new(50.0, -50.0)));
    }

    #[test]
    fn nearest_safe_pos_keeps_a_close_safe_target() {
        // Regression: a target ~36 mm from the robot, with the keep-out far away,
        // must be returned unchanged. The coarse polar grid (50 mm radial step +
        // closeness penalty) used to snap any sub-~37 mm target back to
        // `initial_pos`, freezing the acquire staging approach ~30 mm short.
        let field = FieldGeometry::default();
        let rect = Avoid::Rectangle {
            min: Vector2::new(3150.0, -1000.0),
            max: Vector2::new(14500.0, 1000.0),
        }; // opponent goal-area keep-out, far from the robot below
        let initial = Vector2::new(2308.0, -1051.0);
        let target = initial + Vector2::new(0.7, 0.7).normalize() * 36.0;
        let out = nearest_safe_pos(rect, 80.0, initial, target, 4000, &field);
        assert_relative_eq!(out.x, target.x, epsilon = 1e-6);
        assert_relative_eq!(out.y, target.y, epsilon = 1e-6);
    }

    #[test]
    fn nearest_safe_pos_relocates_a_target_inside_the_keepout() {
        // The grid search still runs when the target is genuinely unsafe: the
        // result must land outside the keep-out.
        let field = FieldGeometry::default();
        let center = Vector2::new(0.0, 0.0);
        let ball = Avoid::Circle { center };
        let initial = Vector2::new(900.0, 0.0);
        let target = center; // right on the ball
        let out = nearest_safe_pos(ball.clone(), 800.0, initial, target, 4000, &field);
        assert!(
            ball.distance_to(out) >= 800.0,
            "expected >=800mm from ball, got {}",
            ball.distance_to(out)
        );
    }

    #[test]
    fn effective_keeper_id_gc_wins_else_fallback() {
        let ids = |v: &[u32]| v.iter().map(|i| PlayerId::new(*i)).collect::<Vec<_>>();
        // GC designation is ground truth, even if that robot is not on the roster.
        assert_eq!(
            effective_keeper_id(Some(PlayerId::new(2)), ids(&[0, 1])),
            Some(PlayerId::new(2))
        );
        // No GC: prefer 4, then 0, then the lowest id present.
        assert_eq!(
            effective_keeper_id(None, ids(&[0, 3, 4, 5])),
            Some(PlayerId::new(4))
        );
        assert_eq!(
            effective_keeper_id(None, ids(&[0, 3, 5])),
            Some(PlayerId::new(0))
        );
        assert_eq!(
            effective_keeper_id(None, ids(&[5, 3])),
            Some(PlayerId::new(3))
        );
        assert_eq!(effective_keeper_id(None, ids(&[])), None);
    }
}
