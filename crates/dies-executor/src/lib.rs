use std::{collections::HashMap, path::PathBuf, time::Duration};

use anyhow::Result;
use dies_basestation_client::BasestationHandle;
use dies_core::{
    Angle, BallData, DebugSubscriber, ExecutorInfo, ExecutorSettings, GameState, GcOverride,
    GcSimCommand, PlayerCmd, PlayerData, PlayerFeedbackMsg, PlayerId, PlayerOverrideCommand,
    SideAssignment, SidelineReason, SimulatorCmd, TeamColor, TeamPlayerId, TeamSpecificSettings,
    Vector2, Vector3, WorldData, WorldInstant, WorldUpdate,
};
use dies_logger::worker;
use dies_protos::{
    ssl_gc_referee_message::Referee, ssl_vision_wrapper::SSL_WrapperPacket,
    ssl_vision_wrapper_tracked::TrackerWrapperPacket,
};
use dies_simulator::Simulation;
use dies_ssl_client::{SslMessage, VisionClient};
use dies_world::WorldTracker;
use gc_client::GcClient;
pub use handle::{ControlMsg, ExecutorHandle};
use protobuf::Message as _;
use tokio::sync::{broadcast, mpsc, oneshot, watch};

mod announcer;
pub mod control;
mod gc_client;
mod handle;
pub mod headless;
pub mod notebook;
pub mod skills;
pub mod strategy_host;

pub use control::*;
pub use headless::{
    EndReason, GoalEvent, HeadlessConfig, MatchResult, ScriptedEvent, ScriptedEventKind,
};

const SIMULATION_DT: Duration = Duration::from_micros(1_000_000 / 60); // 60 Hz
const CMD_INTERVAL: Duration = Duration::from_micros(1_000_000 / 40); // 40 Hz

enum Environment {
    Live {
        ssl_client: VisionClient,
        bs_client: BasestationHandle,
    },
    Simulation {
        simulator: Simulation,
    },
}

struct TeamMap {
    blue_team: Option<TeamController>,
    yellow_team: Option<TeamController>,
    side_assignment: SideAssignment,
}

impl TeamMap {
    fn new(side_assignment: SideAssignment) -> Self {
        Self {
            blue_team: None,
            yellow_team: None,
            side_assignment,
        }
    }

    fn activate_team(&mut self, team_color: TeamColor, settings: &ExecutorSettings) {
        if team_color == TeamColor::Blue {
            if self.blue_team.is_none() {
                self.blue_team = Some(TeamController::new(settings, TeamColor::Blue));
            }
        } else {
            if self.yellow_team.is_none() {
                self.yellow_team = Some(TeamController::new(settings, TeamColor::Yellow));
            }
        }
    }

    fn update_settings(&mut self, settings: &ExecutorSettings) {
        if let Some(controller) = &mut self.blue_team {
            controller.update_controller_settings(settings);
        }
        if let Some(controller) = &mut self.yellow_team {
            controller.update_controller_settings(settings);
        }
    }

    fn swap_teams(&mut self) {
        std::mem::swap(&mut self.blue_team, &mut self.yellow_team);
    }

    /// Get all player commands from active controllers
    fn get_all_commands(&mut self) -> Vec<(TeamColor, PlayerCmd)> {
        let mut commands = Vec::new();

        if let Some(controller) = &mut self.blue_team {
            commands.extend(
                controller
                    .commands(self.side_assignment, TeamColor::Blue)
                    .into_iter()
                    .map(|c| (TeamColor::Blue, c)),
            );
        }
        if let Some(controller) = &mut self.yellow_team {
            commands.extend(
                controller
                    .commands(self.side_assignment, TeamColor::Yellow)
                    .into_iter()
                    .map(|c| (TeamColor::Yellow, c)),
            );
        }

        commands
    }

    /// Per-player world-frame velocity setpoints from both controllers, for
    /// command feedforward into the tracker. World frame == vision/absolute frame
    /// (the controllers' inputs are already transformed strategy→world).
    fn target_velocities_global(&self) -> Vec<(TeamColor, PlayerId, dies_core::Vector2)> {
        let mut out = Vec::new();
        for (color, ctrl) in [
            (TeamColor::Blue, &self.blue_team),
            (TeamColor::Yellow, &self.yellow_team),
        ] {
            if let Some(ctrl) = ctrl {
                out.extend(
                    ctrl.target_velocities_global()
                        .into_iter()
                        .map(|(id, v)| (color, id, v)),
                );
            }
        }
        out
    }

    /// Players kicking this tick on both teams (efference copy for possession).
    fn kicking_players(&self) -> Vec<(TeamColor, PlayerId)> {
        let mut out = Vec::new();
        for (color, ctrl) in [
            (TeamColor::Blue, &self.blue_team),
            (TeamColor::Yellow, &self.yellow_team),
        ] {
            if let Some(ctrl) = ctrl {
                out.extend(ctrl.kicking_players().into_iter().map(|id| (color, id)));
            }
        }
        out
    }

    /// Get list of active team colors
    fn active_teams(&self) -> Vec<TeamColor> {
        let mut teams = Vec::new();

        // Check if team_a is active and get its color
        if self.blue_team.is_some() {
            teams.push(TeamColor::Blue);
        }

        // Check if team_b is active and get its color
        if self.yellow_team.is_some() {
            teams.push(TeamColor::Yellow);
        }

        teams
    }

    /// Update all active team controllers with team-specific data
    fn update(
        &mut self,
        world_data: &dies_core::WorldData,
        manual_override: HashMap<(TeamColor, PlayerId), PlayerControlInput>,
        blue_gc_override: dies_core::GcOverride,
        yellow_gc_override: dies_core::GcOverride,
    ) {
        // Update blue team controller if active
        if let Some(controller) = &mut self.blue_team {
            let team_data = world_data.get_team_data(TeamColor::Blue, blue_gc_override);
            controller.update(
                TeamColor::Blue,
                world_data.side_assignment,
                team_data,
                manual_override
                    .iter()
                    .filter_map(|((color, id), input)| {
                        if *color == TeamColor::Blue {
                            Some((*id, input.clone()))
                        } else {
                            None
                        }
                    })
                    .collect(),
            );
        }

        // Update yellow team controller if active
        if let Some(controller) = &mut self.yellow_team {
            let team_data = world_data.get_team_data(TeamColor::Yellow, yellow_gc_override);
            controller.update(
                TeamColor::Yellow,
                world_data.side_assignment,
                team_data,
                manual_override
                    .iter()
                    .filter_map(|((color, id), input)| {
                        if *color == TeamColor::Yellow {
                            Some((*id, input.clone()))
                        } else {
                            None
                        }
                    })
                    .collect(),
            );
        }
    }
}

struct PlayerOverrideState {
    frame_counter: u32,
    current_command: PlayerOverrideCommand,
}

impl PlayerOverrideState {
    const VELOCITY_TIMEOUT: u32 = 10;

    pub fn new() -> Self {
        Self {
            frame_counter: 0,
            current_command: PlayerOverrideCommand::Stop,
        }
    }

    pub fn set_cmd(&mut self, cmd: PlayerOverrideCommand) {
        self.current_command = cmd;
        self.frame_counter = 0;
    }

    pub fn advance(&mut self) -> PlayerControlInput {
        let input = match self.current_command {
            PlayerOverrideCommand::Stop => PlayerControlInput::new(),
            PlayerOverrideCommand::MoveTo {
                position,
                yaw,
                dribble_speed,
                arm_kick,
            } => PlayerControlInput {
                position: Some(position),
                yaw,
                dribbling_speed: dribble_speed,
                kicker: if arm_kick {
                    KickerControlInput::Arm
                } else {
                    KickerControlInput::default()
                },
                ..Default::default()
            },
            PlayerOverrideCommand::LocalVelocity {
                velocity,
                yaw,
                dribble_speed,
                arm_kick,
            } => PlayerControlInput {
                velocity: Velocity::local(velocity),
                yaw,
                dribbling_speed: dribble_speed,
                kicker: if arm_kick {
                    KickerControlInput::Arm
                } else {
                    KickerControlInput::default()
                },
                ..Default::default()
            },
            PlayerOverrideCommand::GlobalVelocity {
                velocity,
                yaw,
                dribble_speed,
                arm_kick,
            } => PlayerControlInput {
                velocity: Velocity::global(velocity),
                yaw,
                dribbling_speed: dribble_speed,
                kicker: if arm_kick {
                    KickerControlInput::Arm
                } else {
                    KickerControlInput::default()
                },
                ..Default::default()
            },
            PlayerOverrideCommand::Kick { speed } => PlayerControlInput {
                kicker: KickerControlInput::Kick,
                kick_speed: Some(speed),
                ..Default::default()
            },
            PlayerOverrideCommand::DischargeKicker => PlayerControlInput {
                kicker: KickerControlInput::Disarm,
                ..Default::default()
            },
            PlayerOverrideCommand::SetFanSpeed { speed } => PlayerControlInput {
                fan_speed: Some(speed),
                ..Default::default()
            },
        };

        // Advance the frame counter
        self.frame_counter += 1;
        match self.current_command {
            PlayerOverrideCommand::LocalVelocity { .. }
            | PlayerOverrideCommand::GlobalVelocity { .. } => {
                if self.frame_counter > Self::VELOCITY_TIMEOUT {
                    self.current_command = PlayerOverrideCommand::Stop;
                    self.frame_counter = 0;
                }
            }
            PlayerOverrideCommand::Kick { .. } => {
                self.current_command = PlayerOverrideCommand::Stop
            }
            _ => {}
        };

        input
    }
}

/// The central component of the framework. It contains all state and logic needed to
/// run a match -- processing vision and referee messages, executing the strategy, and
/// sending commands to the robots.
///
/// The executor can be used in 3 different regimes: externally driven, automatic, and
/// simulation. Now supports team-agnostic operation with 0, 1, or 2 active team controllers.
/// Manually-driven game state for the in-app "simulated GC". In live mode the
/// GC panel's [`GcSimCommand`]s drive this instead of the simulator's auto-ref;
/// it takes effect only for teams with `ignore_gc` on (see
/// [`Executor::team_gc_override`]), overriding whatever the real GC says.
#[derive(Clone, Copy, Debug)]
struct ManualGc {
    game_state: GameState,
    /// Absolute-color team the GC designated as operating (kicking) for restart
    /// states, if any.
    operating_team: Option<TeamColor>,
}

impl Default for ManualGc {
    fn default() -> Self {
        Self {
            game_state: GameState::Run,
            operating_team: None,
        }
    }
}

impl ManualGc {
    /// Apply a GC command, mirroring the real GC → [`GameState`] mapping in
    /// `dies-world`'s `GameStateTracker`. Kickoff is two-step (`KickOff` →
    /// `PrepareKickoff`, then `NormalStart` → `Kickoff`); free-kick and penalty
    /// are one-shot. Restart states persist until the operator resumes with
    /// `NormalStart`/`ForceStart`. `BallPlacement`/`YellowCard` have no per-team
    /// game-state effect and are ignored in live mode.
    fn apply(&mut self, cmd: GcSimCommand) {
        let (game_state, operating_team) = match cmd {
            GcSimCommand::Stop => (GameState::Stop, None),
            GcSimCommand::Halt => (GameState::Halt, None),
            GcSimCommand::ForceStart => (GameState::Run, None),
            GcSimCommand::NormalStart => match self.game_state {
                GameState::PrepareKickoff => (GameState::Kickoff, self.operating_team),
                _ => (GameState::Run, None),
            },
            GcSimCommand::KickOff { team_color } => (GameState::PrepareKickoff, Some(team_color)),
            GcSimCommand::DirectFree { team_color } => (GameState::FreeKick, Some(team_color)),
            GcSimCommand::Penalty { team_color } => (GameState::Penalty, Some(team_color)),
            // No per-team game-state effect; ignored in live mode.
            GcSimCommand::BallPlacement { .. } | GcSimCommand::YellowCard { .. } => return,
        };
        self.game_state = game_state;
        self.operating_team = operating_team;
    }
}

pub struct Executor {
    tracker: WorldTracker,
    /// Strategy host. None only if the executor was constructed without strategy
    /// support (currently never).
    strategy_host: Option<strategy_host::StrategyHost>,
    team_controllers: TeamMap,
    gc_client: GcClient,
    environment: Option<Environment>,
    manual_override: HashMap<(TeamColor, PlayerId), PlayerOverrideState>,
    /// Manually-driven game state for the in-app "simulated GC" (live mode only;
    /// applies to teams with `ignore_gc` on). Defaults to free play (`Run`).
    manual_gc: ManualGc,
    /// Latest raw basestation feedback per robot, captured at intake and logged
    /// verbatim (as JSON) each frame into the `player_feedback` table.
    latest_feedback: HashMap<(TeamColor, PlayerId), PlayerFeedbackMsg>,
    /// Online open-loop delay estimator per own robot (command → visible motion).
    open_loop_detectors: HashMap<(TeamColor, PlayerId), control::DelayDetector>,
    update_tx: broadcast::Sender<WorldUpdate>,
    /// Notebook (embedded JS) command source; inert until the first cell runs.
    notebook: notebook::NotebookHost,
    notebook_tx: broadcast::Sender<dies_core::NotebookEvent>,
    command_tx: mpsc::UnboundedSender<ControlMsg>,
    command_rx: mpsc::UnboundedReceiver<ControlMsg>,
    paused_tx: watch::Sender<bool>,
    info_channel_rx: mpsc::UnboundedReceiver<oneshot::Sender<ExecutorInfo>>,
    info_channel_tx: mpsc::UnboundedSender<oneshot::Sender<ExecutorInfo>>,
    settings: ExecutorSettings,
    /// Monotonic world-frame counter, minted on each vision-driven frame. Used
    /// as the log join key and exposed to the UI via `WorldUpdate`.
    frame_counter: u64,
    /// Timestamp (relative seconds) of the most recent frame, used to stamp
    /// settings/events/markers logged between frames.
    last_t: f64,
    /// Whether to record raw received vision/GC wire bytes (ground truth). On
    /// for live matches via `DIES_LOG_RAW`, off for routine testing.
    log_raw_vision: bool,
    /// Builds the scrolling announcer commentary feed sent to the UI.
    announcer: announcer::Announcer,
    /// While paused, a frozen copy of the world that Sim Edit manipulations
    /// (teleport/add/remove) patch in place, so the UI reflects edits without
    /// stepping the sim, advancing the clock, running strategy, or logging.
    /// Reset to `None` whenever the executor resumes stepping.
    paused_edit_world: Option<WorldData>,
    /// Last GC team names patched into the log `meta.json`, to avoid rewriting it
    /// every frame (only rewrite when a name actually changes).
    last_logged_team_names: Option<(Option<String>, Option<String>)>,
    /// Whether the one-shot "MATCH START" marker has been dropped yet (match runs
    /// only, once the GC team names are known).
    match_marker_dropped: bool,
    /// Last `(blue, yellow)` score seen from the GC referee, to detect goals on
    /// live matches by score increment (sim goals come from the simulator's
    /// referee events instead). `None` until the first referee message arrives.
    last_gc_score: Option<(u32, u32)>,
    /// True when running against the simulator. Sim goals are logged from the
    /// simulator's `sim_referee` events, so GC-score goal detection is skipped
    /// to avoid double-counting (the sim GC message also carries the score).
    is_simulation: bool,
}

impl Executor {
    /// Create a new executor.
    pub fn new_live(
        settings: ExecutorSettings,
        ssl_client: VisionClient,
        bs_client: BasestationHandle,
    ) -> Self {
        Self::build(
            settings,
            Environment::Live {
                ssl_client,
                bs_client,
            },
        )
    }

    /// Create a new simulation executor.
    pub fn new_simulation(settings: ExecutorSettings, simulator: Simulation) -> Self {
        Self::build(settings, Environment::Simulation { simulator })
    }

    fn build(settings: ExecutorSettings, mut environment: Environment) -> Self {
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let (update_tx, _) = broadcast::channel(16);
        let (notebook_tx, _) = broadcast::channel(1024);
        let (paused_tx, _) = watch::channel(false);
        let (info_channel_tx, info_channel_rx) = mpsc::unbounded_channel();

        // Apply any persisted skill-tunable overrides into the global cells before
        // the first tick (empty map = compile-time defaults, so self-play is
        // unaffected and deterministic).
        crate::skills::apply_skill_tunables(&settings.skill_tunables);

        let mut strategy_host =
            strategy_host::StrategyHost::new(strategy_host::StrategyHostConfig {
                strategies_dir: settings
                    .strategies_dir
                    .clone()
                    .unwrap_or_else(|| std::path::PathBuf::from("target/debug")),
                blue_strategy: settings.team_configuration.blue_strategy.clone(),
                yellow_strategy: settings.team_configuration.yellow_strategy.clone(),
                side_assignment: settings.team_configuration.side_assignment,
                hot_reload: settings.hot_reload,
                blocking: settings.strategy_blocking,
            });
        if let Some(ref name) = settings.team_configuration.blue_strategy {
            strategy_host.set_strategy(TeamColor::Blue, Some(name.clone()));
        }
        if let Some(ref name) = settings.team_configuration.yellow_strategy {
            strategy_host.set_strategy(TeamColor::Yellow, Some(name.clone()));
        }
        // Seed CLI-provided initial strategy params (e.g. match mode's
        // `warmup=true`) into the host cache. The cache is pushed to each
        // strategy once it reaches `Ready`, and re-pushed on reconnect/hot-reload.
        if !settings.initial_strategy_params.is_empty() {
            strategy_host
                .set_strategy_params(TeamColor::Blue, settings.initial_strategy_params.clone());
            strategy_host
                .set_strategy_params(TeamColor::Yellow, settings.initial_strategy_params.clone());
        }

        let mut team_controllers = TeamMap::new(settings.team_configuration.side_assignment);
        let mut controlled_teams = Vec::new();
        if settings.team_configuration.blue_active {
            team_controllers.activate_team(TeamColor::Blue, &settings);
            controlled_teams.push(TeamColor::Blue);
        }
        if settings.team_configuration.yellow_active {
            team_controllers.activate_team(TeamColor::Yellow, &settings);
            controlled_teams.push(TeamColor::Yellow);
        }
        let is_simulation = matches!(environment, Environment::Simulation { .. });
        if let Environment::Simulation { simulator } = &mut environment {
            simulator.set_controlled_teams(&controlled_teams);
        }

        Self {
            tracker: WorldTracker::new(&settings, &controlled_teams, settings.allow_no_vision),
            strategy_host: Some(strategy_host),
            team_controllers,
            gc_client: GcClient::new(),
            environment: Some(environment),
            manual_override: HashMap::new(),
            manual_gc: ManualGc::default(),
            latest_feedback: HashMap::new(),
            open_loop_detectors: HashMap::new(),
            command_tx,
            command_rx,
            update_tx,
            notebook: notebook::NotebookHost::new(notebook_tx.clone()),
            notebook_tx,
            paused_tx,
            info_channel_rx,
            info_channel_tx,
            settings,
            frame_counter: 0,
            last_t: 0.0,
            log_raw_vision: std::env::var("DIES_LOG_RAW")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            announcer: announcer::Announcer::new(),
            paused_edit_world: None,
            last_logged_team_names: None,
            match_marker_dropped: false,
            last_gc_score: None,
            is_simulation,
        }
    }

    /// Update the executor with a vision message.
    pub fn update_from_vision_msg(&mut self, message: SSL_WrapperPacket, time: WorldInstant) {
        // The processed world is recorded as a single frame in
        // `update_team_controller`; the raw packet is recorded only when the
        // gated raw-vision stream is enabled.
        if self.log_raw_vision {
            if let Ok(bytes) = message.write_to_bytes() {
                worker::log_raw(self.frame_counter, self.last_t, "vision", &bytes);
            }
        }
        self.tracker.update_from_vision(&message, time);
        self.update_team_controller(true);
    }

    /// Update the executor with a referee message.
    pub fn update_from_gc_msg(&mut self, message: Referee) {
        if self.log_raw_vision {
            if let Ok(bytes) = message.write_to_bytes() {
                worker::log_raw(self.frame_counter, self.last_t, "referee", &bytes);
            }
        }
        self.tracker.update_from_referee(&message);
        // Persist the full GC event stream (command transitions + autoref game
        // events) to the log; the announcer's dedup gates what's new so each
        // fires once. Works for sim (commands only) and live (commands + events).
        for rec in self.announcer.on_referee(&message, self.last_t) {
            self.log_event(rec.event_type, rec.payload);
        }

        // Detect goals on live matches by watching the GC score. A goal is
        // logged as a `goal` event (which the replay scrubber reads) so live logs
        // carry recoverable goals. Only strict increments count; a decrease is a
        // GC reset and just rebaselines. Skipped in sim, where goals already come
        // from `sim_referee` events (the sim GC message carries the score too, so
        // running this there would double-count).
        if !self.is_simulation {
            let (blue_score, yellow_score) = (message.blue.score(), message.yellow.score());
            if let Some((prev_blue, prev_yellow)) = self.last_gc_score {
                if blue_score > prev_blue {
                    self.log_event(
                        "goal",
                        serde_json::json!({ "team": "Blue", "blue": blue_score, "yellow": yellow_score }),
                    );
                }
                if yellow_score > prev_yellow {
                    self.log_event(
                        "goal",
                        serde_json::json!({ "team": "Yellow", "blue": blue_score, "yellow": yellow_score }),
                    );
                }
            }
            self.last_gc_score = Some((blue_score, yellow_score));
        }

        // Detect side assignment from referee message
        if let Some(blue_on_positive) = message.blue_team_on_positive_half {
            self.team_controllers.side_assignment = if blue_on_positive {
                SideAssignment::BlueOnPositive
            } else {
                SideAssignment::YellowOnPositive
            };
            self.tracker
                .set_side_assignment(self.team_controllers.side_assignment);
        }

        // A referee update re-runs the controllers but does not produce a new
        // vision frame, so it reuses the current `frame_counter` and is not
        // logged as a separate frame.
        self.update_team_controller(false);
    }

    pub fn update_from_tracker_msg(&mut self, mut message: TrackerWrapperPacket) {
        if let Some(tracked_frame) = message.tracked_frame.take() {
            self.tracker.update_from_tracker(tracked_frame);
        }
    }

    /// Update the executor with a feedback message from the robots.
    pub fn update_from_bs_msg(
        &mut self,
        team_color: TeamColor,
        message: PlayerFeedbackMsg,
        time: WorldInstant,
    ) {
        self.latest_feedback
            .insert((team_color, message.id), message);
        self.tracker
            .update_from_feedback(team_color, &message, time);
    }

    /// Get the currently active player commands.
    pub fn player_commands(&mut self) -> Vec<(TeamColor, PlayerCmd)> {
        self.team_controllers.get_all_commands()
    }

    /// Get the GC messages that need to be sent. This will remove the messages from the
    /// internal queue.
    pub fn gc_messages(&mut self) -> Vec<Referee> {
        self.gc_client.messages()
    }

    /// Get runtime information about the executor.
    pub fn info(&self) -> ExecutorInfo {
        ExecutorInfo {
            paused: *self.paused_tx.borrow(),
            manual_controlled_players: self
                .manual_override
                .keys()
                .map(|(team_color, player_id)| TeamPlayerId {
                    team_color: *team_color,
                    player_id: *player_id,
                })
                .collect(),
            active_teams: self.team_controllers.active_teams(),
            strategy_params: self
                .strategy_host
                .as_ref()
                .map(|h| h.param_state())
                .unwrap_or_default(),
            skill_tunable_specs: crate::skills::all_skill_tunable_specs(),
        }
    }

    /// Run the executor in real time, either with real clients or with a simulator,
    /// depending on the configuration.
    pub async fn run_real_time(mut self) -> Result<()> {
        match self.environment.take() {
            Some(Environment::Live {
                ssl_client,
                bs_client,
            }) => self.run_rt_live(ssl_client, bs_client).await?,
            Some(Environment::Simulation { simulator }) => {
                self.run_rt_sim(simulator, SIMULATION_DT).await?
            }
            _ => unreachable!(),
        }

        Ok(())
    }

    pub fn step_simulation(&mut self, simulator: &mut Simulation, dt: Duration) -> Result<()> {
        simulator.step(dt.as_secs_f64());
        if let Some(geom) = simulator.geometry() {
            self.update_from_vision_msg(geom, simulator.time());
        }
        if let Some(det) = simulator.detection() {
            self.update_from_vision_msg(det, simulator.time());
        }
        if let Some(gc_message) = simulator.gc_message() {
            self.update_from_gc_msg(gc_message);
        }
        for ev in simulator.take_referee_events() {
            // Record sim referee events (goals, no-progress, ball-out, fouls,
            // violations) to the log so analytics can attribute goals and count
            // stoppage causes. These otherwise only reach the UI announcer.
            self.log_event(
                "sim_referee",
                serde_json::json!({
                    "kind": ev.kind,
                    "team": ev.team.map(|t| format!("{t:?}")),
                }),
            );
            self.announcer.on_sim_event(&ev, self.last_t);
        }
        // Drain feedback in a deterministic order. `feedback()` returns a
        // HashMap; iterating it directly leaks hash order into the consumer.
        // Sorting by (team, player id) keeps headless runs reproducible.
        let mut feedback: Vec<_> = simulator.feedback().into_iter().collect();
        feedback.sort_by_key(|((team, id), _)| {
            let team_key = match team {
                TeamColor::Blue => 0u8,
                TeamColor::Yellow => 1u8,
            };
            (team_key, id.as_u32())
        });
        for ((team_color, _), feedback) in feedback {
            self.update_from_bs_msg(team_color, feedback, simulator.time());
        }

        Ok(())
    }

    /// Run the executor in real time on the simulator
    async fn run_rt_sim(&mut self, mut simulator: Simulation, dt: Duration) -> Result<()> {
        let mut update_interval = tokio::time::interval(dt);
        update_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut cmd_interval = tokio::time::interval(CMD_INTERVAL);
        cmd_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        let mut paused_rx = self.paused_tx.subscribe();
        loop {
            let paused = { *paused_rx.borrow_and_update() };
            if !paused {
                // Drop any frozen Sim Edit snapshot; the live tracker drives the
                // world view again once we're stepping.
                self.paused_edit_world = None;
                tokio::select! {
                    Some(msg) = self.command_rx.recv() => {
                        match msg {
                            ControlMsg::Stop => break,
                            ControlMsg::GcCommand { command } => {
                                simulator.handle_gc_command(command);
                            }
                            ControlMsg::SimulatorCmd(cmd) => self.handle_simulator_cmd(&mut simulator, cmd),
                            ControlMsg::SetActiveTeams { .. } => {
                                log::warn!("Setting active teams is not supported mid run");
                            }
                            msg => self.handle_control_msg(msg)
                        }
                    }
                    Some(tx) = self.info_channel_rx.recv() => {
                        let _  = tx.send(self.info());
                    }
                    _ = update_interval.tick() => {
                        self.step_simulation(&mut simulator, dt)?;
                    }
                    _ = cmd_interval.tick() => {
                        for (team_color, cmd) in self.player_commands() {
                            match cmd {
                                PlayerCmd::Move(cmd) => {
                                    simulator.push_cmd(team_color, cmd);
                                }
                                PlayerCmd::GlobalMove(cmd) => {
                                    simulator.push_global_cmd(team_color, cmd);
                                }
                            }
                        }
                    }
                }
            } else {
                tokio::select! {
                    Some(msg) = self.command_rx.recv() => {
                        match msg {
                            ControlMsg::Stop => break,
                            // Apply sim manipulation while paused and broadcast a
                            // display-only world update — no stepping, no clock
                            // advance, no strategy, no logging.
                            ControlMsg::SimulatorCmd(cmd) => {
                                self.apply_sim_edit_paused(&mut simulator, cmd);
                            }
                            ControlMsg::GcCommand { command } => {
                                simulator.handle_gc_command(command);
                            }
                            ControlMsg::SetActiveTeams { .. } => {
                                log::warn!("Setting active teams is not supported mid run");
                            }
                            // Other state changes (settings, overrides, GC) don't step
                            // physics, so they're safe to apply while paused.
                            msg => self.handle_control_msg(msg),
                        }
                    }
                    Some(tx) = self.info_channel_rx.recv() => {
                        let _  = tx.send(self.info());
                    }
                    res = paused_rx.changed() => res?
                }
            }
        }

        if let Some(h) = &mut self.strategy_host {
            h.shutdown();
        }
        Ok(())
    }

    /// Run one deterministic, faster-than-realtime headless self-play match.
    ///
    /// Unlike [`Executor::run_real_time`] this is a plain synchronous loop with
    /// no wall-clock pacing: it steps the simulator at a fixed `dt` as fast as
    /// the CPU allows. Determinism comes from (a) the sim clock driving the
    /// whole tracker→strategy→controller pipeline, (b) blocking strategy IPC
    /// (set via `ExecutorSettings::strategy_blocking`) so the loop waits for
    /// each reply instead of racing a wall-clock timeout, and (c) seeded initial
    /// pose jitter in the simulator. Requires a simulation environment.
    pub fn run_headless(mut self, cfg: headless::HeadlessConfig) -> Result<headless::MatchResult> {
        let mut simulator = match self.environment.take() {
            Some(Environment::Simulation { simulator }) => simulator,
            _ => anyhow::bail!("run_headless requires a simulation environment"),
        };

        // Gate the match start on both strategies completing the Init/Ready
        // handshake (runs at startup, so wall-clock waiting is fine here).
        if let Some(host) = self.strategy_host.as_mut() {
            host.wait_until_ready(Duration::from_secs(30))
                .map_err(|e| anyhow::anyhow!("strategy startup failed: {e}"))?;
        }

        // Seed the field from a snapshot, if one was given: teleport robots + ball
        // into place, then force the game state. Default to Run (free play) when
        // the snapshot carries none, so the rollout actually plays rather than
        // sitting through the kickoff sequence.
        if let Some(snapshot) = cfg.initial_snapshot.as_ref() {
            simulator.apply_snapshot(snapshot);
            let state = snapshot.game_state.unwrap_or(dies_core::GameState::Run);
            simulator.seed_game_state(state, snapshot.operating_team);
        }

        // Optionally record a full binary log of the match for offline analytics.
        let log_path = if let Some(base) = cfg.log_dir.clone() {
            let session = format!(
                "selfplay_{}_vs_{}_seed{}",
                cfg.blue_strategy, cfg.yellow_strategy, cfg.seed
            );
            let unix = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0);
            let meta = dies_logger::MetaJson::new(
                unix,
                true,
                Some(cfg.blue_strategy.clone()),
                Some(cfg.yellow_strategy.clone()),
                dies_logger::side_assignment_str(self.settings.team_configuration.side_assignment)
                    .to_string(),
            );
            dies_logger::worker::ArrowLogger::init(base.clone());
            dies_logger::worker::log_start(&session, meta);
            dies_logger::worker::log_set_field_geom(simulator.field_geometry());
            dies_logger::worker::log_settings_baseline(0, &self.settings);
            Some(base.join(&session).to_string_lossy().into_owned())
        } else {
            None
        };

        // Scripted in-match events, fired as the sim clock passes each timestamp.
        // Sorted ascending; a cursor walks through them so each fires once.
        let mut scripted_events = cfg.scripted_events.clone();
        scripted_events.sort_by(|a, b| a.t_secs.total_cmp(&b.t_secs));
        let mut next_event = 0usize;

        let dt = SIMULATION_DT;
        let cmd_period = CMD_INTERVAL.as_secs_f64();
        let mut next_cmd_t = 0.0_f64;
        let (mut prev_blue, mut prev_yellow) = simulator.score();
        let mut goals: Vec<headless::GoalEvent> = Vec::new();
        let mut trace_hash: u64 = 0;
        let mut end_reason;

        loop {
            // Drain control channel non-blocking. Headless has no UI, but honor
            // an explicit Stop and apply any queued sim/GC commands.
            let mut stop = false;
            while let Ok(msg) = self.command_rx.try_recv() {
                match msg {
                    ControlMsg::Stop => stop = true,
                    ControlMsg::GcCommand { command } => simulator.handle_gc_command(command),
                    ControlMsg::SimulatorCmd(cmd) => self.handle_simulator_cmd(&mut simulator, cmd),
                    other => self.handle_control_msg(other),
                }
            }
            if stop {
                end_reason = headless::EndReason::Stopped;
                break;
            }

            // Step physics + vision + strategy. The strategy IPC happens inside
            // (blocking), so this naturally paces to strategy compute time and
            // stays deterministic regardless of wall-clock.
            if let Err(e) = self.step_simulation(&mut simulator, dt) {
                end_reason = headless::EndReason::StrategyError(format!("{e:#}"));
                break;
            }

            // Abort loudly if a strategy died mid-match (its connection drops to
            // None on close), rather than silently running on with that team
            // frozen and reporting a bogus result.
            if let Some(host) = self.strategy_host.as_ref() {
                if !host.configured_strategies_alive() {
                    end_reason =
                        headless::EndReason::StrategyError("strategy disconnected".to_string());
                    break;
                }
            }

            // Fold the physical state into the trajectory fingerprint.
            trace_hash = trace_hash
                .rotate_left(7)
                .wrapping_mul(0x0000_0100_0000_01b3)
                ^ simulator.state_fingerprint();

            let now = simulator.time_secs();

            // Fire any scripted events whose time has arrived.
            while next_event < scripted_events.len() && scripted_events[next_event].t_secs <= now {
                match &scripted_events[next_event].kind {
                    headless::ScriptedEventKind::YellowCard { team } => {
                        log::info!("Scripted yellow card for {team:?} at {now:.1}s");
                        simulator.add_yellow_card(*team);
                    }
                }
                next_event += 1;
            }

            // Goal detection via score delta (authoritative, set by the sim's
            // auto-referee). Stamp each new goal with the current sim time.
            let (blue, yellow) = simulator.score();
            while prev_blue < blue {
                goals.push(headless::GoalEvent {
                    t_secs: now,
                    team: TeamColor::Blue,
                });
                prev_blue += 1;
            }
            while prev_yellow < yellow {
                goals.push(headless::GoalEvent {
                    t_secs: now,
                    team: TeamColor::Yellow,
                });
                prev_yellow += 1;
            }

            // 40 Hz command push via a sim-time accumulator — the deterministic
            // equivalent of run_rt_sim's independent wall-clock cmd interval.
            while next_cmd_t <= now {
                for (team_color, cmd) in self.player_commands() {
                    match cmd {
                        PlayerCmd::Move(cmd) => simulator.push_cmd(team_color, cmd),
                        PlayerCmd::GlobalMove(cmd) => simulator.push_global_cmd(team_color, cmd),
                    }
                }
                next_cmd_t += cmd_period;
            }

            // End conditions.
            if now >= cfg.duration_secs {
                end_reason = headless::EndReason::TimeLimit;
                break;
            }
            if let Some(max) = cfg.max_goals {
                if blue + yellow >= max {
                    end_reason = headless::EndReason::GoalCap;
                    break;
                }
            }
        }

        if let Some(h) = self.strategy_host.as_mut() {
            h.shutdown();
        }

        // Flush + compact the log before returning so the artifact is complete.
        if log_path.is_some() {
            dies_logger::worker::log_close_blocking(Duration::from_secs(30));
        }

        let (blue_score, yellow_score) = simulator.score();
        Ok(headless::MatchResult {
            blue_strategy: cfg.blue_strategy,
            yellow_strategy: cfg.yellow_strategy,
            seed: cfg.seed,
            blue_score,
            yellow_score,
            duration_secs: simulator.time_secs(),
            goals,
            end_reason,
            log_path,
            trace_hash,
        })
    }

    /// Run the executor in real time, processing vision and referee messages and sending
    /// commands to the robots.
    async fn run_rt_live(
        &mut self,
        mut ssl_client: VisionClient,
        mut bs_client: BasestationHandle,
    ) -> Result<()> {
        let mut cmd_interval = tokio::time::interval(CMD_INTERVAL);

        let vision_delay = self.settings.vision_delay_ms;
        if vision_delay > 0 {
            log::warn!("\n\n  ############################################################\n  ##                                                      ##\n  ##   !!!  SIMULATED VISION DELAY ACTIVE: {:>4} ms  !!!    ##\n  ##                                                      ##\n  ############################################################\n", vision_delay);
        }
        let (delayed_vision_tx, mut delayed_vision_rx) = tokio::sync::mpsc::unbounded_channel();

        let mut last_bs_msg = tokio::time::Instant::now();
        loop {
            tokio::select! {
                Some(vision_msg) = delayed_vision_rx.recv() => {
                    self.update_from_vision_msg(vision_msg, WorldInstant::now_real());
                }
                Some(msg) = self.command_rx.recv() => {
                    match msg {
                        ControlMsg::Stop => break,
                        ControlMsg::SetActiveTeams { .. } => {
                            log::warn!("Setting active teams is not supported mid run");
                        }
                        msg => self.handle_control_msg(msg)
                    }
                }
                ssl_msg = ssl_client.recv() => {
                    match ssl_msg {
                        Ok(SslMessage::Vision(vision_msg)) => {
                            if vision_delay > 0 {
                                let tx = delayed_vision_tx.clone();
                                tokio::spawn(async move {
                                    tokio::time::sleep(std::time::Duration::from_millis(vision_delay as u64)).await;
                                    let _ = tx.send(vision_msg);
                                });
                            } else {
                                self.update_from_vision_msg(vision_msg, WorldInstant::now_real());
                            }
                        }
                        Ok(SslMessage::Referee(gc_msg)) => {
                            self.update_from_gc_msg(gc_msg);
                        }
                        Ok(SslMessage::Tracker(tracker_msg)) => {
                            self.update_from_tracker_msg(tracker_msg);
                        }
                        Err(err) => {
                            if !self.settings.allow_no_vision {
                                log::error!("Failed to receive vision/gc msg: {}", err);
                            } else {
                                self.update_team_controller(true);
                            }
                        }
                    }
                }
                Some(tx) = self.info_channel_rx.recv() => {
                    let _  = tx.send(self.info());
                }
                bs_msg = bs_client.recv() => {
                    dies_core::debug_value("exec_bs msg_elapsed", last_bs_msg.elapsed().as_secs_f64() * 1000.0);
                    last_bs_msg = tokio::time::Instant::now();
                    match bs_msg {
                        Ok((team_color, bs_msg)) => {
                            let team_color = team_color.or(self.team_controllers.active_teams().first().copied());
                            if let Some(team_color) = team_color {
                                self.update_from_bs_msg(
                                    team_color,
                                    bs_msg,
                                    WorldInstant::now_real(),
                                );
                            }
                        }
                        Err(err) => {
                            log::error!("Failed to receive BS msg: {}", err);
                        }
                    }
                }
                _ = cmd_interval.tick() => {
                    for (team_color, cmd) in self.player_commands() {
                        bs_client.send_no_wait(team_color, cmd);
                    }
                    // Basestation link metrics (1 s window), so offline log
                    // analysis can separate round-robin TX scheduling from RF
                    // turnaround: with strict one-robot-per-loop round-robin,
                    // loop_hz ~= tx_hz ~= N_robots * per-robot feedback rate.
                    if let Some(info) = bs_client.base_info() {
                        if let Some(hz) = info.loop_hz {
                            dies_core::debug_value("base.loop_hz", hz as f64);
                        }
                        if let Some(hz) = info.tx_hz {
                            dies_core::debug_value("base.tx_hz", hz as f64);
                        }
                        if let Some(hz) = info.rx_hz {
                            dies_core::debug_value("base.rx_hz", hz as f64);
                        }
                        if let Some(us) = info.work_max_us {
                            dies_core::debug_value("base.work_max_us", us as f64);
                        }
                        if let Some(frac) = info.overrun_frac {
                            dies_core::debug_value("base.overrun_frac", frac as f64);
                        }
                    }
                }
            }
        }
        if let Some(h) = &mut self.strategy_host {
            h.shutdown();
        }
        Ok(())
    }

    fn handle_simulator_cmd(&self, sim: &mut Simulation, cmd: SimulatorCmd) {
        match cmd {
            SimulatorCmd::ApplyBallForce { force } => {
                sim.apply_force_to_ball(Vector3::new(force.x, force.y, 0.0));
            }
            SimulatorCmd::TeleportBall { position } => {
                sim.teleport_ball(position);
            }
            SimulatorCmd::TeleportRobot {
                team_color,
                player_id,
                position,
                yaw,
            } => {
                sim.teleport_robot(team_color, player_id, position, yaw);
            }
            SimulatorCmd::AddRobot {
                team_color,
                player_id,
                position,
                yaw,
            } => {
                sim.add_robot(team_color, player_id, position, yaw);
            }
            SimulatorCmd::RemoveRobot {
                team_color,
                player_id,
            } => {
                sim.remove_robot(team_color, player_id);
            }
        }
    }

    /// Apply a Sim Edit manipulation while the executor is paused.
    ///
    /// The real per-frame pipeline (tracker → controllers → log) is the wrong
    /// tool here: it would advance the clock, trip the player-gone detector,
    /// recompute strategy, and log empty frames — all while "nothing is
    /// happening". Instead we patch a frozen snapshot of the world (lazily taken
    /// from the tracker at the first edit) and broadcast it directly, so the UI
    /// shows the moved entity and nothing else changes. The underlying rigid
    /// bodies *are* moved in the sim, so on resume the next real frame reconciles
    /// the tracker via its teleport-detection.
    fn apply_sim_edit_paused(&mut self, sim: &mut Simulation, cmd: SimulatorCmd) {
        self.handle_simulator_cmd(sim, cmd.clone());

        // A queued kick impulse only becomes visible once physics steps again,
        // so there's nothing to redraw while paused.
        if matches!(cmd, SimulatorCmd::ApplyBallForce { .. }) {
            return;
        }

        if self.paused_edit_world.is_none() {
            self.paused_edit_world = Some(self.tracker.get());
        }
        let world = self.paused_edit_world.as_mut().unwrap();
        match cmd {
            SimulatorCmd::TeleportRobot {
                team_color,
                player_id,
                position,
                yaw,
            }
            | SimulatorCmd::AddRobot {
                team_color,
                player_id,
                position,
                yaw,
            } => patch_paused_robot(world, team_color, player_id, position, yaw),
            SimulatorCmd::RemoveRobot {
                team_color,
                player_id,
            } => {
                let team = match team_color {
                    TeamColor::Blue => &mut world.blue_team,
                    TeamColor::Yellow => &mut world.yellow_team,
                };
                team.retain(|p| p.id != player_id);
            }
            SimulatorCmd::TeleportBall { position } => match world.ball.as_mut() {
                Some(ball) => {
                    ball.position = Vector3::new(position.x, position.y, ball.position.z);
                    ball.raw_position = vec![ball.position];
                    ball.velocity = Vector3::zeros();
                    ball.detected = true;
                }
                None => {
                    let pos = Vector3::new(position.x, position.y, 0.0);
                    world.ball = Some(BallData {
                        timestamp: world.t_received,
                        position: pos,
                        raw_position: vec![pos],
                        velocity: Vector3::zeros(),
                        detected: true,
                    });
                }
            },
            SimulatorCmd::ApplyBallForce { .. } => unreachable!("handled above"),
        }

        let update = WorldUpdate {
            world_data: world.clone(),
            frame_id: self.frame_counter,
            announcements: self.announcer.snapshot(),
        };
        let _ = self.update_tx.send(update);
    }

    pub fn handle(&self) -> ExecutorHandle {
        ExecutorHandle {
            control_tx: self.command_tx.clone(),
            update_rx: self.update_tx.subscribe(),
            notebook_rx: self.notebook_tx.subscribe(),
            info_channel: self.info_channel_tx.clone(),
        }
    }

    /// Build the notebook's per-frame snapshot (absolute field coordinates).
    fn notebook_snapshot(&self, world: &WorldData) -> String {
        let side = self.team_controllers.side_assignment;
        let mut robots = Vec::new();
        for (color, ctrl, roster) in [
            (
                TeamColor::Blue,
                &self.team_controllers.blue_team,
                &world.blue_team,
            ),
            (
                TeamColor::Yellow,
                &self.team_controllers.yellow_team,
                &world.yellow_team,
            ),
        ] {
            let Some(ctrl) = ctrl else { continue };
            let statuses = ctrl.get_skill_statuses();
            let passes = ctrl.get_pass_results();
            let target_vels = ctrl.target_velocities_global();
            for p in roster {
                robots.push(notebook::RobotInfo {
                    team: color,
                    id: p.id,
                    skill: statuses.get(&p.id).copied(),
                    target_vel: target_vels
                        .get(&p.id)
                        .map(|v| side.transform_vec2(color, v)),
                    pass: passes.get(&p.id).copied(),
                    feedback: self.latest_feedback.get(&(color, p.id)),
                });
            }
        }
        let snapshot = notebook::Snapshot {
            frame: self.frame_counter,
            t: world.t_received,
            dt: world.dt,
            teams: self.team_controllers.active_teams(),
            world,
            robots,
        };
        notebook::to_json(&snapshot).unwrap_or_else(|e| {
            log::error!("{e}");
            "{}".to_string()
        })
    }

    /// Log a discrete event to the columnar log (no-op on the legacy path).
    fn log_event(&self, event_type: &str, payload: serde_json::Value) {
        if worker::is_active() {
            worker::log_event(
                self.frame_counter,
                self.last_t,
                event_type,
                payload.to_string(),
            );
        }
    }

    fn handle_control_msg(&mut self, msg: ControlMsg) {
        match msg {
            ControlMsg::SetPause(pause) => {
                self.paused_tx.send(pause).ok();
                self.log_event("pause", serde_json::json!({ "paused": pause }));
            }
            ControlMsg::SetPlayerOverride {
                team_color,
                player_id,
                override_active,
            } => {
                if override_active {
                    self.manual_override
                        .insert((team_color, player_id), PlayerOverrideState::new());
                } else {
                    self.manual_override.remove(&(team_color, player_id));
                }
                self.log_event(
                    "player_override",
                    serde_json::json!({
                        "team": format!("{team_color:?}"),
                        "player_id": player_id.as_u32(),
                        "active": override_active,
                    }),
                );
            }
            ControlMsg::PlayerOverrideCommand {
                team_color,
                player_id,
                command,
            } => {
                if let Some(state) = self.manual_override.get_mut(&(team_color, player_id)) {
                    state.set_cmd(command.clone());
                }
                self.log_event(
                    "player_override_command",
                    serde_json::json!({
                        "team": format!("{team_color:?}"),
                        "player_id": player_id.as_u32(),
                        "command": serde_json::to_value(&command).unwrap_or_default(),
                    }),
                );
            }
            ControlMsg::UpdateSettings(settings) => {
                self.team_controllers.update_settings(&settings);
                self.tracker.update_settings(&settings);
                crate::skills::apply_skill_tunables(&settings.skill_tunables);
                self.settings = settings;
                if worker::is_active() {
                    worker::log_settings_diff(self.frame_counter, self.last_t, &self.settings);
                }
            }
            ControlMsg::AddMarker { label } => {
                worker::log_marker(self.frame_counter, self.last_t, label);
            }
            ControlMsg::SetStrategyParams { team, params } => {
                if let Some(h) = &mut self.strategy_host {
                    h.set_strategy_params(team, params);
                }
            }
            ControlMsg::SetSideAssignment(side_assignment) => {
                self.team_controllers.side_assignment = side_assignment;
                self.tracker.set_side_assignment(side_assignment);
                if let Some(h) = &mut self.strategy_host {
                    h.set_side_assignment(side_assignment);
                }
                self.log_event(
                    "side_assignment",
                    serde_json::json!({ "side_assignment": format!("{side_assignment:?}") }),
                );
            }
            ControlMsg::SetTeamConfiguration(_) => {
                log::warn!("Setting team configuration is not supported mid run");
            }
            ControlMsg::SwapTeamColors => {
                self.team_controllers.swap_teams();
                self.log_event("swap_team_colors", serde_json::Value::Null);
            }
            ControlMsg::SwapTeamSides => {
                self.team_controllers.side_assignment = match self.team_controllers.side_assignment
                {
                    SideAssignment::BlueOnPositive => SideAssignment::YellowOnPositive,
                    SideAssignment::YellowOnPositive => SideAssignment::BlueOnPositive,
                };
                self.tracker
                    .set_side_assignment(self.team_controllers.side_assignment);
                if let Some(h) = &mut self.strategy_host {
                    h.set_side_assignment(self.team_controllers.side_assignment);
                }
                self.log_event(
                    "swap_team_sides",
                    serde_json::json!({
                        "side_assignment": format!("{:?}", self.team_controllers.side_assignment),
                    }),
                );
            }
            ControlMsg::Notebook(cmd) => {
                let snapshot = if self.notebook.has_engine() {
                    String::new()
                } else {
                    let world = self.tracker.get();
                    self.notebook_snapshot(&world)
                };
                let (frame_id, t) = (self.frame_counter, self.last_t);
                self.notebook.handle_cmd(cmd, frame_id, t, || snapshot);
            }
            ControlMsg::Stop => {}

            // Live mode: the in-app "simulated GC" drives a manual game state
            // that overrides the real GC for teams with `ignore_gc` on. (In
            // sim/headless the loops intercept `GcCommand` before this and route
            // it to the simulator's auto-ref, so this arm is live-only.)
            ControlMsg::GcCommand { command } => self.manual_gc.apply(command),

            ControlMsg::SimulatorCmd(_) | ControlMsg::SetActiveTeams { .. } => {
                unreachable!();
            }
        }
    }

    /// The effective per-team GC override, combining the team's settings with the
    /// current manually-driven game state. `override_halt` wins; otherwise
    /// `ignore_gc` makes the team follow the manual GC (`manual_gc`, defaulting to
    /// free play); otherwise the real GC governs.
    fn team_gc_override(&self, settings: &TeamSpecificSettings) -> GcOverride {
        if settings.override_halt {
            GcOverride::Halt
        } else if settings.ignore_gc {
            GcOverride::IgnoreGc {
                game_state: self.manual_gc.game_state,
                operating_team: self.manual_gc.operating_team,
            }
        } else {
            GcOverride::None
        }
    }

    /// Run the controllers for the current world state and broadcast it. When
    /// `new_frame` is set (a vision-driven or no-vision tick), the frame is
    /// recorded to the columnar log and the frame counter advances; referee
    /// updates pass `false` and reuse the current frame id.
    fn update_team_controller(&mut self, new_frame: bool) {
        let mut world_data = self.tracker.get();
        let frame_id = self.frame_counter;
        self.last_t = world_data.t_received;

        // Enrich each controlled team's players with the skill they're executing,
        // taken from last frame's controller state (the controllers run later in
        // this function). One frame of lag is irrelevant for display.
        if let Some(ctrl) = self.team_controllers.blue_team.as_ref() {
            let infos = ctrl.get_skill_infos();
            for p in world_data.blue_team.iter_mut() {
                p.skill = infos.get(&p.id).cloned();
            }
        }
        if let Some(ctrl) = self.team_controllers.yellow_team.as_ref() {
            let infos = ctrl.get_skill_infos();
            for p in world_data.yellow_team.iter_mut() {
                p.skill = infos.get(&p.id).cloned();
            }
        }

        // Yellow-card removal pre-pass (before the strategy): decide which robots
        // to bench and stamp them `CardRemoved` on the color lists, so the
        // team-relative transform forks them out of `own_players`. Radio-lost
        // marks are already present from the tracker. Runs before the broadcast so
        // the UI sees both sideline reasons.
        fn stamp_card_removed(
            ctrl: Option<&mut TeamController>,
            roster: &mut [PlayerData],
            max_bots: u32,
            gc_keeper: Option<PlayerId>,
        ) {
            let Some(ctrl) = ctrl else { return };
            // Same GC-first / fallback resolution as the TeamData transform, so
            // the robot exempted here is the one actually playing keeper.
            let keeper_id = dies_core::effective_keeper_id(
                gc_keeper,
                roster.iter().filter(|p| p.sideline.is_none()).map(|p| p.id),
            );
            let removed = ctrl.plan_sidelining(roster, max_bots, keeper_id);
            for p in roster.iter_mut() {
                if p.sideline.is_none() && removed.contains(&p.id) {
                    p.sideline = Some(SidelineReason::CardRemoved);
                }
            }
        }
        stamp_card_removed(
            self.team_controllers.blue_team.as_mut(),
            &mut world_data.blue_team,
            world_data.game_state.blue_team_max_allowed_bots,
            world_data.game_state.blue_team_keeper_id,
        );
        stamp_card_removed(
            self.team_controllers.yellow_team.as_mut(),
            &mut world_data.yellow_team,
            world_data.game_state.yellow_team_max_allowed_bots,
            world_data.game_state.yellow_team_keeper_id,
        );

        // Stamp the latest open-loop delay estimate onto each own robot for the
        // UI. Uses detector state accumulated on prior frames (the detector is
        // fed further below, after the controllers produce this tick's
        // commands), so it carries a harmless one-frame lag.
        {
            let now = world_data.t_received;
            for (color, roster) in [
                (TeamColor::Blue, &mut world_data.blue_team),
                (TeamColor::Yellow, &mut world_data.yellow_team),
            ] {
                for pd in roster.iter_mut() {
                    if let Some(det) = self.open_loop_detectors.get(&(color, pd.id)) {
                        pd.open_loop_delay = det.stats(now);
                        if let Some(s) = pd.open_loop_delay {
                            let id = pd.id.as_u32();
                            dies_core::debug_value(
                                format!("team_{color:?}.p{id}.ol_delay_median_ms"),
                                s.median_ms,
                            );
                            dies_core::debug_value(
                                format!("team_{color:?}.p{id}.ol_delay_max_ms"),
                                s.max_ms,
                            );
                            dies_core::debug_value(
                                format!("team_{color:?}.p{id}.ol_delay_age_s"),
                                s.age_s,
                            );
                        }
                    }
                }
            }
        }

        let update = WorldUpdate {
            world_data: world_data.clone(),
            frame_id,
            // Full rolling window every frame: the watch channel to the UI
            // coalesces, so deltas would be dropped. The UI dedupes by id.
            announcements: self.announcer.snapshot(),
        };
        if let Err(err) = self.update_tx.send(update) {
            log::error!("Failed to broadcast world update: {}", err);
        }

        let blue_gc_override = self.team_gc_override(&self.settings.blue_team_settings);
        let yellow_gc_override = self.team_gc_override(&self.settings.yellow_team_settings);
        let blue_team_data = if self.team_controllers.blue_team.is_some() {
            Some(world_data.get_team_data(TeamColor::Blue, blue_gc_override))
        } else {
            None
        };
        let yellow_team_data = if self.team_controllers.yellow_team.is_some() {
            Some(world_data.get_team_data(TeamColor::Yellow, yellow_gc_override))
        } else {
            None
        };

        if let Some(strategy_host) = self.strategy_host.as_mut() {
            if let Some(ref controller) = self.team_controllers.blue_team {
                let statuses = controller.get_skill_statuses();
                strategy_host.update_skill_statuses(TeamColor::Blue, statuses);
                strategy_host.update_pass_results(TeamColor::Blue, controller.get_pass_results());
            }
            if let Some(ref controller) = self.team_controllers.yellow_team {
                let statuses = controller.get_skill_statuses();
                strategy_host.update_skill_statuses(TeamColor::Yellow, statuses);
                strategy_host.update_pass_results(TeamColor::Yellow, controller.get_pass_results());
            }

            let frame_output =
                strategy_host.update(blue_team_data.as_ref(), yellow_team_data.as_ref());

            if let Some(ref mut controller) = self.team_controllers.blue_team {
                controller.set_strategy_input(StrategyInput {
                    skill_commands: frame_output.blue_commands,
                    player_roles: frame_output.blue_roles,
                });
            }
            if let Some(ref mut controller) = self.team_controllers.yellow_team {
                controller.set_strategy_input(StrategyInput {
                    skill_commands: frame_output.yellow_commands,
                    player_roles: frame_output.yellow_roles,
                });
            }
        }

        // Notebook: a second command source next to the strategy host. Its
        // overlay is (re)set every frame so a stale one never lingers.
        let mut notebook_inputs = if self.notebook.is_active() {
            let snapshot = self.notebook_snapshot(&world_data);
            self.notebook.frame(
                &snapshot,
                self.team_controllers.side_assignment,
                frame_id,
                world_data.t_received,
            )
        } else {
            HashMap::new()
        };
        if let Some(ref mut controller) = self.team_controllers.blue_team {
            controller
                .set_notebook_input(notebook_inputs.remove(&TeamColor::Blue).unwrap_or_default());
        }
        if let Some(ref mut controller) = self.team_controllers.yellow_team {
            controller.set_notebook_input(
                notebook_inputs
                    .remove(&TeamColor::Yellow)
                    .unwrap_or_default(),
            );
        }

        // Webui overrides arrive in absolute world coordinates (matching the absolute
        // world data the UI renders), so transform them into team-relative coordinates
        // here.
        let side_assignment = self.team_controllers.side_assignment;
        let manual_override: HashMap<(TeamColor, PlayerId), PlayerControlInput> = self
            .manual_override
            .iter_mut()
            .map(|((color, id), s)| {
                (
                    (*color, *id),
                    s.advance().to_team_frame(side_assignment, *color),
                )
            })
            .collect();
        self.team_controllers.update(
            &world_data,
            manual_override,
            blue_gc_override,
            yellow_gc_override,
        );

        // Feed this tick's velocity setpoints to the tracker for command
        // feedforward (consumed on the next vision update). The controllers run in
        // team-relative coordinates, but the tracker's filter runs in vision/
        // absolute coordinates — untransform each setpoint back to absolute (the
        // x-flip is its own inverse). A no-op unless `player_use_command_feedforward`.
        let side_assignment = self.team_controllers.side_assignment;
        for (color, id, vel) in self.team_controllers.target_velocities_global() {
            let vel_abs = side_assignment.transform_vec2(color, &vel);
            self.tracker.set_player_command(color, id, vel_abs);

            // Feed the online open-loop delay estimator: raw (unfiltered) vision
            // position vs. the command sent this frame, both in absolute coords.
            // One tick per real vision frame; stats are read back next frame.
            if new_frame {
                let roster = match color {
                    TeamColor::Blue => &world_data.blue_team,
                    TeamColor::Yellow => &world_data.yellow_team,
                };
                if let Some(pd) = roster.iter().find(|p| p.id == id) {
                    self.open_loop_detectors
                        .entry((color, id))
                        .or_default()
                        .push(world_data.t_received, pd.raw_position, vel_abs);
                }
            }
        }

        // Efference copy: tell the possession metric who is kicking this tick so a
        // kicker isn't immediately re-credited with the ball it just released.
        for (color, id) in self.team_controllers.kicking_players() {
            self.tracker.notify_kick(color, id);
        }

        // Record the frame after the controllers (and strategy) have produced
        // this tick's debug output. The debug map is snapshotted here so it is
        // stamped with the same frame id as the world state.
        if new_frame {
            // Age out stale debug keys before snapshotting, so logs and the
            // live UI only carry keys refreshed within the TTL window.
            dies_core::debug_tick();
            if worker::is_active() {
                let debug = DebugSubscriber::instance().get_copy();
                worker::log_frame(frame_id, &world_data, &debug);
                let feedback: Vec<(TeamColor, PlayerFeedbackMsg)> = self
                    .latest_feedback
                    .iter()
                    .map(|(&(team, _), &msg)| (team, msg))
                    .collect();
                worker::log_player_feedback(frame_id, &feedback);
                self.log_match_metadata(&world_data, frame_id);
            }
            self.frame_counter = self.frame_counter.wrapping_add(1);
        }
    }

    /// Bake match identity into the log: patch the GC team names into `meta.json`
    /// when they change, and — on real-match runs (`is_match`) — drop a one-shot
    /// "MATCH START — <blue> vs <yellow>" timeline marker once the names arrive.
    fn log_match_metadata(&mut self, world_data: &WorldData, frame_id: u64) {
        let blue = world_data.game_state.blue_team_name.clone();
        let yellow = world_data.game_state.yellow_team_name.clone();

        // Nothing to record until the GC operator has entered at least one name.
        if blue.is_none() && yellow.is_none() {
            return;
        }

        let names = (blue.clone(), yellow.clone());
        if self.last_logged_team_names.as_ref() != Some(&names) {
            worker::log_set_team_names(blue.clone(), yellow.clone());
            self.last_logged_team_names = Some(names);
        }

        if self.settings.is_match && !self.match_marker_dropped {
            let label = format!(
                "MATCH START — {} vs {}",
                blue.as_deref().unwrap_or("Blue"),
                yellow.as_deref().unwrap_or("Yellow"),
            );
            worker::log_marker(frame_id, self.last_t, Some(label));
            self.match_marker_dropped = true;
        }
    }
}

/// Patch a single robot's pose in a frozen Sim Edit world snapshot, inserting a
/// fresh `PlayerData` if the robot isn't present yet (Add / load-into-empty).
fn patch_paused_robot(
    world: &mut WorldData,
    team: TeamColor,
    id: PlayerId,
    position: Vector2,
    yaw: Angle,
) {
    let list = match team {
        TeamColor::Blue => &mut world.blue_team,
        TeamColor::Yellow => &mut world.yellow_team,
    };
    let player = match list.iter_mut().position(|p| p.id == id) {
        Some(idx) => &mut list[idx],
        None => {
            list.push(PlayerData::new(id));
            list.last_mut().unwrap()
        }
    };
    player.position = position;
    player.raw_position = position;
    player.yaw = yaw;
    player.raw_yaw = yaw;
    player.velocity = Vector2::zeros();
    player.angular_speed = 0.0;
}

#[cfg(test)]
mod manual_gc_tests {
    use super::*;

    #[test]
    fn kickoff_is_two_step() {
        let mut gc = ManualGc::default();
        assert_eq!(gc.game_state, GameState::Run);
        assert_eq!(gc.operating_team, None);

        gc.apply(GcSimCommand::KickOff {
            team_color: TeamColor::Blue,
        });
        assert_eq!(gc.game_state, GameState::PrepareKickoff);
        assert_eq!(gc.operating_team, Some(TeamColor::Blue));

        // Continue (NormalStart) from a prepare phase activates the kickoff,
        // keeping the operating team.
        gc.apply(GcSimCommand::NormalStart);
        assert_eq!(gc.game_state, GameState::Kickoff);
        assert_eq!(gc.operating_team, Some(TeamColor::Blue));

        // Continue again (ForceStart) drops to free play.
        gc.apply(GcSimCommand::ForceStart);
        assert_eq!(gc.game_state, GameState::Run);
        assert_eq!(gc.operating_team, None);
    }

    #[test]
    fn free_kick_and_penalty_are_one_shot() {
        let mut gc = ManualGc::default();
        gc.apply(GcSimCommand::DirectFree {
            team_color: TeamColor::Yellow,
        });
        assert_eq!(gc.game_state, GameState::FreeKick);
        assert_eq!(gc.operating_team, Some(TeamColor::Yellow));

        gc.apply(GcSimCommand::Penalty {
            team_color: TeamColor::Blue,
        });
        assert_eq!(gc.game_state, GameState::Penalty);
        assert_eq!(gc.operating_team, Some(TeamColor::Blue));
    }

    #[test]
    fn stop_and_halt() {
        let mut gc = ManualGc::default();
        gc.apply(GcSimCommand::Stop);
        assert_eq!(gc.game_state, GameState::Stop);
        assert_eq!(gc.operating_team, None);
        gc.apply(GcSimCommand::Halt);
        assert_eq!(gc.game_state, GameState::Halt);
    }

    #[test]
    fn normal_start_outside_prepare_runs() {
        let mut gc = ManualGc::default();
        gc.apply(GcSimCommand::Stop);
        gc.apply(GcSimCommand::NormalStart);
        assert_eq!(gc.game_state, GameState::Run);
        assert_eq!(gc.operating_team, None);
    }

    #[test]
    fn ignored_commands_are_noops() {
        let mut gc = ManualGc::default();
        gc.apply(GcSimCommand::KickOff {
            team_color: TeamColor::Blue,
        });
        let before = (gc.game_state, gc.operating_team);
        gc.apply(GcSimCommand::YellowCard {
            team_color: TeamColor::Yellow,
        });
        gc.apply(GcSimCommand::BallPlacement {
            team_color: TeamColor::Yellow,
            position: Vector2::zeros(),
        });
        assert_eq!((gc.game_state, gc.operating_team), before);
    }
}
