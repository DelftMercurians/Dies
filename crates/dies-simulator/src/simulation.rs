//! Public simulator facade and ordered update pipeline.
use crate::{
    ball::{BallContext, BallInteraction},
    packets::SimOutput,
    physics::PhysicsWorld,
    referee::{AutoReferee, RefereeAction, RefereeContext, SimRefereeEvent},
    robot::{RobotFleet, RobotStepContext},
    SimulationConfig,
};
use dies_core::{
    Angle, FieldGeometry, FieldSnapshot, GameState, GcSimCommand, PlayerFeedbackMsg,
    PlayerGlobalMoveCmd, PlayerId, PlayerMoveCmd, TeamColor, Vector2, WorldInstant,
};
use dies_protos::{ssl_gc_referee_message::Referee, ssl_vision_wrapper::SSL_WrapperPacket};
use rand::{rngs::StdRng, SeedableRng};
use rapier3d_f64::prelude::*;
use serde::Serialize;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize)]
pub struct SimulationState {
    players: Vec<SimulationPlayerState>,
    ball: Option<Vector<f64>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SimulationPlayerState {
    position: Vector2,
    yaw: Angle,
}

/// A complete simulator for testing strategies and robot control in silico.
///
/// ## Usage
///
/// ```ignore
/// let mut simulator = Simulation::default();
///
/// let dt = std::time::Duration::from_millis(10);
/// loop {
///     simulator.step(dt);
///
///     // Get the latest detection packet
///     if let Some(detection) = simulator.detection() {
///        // Do something with the detection packet
///     }
///
///     // Get a geometry packet
///     if let Some(geometry) = simulator.geometry() {
///       // Do something with the geometry packet
///     }
///
///     // Push commands for players
///     simulator.push_cmd(PlayerCmd::zero(0))
/// }
/// ```
pub struct Simulation {
    pub(crate) config: SimulationConfig,
    current_time: f64,
    pub(crate) rng: StdRng,
    pub(crate) physics: PhysicsWorld,
    pub(crate) robots: RobotFleet,
    ball_interaction: BallInteraction,
    referee: AutoReferee,
    output: SimOutput,
}

impl Simulation {
    /// Create an empty simulation, ready for players and a ball.
    pub fn new(config: SimulationConfig) -> Simulation {
        let rng = StdRng::seed_from_u64(config.seed);
        let physics = PhysicsWorld::new(&config);
        let robots = RobotFleet::new();
        let ball_interaction = BallInteraction::new();
        let referee = AutoReferee::new(config.initial_side_assignment);
        let output = SimOutput::new(&config);
        Simulation {
            config,
            current_time: 0.0,
            rng,
            physics,
            robots,
            ball_interaction,
            referee,
            output,
        }
    }

    pub fn time(&self) -> WorldInstant {
        WorldInstant::Simulated(self.current_time)
    }

    /// Current simulated time in seconds.
    pub fn time_secs(&self) -> f64 {
        self.current_time
    }

    /// Current match score as `(blue, yellow)`.
    pub fn score(&self) -> (u32, u32) {
        (self.referee.blue_score, self.referee.yellow_score)
    }

    /// The field geometry this simulation was built with (logged into the log's
    /// `meta.json` so analytics know the goal/field coordinates).
    pub fn field_geometry(&self) -> &FieldGeometry {
        &self.config.field_geometry
    }

    /// A deterministic 64-bit fingerprint of the current physical state: the
    /// ball position plus every player's pose, in player-vector order. Folding
    /// this each frame yields a trajectory hash that is byte-identical across
    /// runs with the same `(seed, strategies)` on the same binary, and diverges
    /// as soon as the trajectories differ — used to verify determinism and to
    /// confirm that the seed actually varies the match. Uses raw float bits
    /// (FNV-1a) for exact same-binary equality.
    pub fn state_fingerprint(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis
        let mut mix = |x: f64, h: &mut u64| {
            for b in x.to_bits().to_le_bytes() {
                *h ^= b as u64;
                *h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        if let Some(ball) = &self.physics.ball {
            if let Some(rb) = self.physics.rigid_body_set.get(ball._rigid_body_handle) {
                let p = rb.translation();
                mix(p.x, &mut h);
                mix(p.y, &mut h);
                mix(p.z, &mut h);
            }
        }
        for player in &self.robots.players {
            if let Some(rb) = self.physics.rigid_body_set.get(player.rigid_body_handle) {
                let t = rb.translation();
                let yaw = rb.rotation().euler_angles().2;
                mix(t.x, &mut h);
                mix(t.y, &mut h);
                mix(yaw, &mut h);
            }
        }
        h
    }

    /// Get the controlled teams
    pub fn controlled_teams(&self) -> HashSet<TeamColor> {
        let mut teams = HashSet::new();
        if self.config.blue_controlled {
            teams.insert(TeamColor::Blue);
        }
        if self.config.yellow_controlled {
            teams.insert(TeamColor::Yellow);
        }
        teams
    }

    /// Set which teams are controlled by the simulation
    pub fn set_controlled_teams(&mut self, teams: &[TeamColor]) {
        self.config.blue_controlled = teams.contains(&TeamColor::Blue);
        self.config.yellow_controlled = teams.contains(&TeamColor::Yellow);
    }

    /// Check if a team is controlled
    pub fn is_team_controlled(&self, team_color: TeamColor) -> bool {
        match team_color {
            TeamColor::Blue => self.config.blue_controlled,
            TeamColor::Yellow => self.config.yellow_controlled,
        }
    }

    /// Apply a saved field snapshot: teleport existing robots to the snapshot's
    /// poses, add ones it has that the field doesn't, remove ones it omits, and
    /// place the ball. Mirrors the Web UI snapshot-load flow but runs directly
    /// against the simulator (used to seed a headless rollout). Game state is
    /// seeded separately via [`Simulation::seed_game_state`].
    pub fn apply_snapshot(&mut self, snapshot: &FieldSnapshot) {
        for (color, robots) in [
            (TeamColor::Blue, &snapshot.blue),
            (TeamColor::Yellow, &snapshot.yellow),
        ] {
            let snap_ids: HashSet<PlayerId> = robots.iter().map(|r| r.id).collect();
            for r in robots {
                let exists = self
                    .robots
                    .players
                    .iter()
                    .any(|p| p.id == r.id && p.team_color == color);
                if exists {
                    self.teleport_robot(color, r.id, r.position, r.yaw);
                } else {
                    self.add_robot(color, r.id, r.position, r.yaw);
                }
            }
            // Remove robots not present in the snapshot so the setup reproduces exactly.
            let to_remove: Vec<PlayerId> = self
                .robots
                .players
                .iter()
                .filter(|p| p.team_color == color && !snap_ids.contains(&p.id))
                .map(|p| p.id)
                .collect();
            for id in to_remove {
                self.remove_robot(color, id);
            }
        }
        if let Some(ball) = snapshot.ball {
            self.teleport_ball(ball);
        }
    }

    pub fn handle_gc_command(&mut self, command: GcSimCommand) {
        self.with_referee(|referee, ctx| referee.handle_gc_command(ctx, command))
    }

    /// Seed the auto-referee game state directly, e.g. when starting a headless
    /// rollout from a snapshot. Maps the strategy-facing [`GameState`] onto the
    /// equivalent GC command; `Run`/`Kickoff`/`PenaltyRun` drop the match straight
    /// into free play via `ForceStart` (skipping the kickoff cycle). `operating_team`
    /// selects the team for asymmetric states (kickoff/free-kick/penalty/placement)
    /// and is ignored for symmetric ones; it defaults to Blue when unknown.
    pub fn seed_game_state(&mut self, state: GameState, operating_team: Option<TeamColor>) {
        self.with_referee(|referee, ctx| referee.seed_game_state(ctx, state, operating_team))
    }

    /// Drain the simulator-internal referee events accumulated since the last call.
    pub fn take_referee_events(&mut self) -> Vec<SimRefereeEvent> {
        self.referee.take_referee_events()
    }

    /// Show a yellow card to `team` (SSL §5.5.1): record a 120 s expiry, lowering
    /// the team's `max_allowed_bots` by one until it expires, and immediately
    /// re-broadcast the current referee command so the new allowance reaches the
    /// world tracker without waiting for the next game-state transition. The
    /// executor's compliance layer then parks the excess robot off-field; on
    /// expiry the allowance is restored (updated during each simulation step).
    pub fn add_yellow_card(&mut self, team: TeamColor) {
        self.with_referee(|referee, ctx| referee.add_yellow_card(ctx, team))
    }

    pub fn gc_message(&mut self) -> Option<Referee> {
        self.referee.gc_message()
    }

    pub fn update_game_state(&mut self) {
        self.with_referee(|referee, ctx| referee.update_game_state(ctx))
    }

    pub fn apply_force_to_ball(&mut self, force: Vector<f64>) {
        self.physics.apply_force_to_ball(force)
    }

    /// Teleport the ball to a field position (z held at rest height) and zero
    /// its velocity. Used by scenarios to reset to a clean state between drills.
    pub fn teleport_ball(&mut self, position: Vector2) {
        self.physics.teleport_ball(position)
    }

    pub fn teleport_robot(
        &mut self,
        team_color: TeamColor,
        player_id: PlayerId,
        position: Vector2,
        yaw: Angle,
    ) {
        self.robots
            .teleport_robot(&mut self.physics, team_color, player_id, position, yaw)
    }

    pub fn add_robot(
        &mut self,
        team_color: TeamColor,
        player_id: PlayerId,
        position: Vector2,
        yaw: Angle,
    ) {
        self.robots.add_robot(
            &mut self.physics,
            &self.config,
            team_color,
            player_id,
            position,
            yaw,
        )
    }

    pub fn remove_robot(&mut self, team_color: TeamColor, player_id: PlayerId) {
        self.robots
            .remove_robot(&mut self.physics, team_color, player_id)
    }

    /// Pushes a PlayerCmd onto the execution queue with the time delay specified in
    /// the config
    pub fn push_cmd(&mut self, team_color: TeamColor, cmd: PlayerMoveCmd) {
        self.robots.push_cmd(
            self.current_time,
            self.config.command_delay,
            team_color,
            cmd,
        )
    }

    pub fn push_global_cmd(&mut self, team_color: TeamColor, cmd: PlayerGlobalMoveCmd) {
        self.robots.push_global_cmd(
            self.current_time,
            self.config.command_delay,
            team_color,
            cmd,
        )
    }

    pub fn detection(&mut self) -> Option<SSL_WrapperPacket> {
        self.output.detection()
    }

    pub fn geometry(&mut self) -> Option<SSL_WrapperPacket> {
        self.output.geometry(self.current_time)
    }

    pub fn feedback(&mut self) -> HashMap<(TeamColor, PlayerId), PlayerFeedbackMsg> {
        self.output.feedback()
    }

    pub fn step(&mut self, dt: f64) {
        // Preserve original phase order, packet timing, and previous-step dt reads.
        self.with_referee(|referee, ctx| referee.observe_touches(ctx));
        self.with_referee(|referee, ctx| referee.update_cards(ctx));
        self.with_referee(|referee, ctx| referee.update_card_substitutions(ctx));
        self.with_referee(|referee, ctx| referee.check_robot_allowances(ctx));
        self.update_game_state();
        if self.output.detection_interval.trigger(self.current_time) {
            self.output
                .new_detection_packet(&self.physics, &self.robots, self.current_time);
        }
        let commands = self.robots.take_due_commands(self.current_time);
        self.physics.update_queries();
        crate::ball::apply_rolling_friction(&mut self.physics, &self.config);
        let frame = self.robots.update(
            &mut RobotStepContext {
                config: &self.config,
                now: self.current_time,
                physics: &mut self.physics,
                referee: &mut self.referee,
                output: &mut self.output,
                rng: &mut self.rng,
            },
            &commands,
            dt,
        );
        self.ball_interaction.resolve(
            &mut BallContext {
                config: &self.config,
                physics: &mut self.physics,
                robots: &self.robots,
                rng: &mut self.rng,
            },
            dt,
            &frame.dribble_claimants,
            frame.ball_kicked_this_frame,
        );
        self.physics.step(&self.config, dt);
        self.physics.clamp_ball(&self.config);
        self.current_time += dt;
    }

    #[cfg(test)]
    /// Place the ball at rest at the given (x, y) position. Used to reset the
    /// ball onto a reachable spot after it leaves the field or after a goal, so
    /// the match keeps running in self-play.
    fn place_ball(&mut self, x: f64, y: f64) {
        self.with_referee(|referee, ctx| referee.place_ball(ctx, x, y))
    }

    #[cfg(test)]
    fn ball_out(&mut self) -> bool {
        self.with_referee(|referee, ctx| referee.ball_out(ctx))
    }

    #[cfg(test)]
    fn valid_free_kick_position(&self, desired: Vector2) -> Vector2 {
        crate::referee::valid_free_kick_position(&self.config.field_geometry, desired)
    }

    /// Apply referee effects before the next observable simulation phase.
    fn with_referee<T>(
        &mut self,
        update: impl FnOnce(&mut AutoReferee, &mut RefereeContext<'_>) -> T,
    ) -> T {
        let mut ctx =
            RefereeContext::new(&self.config, self.current_time, &self.physics, &self.robots);
        let result = update(&mut self.referee, &mut ctx);
        let actions = ctx.actions;
        for action in actions {
            match action {
                RefereeAction::PlaceBall { x, y } => self.physics.place_ball(x, y),
                RefereeAction::RemoveRobot { team, id } => {
                    self.robots.remove_robot(&mut self.physics, team, id)
                }
                RefereeAction::AddRobot {
                    team,
                    id,
                    position,
                    yaw,
                } => {
                    self.robots
                        .add_robot(&mut self.physics, &self.config, team, id, position, yaw)
                }
            }
        }
        result
    }
}

impl Default for Simulation {
    fn default() -> Self {
        Simulation::new(SimulationConfig::default())
    }
}

#[cfg(test)]
mod tests;
