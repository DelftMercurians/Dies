//! Builder APIs and original default/seeded match layouts.
use crate::{RealismConfig, Simulation, SimulationConfig};
use dies_core::{Angle, PlayerId, TeamColor, Vector2};
use rand::Rng;
use rapier3d_f64::prelude::*;

pub struct SimulationBuilder {
    sim: Simulation,
    last_blue_id: u32,
    last_yellow_id: u32,
}

impl SimulationBuilder {
    pub fn new(config: SimulationConfig) -> Self {
        SimulationBuilder {
            sim: Simulation::new(config),
            last_blue_id: 0,
            last_yellow_id: 0,
        }
    }

    pub fn add_blue_player_with_id(mut self, id: u32, position: Vector2, yaw: Angle) -> Self {
        self.add_player(id, TeamColor::Blue, position, yaw);
        self
    }

    pub fn add_yellow_player_with_id(mut self, id: u32, position: Vector2, yaw: Angle) -> Self {
        self.add_player(id, TeamColor::Yellow, position, yaw);
        self
    }

    pub fn add_blue_player(mut self, position: Vector2, yaw: Angle) -> Self {
        self.add_player(self.last_blue_id, TeamColor::Blue, position, yaw);
        self.last_blue_id += 1;
        self
    }

    pub fn add_yellow_player(mut self, position: Vector2, yaw: Angle) -> Self {
        self.add_player(self.last_yellow_id, TeamColor::Yellow, position, yaw);
        self.last_yellow_id += 1;
        self
    }

    /// Set which teams should be controlled by the simulation
    pub fn with_controlled_teams(mut self, blue_controlled: bool, yellow_controlled: bool) -> Self {
        self.sim.config.blue_controlled = blue_controlled;
        self.sim.config.yellow_controlled = yellow_controlled;
        self
    }

    pub fn add_ball(mut self, position: Vector<f64>) -> Self {
        self.sim.physics.spawn_ball(&self.sim.config, position);
        self
    }

    pub fn build(self) -> Simulation {
        self.sim
    }

    fn add_player(&mut self, id: u32, team_color: TeamColor, position: Vector2, yaw: Angle) {
        self.sim.robots.spawn_robot(
            &mut self.sim.physics,
            &self.sim.config,
            team_color,
            PlayerId::new(id),
            position,
            yaw,
        );
    }
}

impl SimulationBuilder {
    /// Like [`SimulationBuilder::default`] but with the given config, and with
    /// each robot's initial pose deterministically jittered by `config.seed`.
    /// The ball stays centered (kickoff repositions it anyway). Same seed →
    /// identical layout, so headless self-play matches are reproducible while
    /// different seeds produce different-but-reproducible openings.
    ///
    /// Draws are made in a fixed order (blue 0..6, then yellow 0..6) so the
    /// perturbation is a pure function of the seed. Jitter is bounded so robots
    /// never spawn overlapping: the line runs along x (tight spacing), so x
    /// jitter is kept small while y (toward the open center) is generous.
    pub fn default_seeded(config: SimulationConfig) -> Self {
        // Position jitter bounds (mm) and yaw jitter (rad). x is constrained by
        // the inter-robot spacing along the line; y has the open field.
        const JITTER_X: f64 = 30.0;
        const JITTER_Y: f64 = 150.0;
        const JITTER_YAW: f64 = 0.3;

        let mut builder = SimulationBuilder::new(config);
        let field_width = builder.sim.config.field_geometry.field_width / 2.0;
        let field_length = builder.sim.config.field_geometry.field_length / 2.0;
        let boundary_width = builder.sim.config.field_geometry.boundary_width;
        let player_radius = builder.sim.config.player_radius;
        let player_margin = 0.75 * player_radius;
        let sides = builder.sim.config.initial_side_assignment;
        let n_blue = builder.sim.config.n_blue_robots;
        let n_yellow = builder.sim.config.n_yellow_robots;

        let mut jitter = |builder: &mut SimulationBuilder| {
            let rng = &mut builder.sim.rng;
            (
                rng.gen_range(-JITTER_X..=JITTER_X),
                rng.gen_range(-JITTER_Y..=JITTER_Y),
                rng.gen_range(-JITTER_YAW..=JITTER_YAW),
            )
        };

        // Base x is negative so robots line up on their OWN half (team-relative
        // +x points at the enemy goal, so own half is x < 0); transform_vec2 then
        // maps that to the correct absolute side per the side assignment.
        for i in 0..n_blue {
            let (dx, dy, dyaw) = jitter(&mut builder);
            let position = Vector2::new(
                -(field_length - player_margin - i as f64 * (2.0 * player_radius + player_margin))
                    + dx,
                field_width - player_radius - boundary_width + dy,
            );
            builder = builder.add_blue_player(
                sides.transform_vec2(TeamColor::Blue, &position),
                Angle::from_radians(dyaw),
            );
        }
        for i in 0..n_yellow {
            let (dx, dy, dyaw) = jitter(&mut builder);
            let position = Vector2::new(
                -(field_length - player_margin - i as f64 * (2.0 * player_radius + player_margin))
                    + dx,
                -(field_width - player_radius - boundary_width) + dy,
            );
            builder = builder.add_yellow_player(
                sides.transform_vec2(TeamColor::Yellow, &position),
                Angle::from_radians(dyaw),
            );
        }

        builder = builder.add_ball(Vector::new(0.0, 0.0, 20.0));

        builder
    }
}

impl Default for SimulationBuilder {
    fn default() -> Self {
        // By default we add 6 players on each side, lined up on their respective
        // half of the field.
        //
        // Realism (sim2real hardware noise) is env-opt-in here, like in headless
        // self-play: setting `DIES_REALISM_LEVEL` / `DIES_REALISM_*` degrades the
        // default (interactive/webui) sim without a recompile. With no env set
        // this is the perfect actuator, so tests and normal runs are unaffected.
        let config = SimulationConfig {
            realism: RealismConfig::from_env(),
            ..Default::default()
        };
        let mut builder = SimulationBuilder::new(config);
        let field_width = builder.sim.config.field_geometry.field_width / 2.0;
        let field_length = builder.sim.config.field_geometry.field_length / 2.0;
        let boundary_width = builder.sim.config.field_geometry.boundary_width;
        let player_radius = builder.sim.config.player_radius;
        let player_margin = 0.75 * player_radius;
        let sides = builder.sim.config.initial_side_assignment;

        // Base x is negative so robots line up on their OWN half (team-relative
        // +x points at the enemy goal); transform_vec2 maps to the absolute side.
        for i in 0..6 {
            let position = Vector2::new(
                -(field_length - player_margin - i as f64 * (2.0 * player_radius + player_margin)),
                field_width - player_radius - boundary_width,
            );
            builder = builder.add_blue_player(
                sides.transform_vec2(TeamColor::Blue, &position),
                Angle::from_radians(0.0),
            );
        }
        for i in 0..6 {
            let position = Vector2::new(
                -(field_length - player_margin - i as f64 * (2.0 * player_radius + player_margin)),
                -(field_width - player_radius - boundary_width),
            );
            builder = builder.add_yellow_player(
                sides.transform_vec2(TeamColor::Yellow, &position),
                Angle::from_radians(0.0),
            );
        }

        builder = builder.add_ball(Vector::new(0.0, 0.0, 20.0));

        builder
    }
}
