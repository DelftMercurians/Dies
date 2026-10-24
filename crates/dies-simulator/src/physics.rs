//! Rapier world, field construction, body lifecycle, and integration.
use crate::{ball::BALL_RADIUS, config::SimulationConfig};
use dies_core::Vector2;
use rapier3d_f64::prelude::*;
const GROUND_THICKNESS: f64 = 10.0;
const WALL_HEIGHT: f64 = 1000.0;
const WALL_THICKNESS: f64 = 1.0;

#[derive(Debug)]
pub(crate) struct Ball {
    pub(crate) _rigid_body_handle: RigidBodyHandle,
    pub(crate) _collider_handle: ColliderHandle,
}

pub(crate) struct PhysicsWorld {
    pub(crate) rigid_body_set: RigidBodySet,
    pub(crate) collider_set: ColliderSet,
    pub(crate) integration_parameters: IntegrationParameters,
    pub(crate) physics_pipeline: PhysicsPipeline,
    pub(crate) island_manager: IslandManager,
    pub(crate) broad_phase: BroadPhase,
    pub(crate) narrow_phase: NarrowPhase,
    pub(crate) impulse_joint_set: ImpulseJointSet,
    pub(crate) multibody_joint_set: MultibodyJointSet,
    pub(crate) ccd_solver: CCDSolver,
    pub(crate) query_pipeline: QueryPipeline,
    pub(crate) ball: Option<Ball>,
}

impl PhysicsWorld {
    pub(crate) fn new(config: &SimulationConfig) -> Self {
        let field_length =
            config.field_geometry.field_length + 2.0 * config.field_geometry.boundary_width;
        let field_width =
            config.field_geometry.field_width + 2.0 * config.field_geometry.boundary_width;
        let goal_width = config.field_geometry.goal_width;
        let goal_depth =
            config.field_geometry.field_length / 2.0 + config.field_geometry.goal_depth;
        let boundary_width = config.field_geometry.boundary_width;
        let mut world = Self {
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            integration_parameters: IntegrationParameters::default(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: BroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),
            ball: None,
        };
        // Create the ground
        let ground_body = RigidBodyBuilder::fixed()
            // z=0.0 is the ground surface
            .translation(Vector::new(0.0, 0.0, -GROUND_THICKNESS / 2.0))
            .build();
        let ground_collider =
            ColliderBuilder::cuboid(field_length / 2.0, field_width / 2.0, GROUND_THICKNESS)
                .build();
        let ground_body_handle = world.rigid_body_set.insert(ground_body);
        world.collider_set.insert_with_parent(
            ground_collider,
            ground_body_handle,
            &mut world.rigid_body_set,
        );

        // Create the field walls
        world.add_wall(0.0, field_width / 2.0, field_length / 2.0, WALL_THICKNESS);
        world.add_wall(0.0, -field_width / 2.0, field_length / 2.0, WALL_THICKNESS);
        world.add_wall(field_length / 2.0, 0.0, WALL_THICKNESS, field_width / 2.0);
        world.add_wall(-field_length / 2.0, 0.0, WALL_THICKNESS, field_width / 2.0);
        // Create the goal walls
        world.add_wall(
            -field_length / 2.0 + boundary_width / 2.0,
            goal_width / 2.0,
            boundary_width / 2.0,
            WALL_THICKNESS,
        );
        world.add_wall(
            -field_length / 2.0 + boundary_width / 2.0,
            -goal_width / 2.0,
            boundary_width / 2.0,
            WALL_THICKNESS,
        );
        world.add_wall(
            field_length / 2.0 - boundary_width / 2.0,
            goal_width / 2.0,
            boundary_width / 2.0,
            WALL_THICKNESS,
        );
        world.add_wall(
            field_length / 2.0 - boundary_width / 2.0,
            -goal_width / 2.0,
            boundary_width / 2.0,
            WALL_THICKNESS,
        );
        world.add_wall(-goal_depth, 0.0, WALL_THICKNESS, goal_width / 2.0);
        world.add_wall(goal_depth, 0.0, WALL_THICKNESS, goal_width / 2.0);

        world
    }

    pub(crate) fn add_wall(&mut self, x: f64, y: f64, half_width: f64, half_length: f64) {
        let wall_body = RigidBodyBuilder::fixed()
            .translation(Vector::new(x, y, 0.0))
            .build();
        let wall_collider = ColliderBuilder::cuboid(half_width, half_length, WALL_HEIGHT)
            .restitution(0.0)
            .restitution_combine_rule(CoefficientCombineRule::Min)
            .build();
        let wall_body_handle = self.rigid_body_set.insert(wall_body);
        self.collider_set.insert_with_parent(
            wall_collider,
            wall_body_handle,
            &mut self.rigid_body_set,
        );
    }

    pub(crate) fn apply_force_to_ball(&mut self, force: Vector<f64>) {
        if let Some(ball) = self.ball.as_ref() {
            let ball_body = self
                .rigid_body_set
                .get_mut(ball._rigid_body_handle)
                .unwrap();
            ball_body.apply_impulse(force, true);
        }
    }

    /// Teleport the ball to a field position (z held at rest height) and zero
    /// its velocity. Used by scenarios to reset to a clean state between drills.
    pub(crate) fn teleport_ball(&mut self, position: Vector2) {
        if let Some(ball) = self.ball.as_ref() {
            let handle = ball._rigid_body_handle;
            if let Some(ball_body) = self.rigid_body_set.get_mut(handle) {
                ball_body.set_position(
                    Isometry::translation(position.x, position.y, BALL_RADIUS),
                    true,
                );
                ball_body.set_linvel(Vector::zeros(), true);
                ball_body.set_angvel(Vector::zeros(), true);
            }
        }
    }

    pub(crate) fn update_queries(&mut self) {
        self.query_pipeline
            .update(&self.rigid_body_set, &self.collider_set);
    }

    pub(crate) fn step(&mut self, config: &SimulationConfig, dt: f64) {
        self.integration_parameters.dt = dt;
        self.physics_pipeline.step(
            &config.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            Some(&mut self.query_pipeline),
            &(),
            &(),
        );
    }

    pub(crate) fn clamp_ball(&mut self, config: &SimulationConfig) {
        // Clamp ball position to the field
        if let Some(ball) = self.ball.as_ref() {
            let ball_body = self
                .rigid_body_set
                .get_mut(ball._rigid_body_handle)
                .unwrap();
            let position = ball_body.position().translation.vector;
            let x = position.x.clamp(
                -config.field_geometry.field_length / 2.0 - config.field_geometry.boundary_width,
                config.field_geometry.field_length / 2.0 + config.field_geometry.boundary_width,
            );
            let y = position.y.clamp(
                -config.field_geometry.field_width / 2.0 - config.field_geometry.boundary_width,
                config.field_geometry.field_width / 2.0 + config.field_geometry.boundary_width,
            );
            let z = position.z.clamp(24.0, 100.0);
            ball_body.set_position(Isometry::translation(x, y, z), true);
        }
    }

    pub(crate) fn place_ball(&mut self, x: f64, y: f64) {
        if let Some(ball) = self.ball.as_ref() {
            let ball_body = self
                .rigid_body_set
                .get_mut(ball._rigid_body_handle)
                .unwrap();
            ball_body.set_position(Isometry::translation(x, y, BALL_RADIUS), true);
            ball_body.set_linvel(Vector::zeros(), true);
            // Also kill the spin: a ball that rolled out carries angular velocity,
            // and ground friction would convert that residual spin back into
            // linear motion, dragging the freshly-placed ball off its spot and
            // back over the line during the Stop period — which re-triggers
            // ball-out and collapses the free-kick restart into a force-start.
            ball_body.set_angvel(Vector::zeros(), true);
        }
    }

    pub(crate) fn spawn_ball(&mut self, config: &SimulationConfig, position: Vector<f64>) {
        let ball_body = RigidBodyBuilder::dynamic()
            .can_sleep(false)
            .translation(position)
            .linear_damping(config.ball_damping)
            .build();
        let ball_collider = ColliderBuilder::ball(BALL_RADIUS)
            .mass(1.0)
            .restitution(0.0)
            .restitution_combine_rule(CoefficientCombineRule::Min)
            .build();
        let ball_body_handle = self.rigid_body_set.insert(ball_body);
        let ball_collider_handle = self.collider_set.insert_with_parent(
            ball_collider,
            ball_body_handle,
            &mut self.rigid_body_set,
        );
        self.ball = Some(Ball {
            _rigid_body_handle: ball_body_handle,
            _collider_handle: ball_collider_handle,
        });
    }
}
