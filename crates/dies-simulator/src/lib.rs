use rapier3d::prelude::*;

const PLAYER_RADIUS: f32 = 200.0;
const PLAYER_HEIGHT: f32 = 140.0;
const PLAYER_MASS: f32 = 1.0;

pub struct SimulationConfig {
    pub gravity: Vector<f32>,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        SimulationConfig {
            gravity: Vector::y() * -9.81 * 1000.0,
        }
    }
}

struct Player {
    id: u32,
    is_own: bool,
    rigid_body_handle: RigidBodyHandle,
    collider_handle: ColliderHandle,
}

pub struct Simulation {
    gravity: Vector<f32>,
    rigid_body_set: RigidBodySet,
    collider_set: ColliderSet,
    integration_parameters: IntegrationParameters,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: BroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,
    query_pipeline: QueryPipeline,

    players: Vec<Player>,
}

impl Simulation {
    pub fn new(config: SimulationConfig) -> Simulation {
        let mut simulation = Simulation {
            gravity: config.gravity,
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
            players: Vec::new(),
        };

        // Add players
        for i in 0..3 {
            let rigid_body = RigidBodyBuilder::dynamic()
                .translation(Vector::new(i as f32 * 2.0, 5.0, 0.0))
                .build();
            let collider = ColliderBuilder::cuboid(1.0, 1.0, 1.0).build();
            let rigid_body_handle = simulation.rigid_body_set.insert(rigid_body);
            let collider_handle = simulation.collider_set.insert_with_parent(
                collider,
                rigid_body_handle,
                &mut simulation.rigid_body_set,
            );
            simulation.players.push(Player {
                id: i,
                is_own: true,
                rigid_body_handle,
                collider_handle,
            })
        }

        simulation
    }

    pub fn step(&mut self) {
        self.physics_pipeline.step(
            &self.gravity,
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
}
