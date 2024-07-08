use std::{borrow::Cow, vec};

use dies_core::{dbg_draw, Vector2};
use dodgy_2d::{Agent, AvoidanceOptions, Obstacle, Vec2};

use crate::{
    control::Velocity,
    invoke_skill,
    roles::{skills::GoToPositionSkill, Role, SkillResult},
    skill, PlayerControlInput,
};

use super::{skills, RoleCtx};

pub struct TestRole {
    agent: Agent,
}

impl TestRole {
    pub fn new() -> Self {
        Self {
            agent: Agent {
                position: Vec2::ZERO,
                velocity: Vec2::ZERO,
                radius: 5.0,
                avoidance_responsibility: 1.0,
            },
        }
    }
}

impl Role for TestRole {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        // dbg_draw!(("p{}.target", ctx.player.id), cross, Vector2::new(0.0, 0.0));
        match invoke_skill!(ctx, GoToPositionSkill::new(Vector2::new(0.0, 0.0))) {
            SkillResult::Continue(mut input) => {
                let desired_vel = match input.velocity {
                    Velocity::Global(v) => Vec2::new(v.x as f32, v.y as f32),
                    Velocity::Local(_) => return input,
                };

                // PlayerControlInput::new()
                let obstacles = vec![
                    Obstacle::Closed {
                        vertices: vec![
                            Vec2::new(-1000.0, -1000.0),
                            Vec2::new(-1000.0, 1000.0),
                            Vec2::new(1000.0, 1000.0),
                            Vec2::new(1000.0, -1000.0),
                        ],
                    },
                    // Add more obstacles here.
                ];
                let nearby_obstacles = obstacles
                    .iter()
                    .map(|obstacle| Cow::Borrowed(obstacle))
                    .collect::<Vec<_>>();

                let time_horizon = 3.0;
                let obstacle_time_horizon = 1.0;

                let avoidance_velocity = self.agent.compute_avoiding_velocity(
                    // Neighbors - other players
                    &vec![],
                    // Obstacles
                    nearby_obstacles.as_slice(),
                    desired_vel,
                    100.0,
                    ctx.world.dt as f32,
                    &AvoidanceOptions {
                        obstacle_margin: 0.1,
                        time_horizon,
                        obstacle_time_horizon,
                    },
                );

                input.velocity = Velocity::global(Vector2::new(
                    avoidance_velocity.x as f64,
                    avoidance_velocity.y as f64,
                ));
                input
            }
            SkillResult::Done => PlayerControlInput::new(),
        }
    }
}
