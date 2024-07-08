use std::{borrow::Cow, fmt::format, vec};

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
                radius: 50.0,
                avoidance_responsibility: 1.0,
            },
        }
    }
}

impl Role for TestRole {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        
        // CONSTANTS OF GEOMETRY

        let target = Vector2::new(2000.0, 2000.0);
        let size = 250;
        let a = Vector2::new(-size as f64, -size as f64);
        let b = Vector2::new(-size as f64, size as f64);
        let c = Vector2::new(size as f64, size as f64);
        let d = Vector2::new(size as f64, -size as f64);

        // DRAWING
        dbg_draw!(("p{}.target", ctx.player.id), cross, target);

        dbg_draw!(("p{}.obs1", ctx.player.id), line, a, b);
        dbg_draw!(("p{}.obs2", ctx.player.id), line, b, c);
        dbg_draw!(("p{}.obs3", ctx.player.id), line, c, d);
        dbg_draw!(("p{}.obs4", ctx.player.id), line, d, a);

        // match invoke_skill!(ctx, GoToPositionSkill::new(target)) {
        //     SkillResult::Continue(mut input) => {
        //         let desired_vel = match input.velocity {
        //             Velocity::Global(v) => Vec2::new(v.x as f32, v.y as f32),
        //             Velocity::Local(_) => return input,
        //         };

        //         // PlayerControlInput::new()
        //         let obstacles = vec![
        //             Obstacle::Closed {
        //                 vertices: vec![
        //                     Vec2::new(a.x as f32, a.y as f32),
        //                     Vec2::new(b.x as f32, b.y as f32),
        //                     Vec2::new(c.x as f32, c.y as f32),
        //                     Vec2::new(d.x as f32, d.y as f32),
        //                 ],
        //             },
        //             // Add more obstacles here.
        //         ];
        //         let nearby_obstacles = obstacles
        //             .iter()
        //             .map(|obstacle| Cow::Borrowed(obstacle))
        //             .collect::<Vec<_>>();

        //         let time_horizon = 3.0;
        //         let obstacle_time_horizon = 1.0;

        //         let avoidance_velocity : Vec2 = self.agent.compute_avoiding_velocity(
        //             // Neighbors - other players
        //             &vec![],
        //             // Obstacles
        //             nearby_obstacles.as_slice(),
        //             desired_vel,
        //             100.0,
        //             ctx.world.dt as f32,
        //             &AvoidanceOptions {
        //                 obstacle_margin: 0.1,
        //                 time_horizon,
        //                 obstacle_time_horizon,
        //             },
        //         );

        //         input.velocity = Velocity::global(Vector2::new(
        //             avoidance_velocity.x as f64,
        //             avoidance_velocity.y as f64,
        //         ));
        //         dies_core::debug_line(format!("p{}.orca_velocity", ctx.player.id), ctx.player.position, ctx.player.position + Vector2::new(avoidance_velocity.x as f64, avoidance_velocity.y as f64), dies_core::DebugColor::Purple);
        //         dies_core::debug_value(format!("p{}.orca_vel_x", ctx.player.id), avoidance_velocity.x as f64);
        //         dies_core::debug_value(format!("p{}.orca_vel_y", ctx.player.id), avoidance_velocity.y as f64);
        //         input
        //     }
        //     SkillResult::Done => PlayerControlInput::new(),
        // }

        // let target = Vector2::new(1000.0, 1000.0);
        // dbg_draw!(("p{}.target", ctx.player.id), cross, target);
        // skill!(ctx, GoToPositionSkill::new(target));
        // // skill!(ctx, GoToPositionSkill::new(target2));
        let mut input = PlayerControlInput::new();
        let preferred_vel = target - ctx.player.position;
        let max_speed = 100000.0;
        input.velocity = Velocity::global(Vector2::new(preferred_vel.x , preferred_vel.y).normalize() * max_speed);

        self.agent.position = Vec2::new(ctx.player.position.x as f32, ctx.player.position.y as f32);
        self.agent.velocity = Vec2::new(ctx.player.velocity.x as f32, ctx.player.velocity.y as f32);
        let desired_vel = match input.velocity {
                        Velocity::Global(v) => Vec2::new(v.x as f32, v.y as f32),
                        Velocity::Local(_) => return input,
                    };
    
                    // PlayerControlInput::new()
                    let obstacles: Vec<Cow<'static, Obstacle>> = vec![
                                Cow::Owned(Obstacle::Closed{
                                    vertices: vec![
                                    Vec2::new(a.x as f32, a.y as f32),
                                    Vec2::new(b.x as f32, b.y as f32),
                                    Vec2::new(c.x as f32, c.y as f32),
                                    Vec2::new(d.x as f32, d.y as f32),
                                    ],
                                }),
                                // Add more obstacles here.
                                ];

                    
                    let nearby_obstacles = obstacles
                                .iter()
                                .map(|obstacle| obstacle.clone())
                                .collect::<Vec<Cow<'_, Obstacle>>>();
                    let time_horizon = 100.0;
                    let obstacle_time_horizon = 10.0;
    
                    let avoidance_velocity : Vec2 = self.agent.compute_avoiding_velocity(
                        // Neighbors - other players
                        &vec![],
                        // Obstacles
                        &nearby_obstacles,
                        desired_vel,
                        1000.0,
                        ctx.world.dt as f32,
                        &AvoidanceOptions {
                            obstacle_margin: 1.,
                            time_horizon,
                            obstacle_time_horizon,
                        },
                    );
    
                    input.velocity = Velocity::global(Vector2::new(
                        avoidance_velocity.x as f64,
                        avoidance_velocity.y as f64,
                    ));
                    dies_core::debug_line(format!("p{}.orca_velocity", ctx.player.id), ctx.player.position, ctx.player.position + Vector2::new(avoidance_velocity.x as f64, avoidance_velocity.y as f64), dies_core::DebugColor::Purple);
                    dies_core::debug_value(format!("p{}.orca_vel_x", ctx.player.id), avoidance_velocity.x as f64);
                    dies_core::debug_value(format!("p{}.orca_vel_y", ctx.player.id), avoidance_velocity.y as f64);
                    input
    }
}
