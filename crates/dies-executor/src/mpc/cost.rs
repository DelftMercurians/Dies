use dies_core::{Vector2, WorldData};

use super::{control_output::ControlOutput, state::State, target::MpcTarget, MpcConfig};

pub fn cost(config: &MpcConfig, targets: &Vec<MpcTarget>, world: &WorldData, u: &[f64]) -> f64 {
    let num_timesteps = config.timesteps();
    let num_players = world.own_players.len();
    let mut state = State::new(world);
    let u = ControlOutput::new(u, num_players, num_timesteps);

    let mut cost = 0.0;
    for timestep in 0..num_timesteps {
        let current_u = u.timestep(timestep);

        // Calculate cost for each player
        for (target, player) in targets.iter().zip(state.own_players.iter()) {
            cost += target.cost(player.position, player.orientation, state.ball_position());
        }

        state.step(config.dt, &current_u);
    }

    println!("Cost: {}", cost);
    cost
}

pub fn cost_grad(
    config: &MpcConfig,
    targets: &Vec<MpcTarget>,
    world: &WorldData,
    u: &[f64],
    grad: &mut [f64],
) {
    let num_timesteps = config.timesteps();
    let num_players = world.own_players.len();
    let mut state = State::new(world);
    let u = ControlOutput::new(u, num_players, num_timesteps);

    for timestep in 0..num_timesteps {
        let current_u = u.timestep(timestep);

        // Calculate cost for each player
        for (player_idx, (target, player)) in
            targets.iter().zip(state.own_players.iter()).enumerate()
        {
            let grad_idx = timestep * 3 * num_players + player_idx * 3;
            let (vel_grad, ang_vel_grad) = target.cost_grad(
                config.dt,
                player.position,
                player.orientation,
                state.ball_position(),
            );

            grad[grad_idx] = vel_grad.x;
            grad[grad_idx + 1] = vel_grad.y;
            grad[grad_idx + 2] = ang_vel_grad;
        }

        state.step(config.dt, &current_u);
    }
}

#[cfg(test)]
mod test {
    use std::f64::consts::{FRAC_PI_2, PI};

    use dies_core::{PlayerData, PlayerId, Vector2, WorldData};

    use crate::mpc::{
        control_output::initialize_u,
        state::{OwnPlayerState, State},
        target::{HeadingTarget, MpcTarget, PositionTarget},
        MpcConfig,
    };

    use super::cost;

    #[test]
    fn test_position_target_grad_step() {
        let num_players = 4;
        let config = MpcConfig {
            dt: 0.1,
            time_horizon: 1.0,
        };

        let world = WorldData {
            own_players: (0..num_players)
                .map(|i| PlayerData {
                    id: PlayerId::new(i),
                    position: Vector2::new((i + 1) as f64 * 100.0, 0.0),
                    orientation: 0.0,
                    raw_position: Vector2::new(0.0, 0.0),
                    velocity: Vector2::new(0.0, 0.0),
                    angular_speed: 0.0,
                    timestamp: 0.0,
                })
                .collect(),
            ..Default::default()
        };
        let targets: Vec<_> = (0..num_players)
            .map(|_| {
                let target = PositionTarget::ConstantPosition(Vector2::new(0.0, 0.0));
                MpcTarget {
                    position_target: target,
                    heading_target: HeadingTarget::None,
                }
            })
            .collect();

        let u_init: Vec<f64> = initialize_u(&world, config.timesteps());
        let mut u_grad = vec![0.0; u_init.len()];

        let cost1 = cost(&config, &targets, &world, &u_init);
        super::cost_grad(&config, &targets, &world, &u_init, &mut u_grad);

        let epsilon = 1e-4;
        let mut u = u_init;
        for i in 0..u.len() {
            // Take a step in the -gradient direction
            u[i] += epsilon * -u_grad[i];
        }

        let cost2 = cost(&config, &targets, &world, &u);
        assert!(cost2 < cost1);
    }
}
