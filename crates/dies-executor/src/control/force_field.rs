use dies_core::{PlayerData, Vector2, WorldData};

/// The influence factor constants
const INFLUENCE_FACTOR: (f64, f64) = (5.0, 1.0);

/// The player radius in mm
const PLAYER_RADIUS: f64 = 90.0;

/// Computes a force at a given position
///
/// # Arguments
///
/// * `target_player` - The player to compute the force for
/// * `target_velocity` - The target velocity of the player
/// * `world` - The world data
/// * `alpha` - The attractive force constant
/// * `beta` - The repulsive force constant
pub fn compute_force(
    target_player: &PlayerData,
    target_velocity: &Vector2,
    world: &WorldData,
    alpha: f64,
    beta: f64,
) -> Vector2 {
    let pos = target_player.position;
    let velocity = target_player.velocity;

    let mut f: Vector2 = target_velocity * alpha;

    let (base_factor, speed_factor) = INFLUENCE_FACTOR;
    for player in world.opp_players.iter() {
        // Could compare id, but opponent players can have the same id as own players
        // This is a lazy way around that -- probably not the best way to do it
        if target_player.position == player.position {
            continue;
        }

        let d = (player.position - pos).norm();
        let relative_speed = (pos - player.position)
            .try_normalize(f64::EPSILON)
            .unwrap_or(Vector2::zeros())
            .dot(&(velocity - player.velocity))
            .abs();
        let influence_radius = base_factor * PLAYER_RADIUS + relative_speed * speed_factor;
        if d < influence_radius {
            let repulsive_force = (1.0 / ((d - 2.0 * PLAYER_RADIUS) + f64::EPSILON)
                - 1.0 / ((influence_radius - 2.0 * PLAYER_RADIUS) + f64::EPSILON))
                * (relative_speed * (beta + 1.0));

            f += (pos - player.position) * repulsive_force;
        }
    }

    f
}
