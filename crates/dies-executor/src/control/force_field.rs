use dies_core::{
    debug_circle_stroke, debug_line, debug_remove, debug_string, PlayerData, Vector2, WorldData,
};

/// The influence factor constants
const INFLUENCE_FACTOR: (f64, f64) = (2.4, 2.0);

/// The player radius in mm
const PLAYER_RADIUS: f64 = 70.0;

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
    target: &Vector2,
    world: &WorldData,
    alpha: f64,
    beta: f64,
) -> Vector2 {
    let (base_factor, speed_factor) = INFLUENCE_FACTOR;
    let pos = target_player.position;
    let velocity = target_player.velocity;

    let distance = (target - pos).norm();
    if distance < f64::EPSILON {
        return Vector2::zeros();
    }

    // Check if there is a straight path to the target
    let mut has_path = true;
    let direction = (target - pos).normalize();
    for other_player in world.opp_players.iter() {
        if target_player.position == other_player.position {
            continue;
        }

        // Compute distance to straight line between target and player
        let v = other_player.position - pos;
        // Project v onto direction
        let projection = direction.dot(&v) * direction;
        let perp = v - projection;
        if perp.norm() < base_factor * PLAYER_RADIUS {
            has_path = false;
            break;
        }
    }
    if has_path {
        return direction;
    }

    let mut f: Vector2 = direction * alpha / (distance + f64::EPSILON);

    for other_player in world.opp_players.iter() {
        // Could compare id, but opponent players can have the same id as own players
        // This is a lazy way around that -- probably not the best way to do it
        if target_player.position == other_player.position {
            continue;
        }

        let d = (other_player.position - pos).norm();
        let relative_speed = (pos - other_player.position)
            .try_normalize(f64::EPSILON)
            .unwrap_or(Vector2::zeros())
            .dot(&(velocity - other_player.velocity))
            .abs();
        let influence_radius = base_factor * PLAYER_RADIUS + relative_speed * speed_factor;
        if d < influence_radius {
            // Compute the repulsive force
            // f = (1 / (d - 2 * r) - 1 / (influence_radius - 2 * r)) * (relative_speed * (beta + 1))
            let repulsion = (1.0 / ((d - 2.0 * PLAYER_RADIUS) + f64::EPSILON)
                - 1.0 / ((influence_radius - 2.0 * PLAYER_RADIUS) + f64::EPSILON))
                * (relative_speed * (beta));
            let direction = (pos - other_player.position).normalize();
            let repulsive_force = direction * repulsion;
            debug_line(
                format!("repulsion_{}", other_player.id),
                other_player.position,
                other_player.position + repulsive_force,
                dies_core::DebugColor::Purple,
            );
            f += repulsive_force;
        } else {
            debug_remove(format!("repulsion_{}", other_player.id));
        }
    }

    f.try_normalize(f64::EPSILON).unwrap_or(Vector2::zeros())
}
