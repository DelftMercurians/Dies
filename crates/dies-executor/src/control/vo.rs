use dies_core::{perp, Angle, PlayerData, Vector2};

const PLAYER_RADIUS: f64 = 80.0;

pub enum Obstacle {
    Circular {
        center: Vector2,
        velocity: Vector2,
        radius: f64,
    },
    Rectangular {
        center: Vector2,
        width: f64,
        height: f64,
    },
}

pub struct ContraintCone {
    pub apex: Vector2,
    pub left: Vector2,
    pub right: Vector2,
}

pub fn compute_velocity_constraints(
    current_player: &PlayerData,
    other_players: &[PlayerData],
    obstacles: &[Obstacle],
    time_horizon: f64,
) -> Vec<ContraintCone> {
    let mut constraints = Vec::new();

    // Compute constraints for other players
    for other in other_players {
        if other.position != current_player.position {
            let relative_position = other.position - current_player.position;
            let relative_velocity = other.velocity - current_player.velocity;
            let combined_radius = 2.0 * PLAYER_RADIUS;
            let distance = relative_position.norm();

            if distance < combined_radius {
                log::error!(
                    "Player at {:.0} is overlapping with player at {:.0}",
                    current_player.position,
                    other.position
                );
            } else {
                let center_rotation =
                    Angle::between_points(current_player.position, other.position);
                let dtheta = Angle::from_radians((combined_radius / distance).asin());
                let left_rotation = center_rotation + dtheta;
                let right_rotation = center_rotation - dtheta;
                let left_tangent = left_rotation * Vector2::x();
                let right_tangent = right_rotation * Vector2::x();

                constraints.push(ContraintCone {
                    apex: relative_velocity * time_horizon,
                    left: left_tangent,
                    right: right_tangent,
                });
            }
        }
    }

    // Compute constraints for obstacles
    for obstacle in obstacles {
        match obstacle {
            Obstacle::Circular {
                center,
                velocity,
                radius,
            } => {
                let relative_position = center - current_player.position;
                let relative_velocity = *velocity - current_player.velocity;
                let combined_radius = PLAYER_RADIUS + radius;
                let distance = relative_position.norm();

                if distance < combined_radius {
                    log::error!(
                        "Player at {:.0} is overlapping with obstacle at {:.0}",
                        current_player.position,
                        center
                    );
                } else {
                    let center_rotation = Angle::between_points(current_player.position, *center);
                    let dtheta = Angle::from_radians((combined_radius / distance).asin());
                    let left_rotation = center_rotation + dtheta;
                    let right_rotation = center_rotation - dtheta;
                    let left_tangent = left_rotation * Vector2::x();
                    let right_tangent = right_rotation * Vector2::x();

                    constraints.push(ContraintCone {
                        apex: relative_velocity * time_horizon,
                        left: left_tangent,
                        right: right_tangent,
                    });
                }
            }
            Obstacle::Rectangular {
                center,
                width,
                height,
            } => {
                let half_width = width / 2.0;
                let half_height = height / 2.0;
                let corners = [
                    *center + Vector2::new(-half_width, -half_height),
                    *center + Vector2::new(half_width, -half_height),
                    *center + Vector2::new(half_width, half_height),
                    *center + Vector2::new(-half_width, half_height),
                ];

                for i in 0..4 {
                    let corner1 = corners[i];
                    let corner2 = corners[(i + 1) % 4];
                    let edge = corner2 - corner1;
                    let edge_normal = perp(edge).normalize();
                    let relative_position = corner1 - current_player.position;

                    let projected_distance = relative_position.dot(&edge.normalize());
                    // Closest point to the player on the edge
                    let closest_point = if projected_distance < 0.0 {
                        corner1
                    } else if projected_distance > edge.norm() {
                        corner2
                    } else {
                        corner1 + edge.normalize() * projected_distance
                    };

                    let to_closest = closest_point - current_player.position;
                    let distance = to_closest.norm();

                    if distance < PLAYER_RADIUS {
                        log::error!(
                            "Player at {:.0} is overlapping with rectangular obstacle edge",
                            current_player.position
                        );
                    } else {
                        let center_rotation =
                            Angle::between_points(current_player.position, closest_point);
                        let dtheta = Angle::from_radians((PLAYER_RADIUS / distance).asin());
                        let left_rotation = center_rotation + dtheta;
                        let right_rotation = center_rotation - dtheta;

                        let left_tangent = left_rotation * Vector2::x();
                        let right_tangent = right_rotation * Vector2::x();

                        constraints.push(ContraintCone {
                            apex: Vector2::zeros(), // No velocity for static rectangular obstacle
                            left: left_tangent,
                            right: right_tangent,
                        });
                    }
                }
            }
        }
    }

    constraints
}
