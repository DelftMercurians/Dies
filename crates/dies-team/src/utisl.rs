// Enum to represent different obstacle types
#[derive(Debug, Clone, Serialize)]
pub enum Obstacle {
    Circle { center: Vector2, radius: f64 },
    Rectangle { min: Vector2, max: Vector2 },
}
pub fn get_obstacles_for_player(&self, role: RoleType) -> Vec<Obstacle> {
    if let Some(field_geom) = &self.field_geom {
        let field_boundary = {
            let hl = field_geom.field_length / 2.0;
            let hw = field_geom.field_width / 2.0;
            Obstacle::Rectangle {
                min: Vector2::new(
                    -hl - field_geom.boundary_width,
                    -hw - field_geom.boundary_width,
                ),
                max: Vector2::new(
                    hl + field_geom.boundary_width,
                    hw + field_geom.boundary_width,
                ),
            }
        };
        let mut obstacles = vec![field_boundary];

        // Add own defence area for non-keeper robots
        if role != RoleType::Goalkeeper {
            let lower = Vector2::new(-10_000.0, -field_geom.penalty_area_width / 2.0);
            let upper = Vector2::new(
                -field_geom.field_length / 2.0 + field_geom.penalty_area_depth + 50.0,
                field_geom.penalty_area_width / 2.0,
            );

            let defence_area = Obstacle::Rectangle {
                min: lower,
                max: upper,
            };
            obstacles.push(defence_area);
        }

        // Add opponent defence area for all robots
        let lower = Vector2::new(
            field_geom.field_length / 2.0 - field_geom.penalty_area_depth - 50.0,
            -field_geom.penalty_area_width / 2.0,
        );
        let upper = Vector2::new(10_0000.0, field_geom.penalty_area_width / 2.0);
        let defence_area = Obstacle::Rectangle {
            min: lower,
            max: upper,
        };
        obstacles.push(defence_area);

        match self.current_game_state.game_state {
            GameState::Stop => {
                // Add obstacle to prevent getting close to the ball
                if let Some(ball) = &self.ball {
                    obstacles.push(Obstacle::Circle {
                        center: ball.position.xy(),
                        radius: STOP_BALL_AVOIDANCE_RADIUS,
                    });
                }
            }
            GameState::Kickoff | GameState::PrepareKickoff => match role {
                RoleType::KickoffKicker => {}
                _ => {
                    // Add center circle for non kicker robots
                    obstacles.push(Obstacle::Circle {
                        center: Vector2::zeros(),
                        radius: field_geom.center_circle_radius,
                    });
                }
            },
            GameState::BallReplacement(_) => {}
            GameState::PreparePenalty => {}
            GameState::FreeKick => {}
            GameState::Penalty => {}
            GameState::PenaltyRun => {}
            GameState::Run | GameState::Halt | GameState::Timeout | GameState::Unknown => {
                // Nothing to do
            }
        };

        obstacles
    } else {
        vec![]
    }
}

#[derive(Debug, Clone)]
pub enum Avoid {
    Line { start: Vector2, end: Vector2 },
    Circle { center: Vector2 },
}

impl Avoid {
    fn distance_to(&self, pos: Vector2) -> f64 {
        match self {
            Avoid::Line { start, end } => distance_to_line(*start, *end, pos),
            Avoid::Circle { center } => (center - pos).norm(),
        }
    }
}

fn distance_to_line(start: Vector2, end: Vector2, pos: Vector2) -> f64 {
    let line = end - start;
    let line_norm = line.norm();
    let line_dir = line / line_norm;
    let pos_dir = pos - start;
    let proj = pos_dir.dot(&line_dir);
    if proj < 0.0 {
        (pos - start).norm()
    } else if proj > line_norm {
        (pos - end).norm()
    } else {
        let proj_vec = line_dir * proj;
        (pos_dir - proj_vec).norm()
    }
}

pub fn nearest_safe_pos(
    avoding_point: Avoid,
    min_distance: f64,
    initial_pos: Vector2,
    target_pos: Vector2,
    max_radius: i32,
    field: &FieldGeometry,
) -> Vector2 {
    let mut best_pos = Vector2::new(f64::INFINITY, f64::INFINITY);
    let mut found_better = false;
    let min_theta = 0;
    let max_theta = 360;
    let mut i = 0;
    for theta in (min_theta..max_theta).step_by(10) {
        let theta = Angle::from_degrees(theta as f64);
        for radius in (0..max_radius).step_by(50) {
            let position = initial_pos + theta.to_vector() * (radius as f64);
            if is_pos_in_field(position, field)
                && avoding_point.distance_to(position) > min_distance
            {
                if (position - target_pos).norm() < (best_pos - target_pos).norm() {
                    // crate::debug_cross(format!("{i}"), position, crate::DebugColor::Green);
                    best_pos = position;
                    found_better = true;
                }
            } else {
                // crate::debug_cross(format!("{i}"), position, crate::DebugColor::Red);
            }
            i += 1;
        }
    }
    if !found_better {
        log::warn!("Could not find a safe position from {initial_pos}, avoiding {avoding_point:?}");
    }

    best_pos
}

pub fn is_pos_in_field(pos: Vector2, field: &FieldGeometry) -> bool {
    const MARGIN: f64 = 100.0;
    // check if pos outside field
    if pos.x.abs() > field.field_length / 2.0 - MARGIN
        || pos.y.abs() > field.field_width / 2.0 - MARGIN
    {
        return false;
    }

    true
}
