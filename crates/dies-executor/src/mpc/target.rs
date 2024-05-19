use std::f64::consts::PI;

use dies_core::Vector2;
use nalgebra::DMatrix;

#[derive(Debug, Clone)]
pub enum PositionTarget {
    ConstantPosition(Vector2),
    BallPosition,
    Offset {
        target: Box<PositionTarget>,
        offset: Vector2,
    },
}

impl PositionTarget {
    pub fn cost(&self, position: Vector2, ball_position: Option<Vector2>) -> f64 {
        match self {
            PositionTarget::ConstantPosition(target) => (position - target).norm(),
            PositionTarget::BallPosition => {
                if let Some(ball_position) = ball_position {
                    (position - ball_position).norm()
                } else {
                    tracing::warn!("Cannot compute ball position target cost, no ball found");
                    0.0
                }
            }
            PositionTarget::Offset { target, offset } => {
                let target_position = target.cost(position, ball_position);
                target_position + offset.norm()
            }
        }
    }

    pub fn cost_grad(&self, position: Vector2, ball_position: Option<Vector2>) -> Vector2 {
        match self {
            PositionTarget::ConstantPosition(target) => euclidian_dist_grad(&position, target),
            PositionTarget::BallPosition => {
                if let Some(ball_position) = ball_position {
                    euclidian_dist_grad(&position, &ball_position)
                } else {
                    tracing::warn!(
                        "Cannot compute ball position target cost gradient, no ball found"
                    );
                    Vector2::zeros()
                }
            }
            PositionTarget::Offset { target, offset } => {
                let target_grad = target.cost_grad(position, ball_position);
                let with_offset = target_grad + offset;
                euclidian_dist_grad(&position, &with_offset)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum HeadingTarget {
    None,
    ConstantHeading(f64),
}

impl HeadingTarget {
    pub fn cost(&self, heading: f64) -> f64 {
        match self {
            HeadingTarget::None => 0.0,
            HeadingTarget::ConstantHeading(target) => {
                let diff = angle_diff(*target, heading);
                diff.abs()
            }
        }
    }

    pub fn cost_grad(&self, heading: f64) -> f64 {
        match self {
            HeadingTarget::None => 0.0,
            HeadingTarget::ConstantHeading(target) => {
                let diff = angle_diff(*target, heading);
                // Normalize to (-1, 1)
                diff / PI
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct MpcTarget {
    pub position_target: PositionTarget,
    pub heading_target: HeadingTarget,
}

impl MpcTarget {
    pub fn cost(&self, position: Vector2, heading: f64, ball_position: Option<Vector2>) -> f64 {
        let position_cost = self.position_target.cost(position, ball_position);
        let heading_cost = self.heading_target.cost(heading);
        position_cost + heading_cost
    }

    pub fn cost_grad(
        &self,
        position: Vector2,
        heading: f64,
        ball_position: Option<Vector2>,
    ) -> (Vector2, f64) {
        (
            self.position_target.cost_grad(position, ball_position),
            self.heading_target.cost_grad(heading),
        )
    }
}

/// Computes the gradient with respect to `x` of the euclidian distance between `x` and `y`.
fn euclidian_dist_grad(x: &Vector2, y: &Vector2) -> Vector2 {
    let diff = x - y;
    let norm = diff.norm();
    if norm > 0.0 {
        diff / norm
    } else {
        Vector2::zeros()
    }
}

/// Computes the signed distance between two angles (in radians)
fn angle_diff(target: f64, phi: f64) -> f64 {
    // Reference from Python:
    //  (target - phi + np.pi) % (2 * np.pi) - np.pi
    (target - phi + PI).rem_euclid(2.0 * PI) - PI
}
