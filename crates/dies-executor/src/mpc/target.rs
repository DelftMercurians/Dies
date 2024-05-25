use std::f64::consts::PI;

use dies_core::Vector2;

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
    fn target_position(&self, ball_position: Option<Vector2>) -> Vector2 {
        match self {
            PositionTarget::ConstantPosition(target) => *target,
            PositionTarget::BallPosition => ball_position.unwrap(),
            PositionTarget::Offset { target, offset } => {
                let target_position = target.target_position(ball_position);
                target_position + offset
            }
        }
    }

    pub fn cost(&self, position: Vector2, ball_position: Option<Vector2>) -> f64 {
        (position - self.target_position(ball_position)).norm()
    }

    pub fn cost_grad(&self, dt: f64, position: Vector2, ball_position: Option<Vector2>) -> Vector2 {
        let target_position = self.target_position(ball_position);
        // x[i] = x[i-1] + u[i] * dt
        // -> grad wrt u[i] = dt * (x[i] - x_target) / ||x[i] - x_target||
        dt * euclidian_dist_grad(&position, &target_position)
    }
}

#[derive(Debug, Clone)]
pub enum HeadingTarget {
    None,
    /// Target heading in radians \[-pi, pi\]
    ConstantHeading(f64),
}

impl HeadingTarget {
    pub fn cost(&self, heading: f64) -> f64 {
        match self {
            HeadingTarget::None => 0.0,
            HeadingTarget::ConstantHeading(target) => {
                let diff = angle_diff(heading, *target);
                diff.abs()
            }
        }
    }

    pub fn cost_grad(&self, dt: f64, heading: f64) -> f64 {
        match self {
            HeadingTarget::None => 0.0,
            HeadingTarget::ConstantHeading(target) => {
                let diff = angle_diff(heading, *target);
                // phi[i] = phi[i-1] + u[i] * dt
                // -> grad wrt u[i] = dt * sign(phi[i] - phi_target)
                dt * diff.signum()
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
        dt: f64,
        position: Vector2,
        heading: f64,
        ball_position: Option<Vector2>,
    ) -> (Vector2, f64) {
        (
            self.position_target.cost_grad(dt, position, ball_position),
            self.heading_target.cost_grad(dt, heading),
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

#[cfg(test)]
mod test {
    use std::f64::consts::{FRAC_PI_2, PI};

    use dies_core::Vector2;

    use crate::mpc::{
        state::{OwnPlayerState, State},
        target::{HeadingTarget, PositionTarget},
    };

    #[test]
    fn test_position_target_cost() {
        let target = PositionTarget::ConstantPosition(Vector2::new(1.0, 1.0));
        let position = Vector2::new(0.0, 0.0);
        assert_eq!(target.cost(position, None), (2.0_f64).sqrt());

        let target = PositionTarget::BallPosition;
        let position = Vector2::new(0.0, 0.0);
        let ball_position = Some(Vector2::new(1.0, 1.0));
        assert_eq!(target.cost(position, ball_position), (2.0_f64).sqrt());

        let target = PositionTarget::Offset {
            target: Box::new(PositionTarget::ConstantPosition(Vector2::new(1.0, 1.0))),
            offset: Vector2::new(1.0, 1.0),
        };
        let position = Vector2::new(0.0, 0.0);
        assert_eq!(target.cost(position, None), (8.0_f64).sqrt());
    }

    #[test]
    fn test_position_target_cost_grad() {
        let dt = 0.1;
        let target = PositionTarget::ConstantPosition(Vector2::new(1.0, 1.0));
        let position = Vector2::new(0.0, 0.0);
        assert_eq!(
            target.cost_grad(dt, position, None),
            Vector2::new(-1.0, -1.0) / (2.0_f64).sqrt() * dt
        );

        let target = PositionTarget::BallPosition;
        let position = Vector2::new(0.0, 0.0);
        let ball_position = Some(Vector2::new(1.0, 1.0));
        assert_eq!(
            target.cost_grad(dt, position, ball_position),
            Vector2::new(-1.0, -1.0) / (2.0_f64).sqrt() * dt
        );

        let target = PositionTarget::Offset {
            target: Box::new(PositionTarget::ConstantPosition(Vector2::new(1.0, 1.0))),
            offset: Vector2::new(1.0, 1.0),
        };
        let position = Vector2::new(0.0, 0.0);
        assert_eq!(
            target.cost_grad(dt, position, None),
            Vector2::new(-2.0, -2.0) / (8.0_f64).sqrt() * dt
        );
    }

    #[test]
    fn test_grad_step_pos_target() {
        // Test that making a step in the -grad direction decreases the cost
        let dt = 0.1;
        let state = State {
            own_players: vec![OwnPlayerState {
                position: Vector2::new(0.0, 0.0),
                orientation: 0.0,
            }],
            opp_players: vec![],
            ball: None,
        };
        let target = PositionTarget::ConstantPosition(Vector2::new(100.0, 0.0));

        let cost1 = target.cost(state.own_players[0].position, None);
        let grad = target.cost_grad(dt, state.own_players[0].position, None);
        let step_size = 0.1;
        let new_position = state.own_players[0].position - grad * step_size;
        let cost2 = target.cost(new_position, None);

        assert!(cost2 < cost1);
    }

    #[test]
    fn test_grad_step_heading_target() {
        // Test that making a step in the -grad direction decreases the cost
        let dt = 0.1;
        let state = State {
            own_players: vec![OwnPlayerState {
                position: Vector2::new(0.0, 0.0),
                orientation: 0.0,
            }],
            opp_players: vec![],
            ball: None,
        };
        let target = HeadingTarget::ConstantHeading(FRAC_PI_2);

        let cost1 = target.cost(state.own_players[0].orientation);
        let grad = target.cost_grad(dt, state.own_players[0].orientation);
        let step_size = 0.01;
        let new_orientation = state.own_players[0].orientation - grad * step_size;
        let cost2 = target.cost(new_orientation);

        assert!(cost2 < cost1);
    }

    #[test]
    fn test_grad_step_heading_wraparound() {
        // Test that making a step in the -grad direction decreases the cost
        let dt = 0.1;
        let state = State {
            own_players: vec![OwnPlayerState {
                position: Vector2::new(0.0, 0.0),
                orientation: -PI,
            }],
            opp_players: vec![],
            ball: None,
        };
        let target = HeadingTarget::ConstantHeading(PI - 0.1);

        let cost1 = target.cost(state.own_players[0].orientation);
        let grad = target.cost_grad(dt, state.own_players[0].orientation);
        let step_size = 0.01;
        let new_orientation = state.own_players[0].orientation - grad * step_size;
        // Wrap new_orientation to (-PI, PI)
        let new_orientation = (new_orientation + PI).rem_euclid(2.0 * PI) - PI;
        let cost2 = target.cost(new_orientation);

        assert!(cost2 < cost1);
    }
}
