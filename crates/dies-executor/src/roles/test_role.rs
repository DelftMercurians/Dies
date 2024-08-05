use std::time::Instant;

use dies_core::Vector2;

use super::RoleCtx;
use crate::{roles::Role, PlayerControlInput};

pub struct TestRole {
    targets: Vec<Vector2>,
    start_time: Option<Instant>,
    predicted_time: Option<f64>,
    current_target: usize,
}

impl Role for TestRole {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        if self.current_target >= self.targets.len() {
            return PlayerControlInput::default();
        }

        let target = self.targets[self.current_target];
        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
            self.predicted_time = Some(ctx.world.time_to_reach_point(ctx.player, target));
        }
        let distance = (target - ctx.player.position).magnitude();
        if distance < 72.0 {
            self.current_target += 1;

            if let (Some(start_time), Some(predicted_time)) = (self.start_time, self.predicted_time)
            {
                let elapsed = start_time.elapsed().as_secs_f64();
                dies_core::debug_string(
                    format!("p{}.prediction", ctx.player.id),
                    format!("Elapsed: {:.2}, Predicted: {:.2}", elapsed, predicted_time),
                );
            }

            self.start_time = None;
            if self.current_target >= self.targets.len() {
                return PlayerControlInput::default();
            }
        }

        let mut input = PlayerControlInput::default();
        input.with_position(target);
        input
    }
}
