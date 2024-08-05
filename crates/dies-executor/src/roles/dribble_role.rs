use dies_core::{Angle, Vector2};

use super::RoleCtx;
use crate::{
    roles::{
        skills::{GoToPosition, Wait},
        Role,
    },
    skill, PlayerControlInput,
};

pub struct DribbleRole {}

impl Role for DribbleRole {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        if let Some(ball) = ctx.world.ball.as_ref() {
            let target = Vector2::zeros();
            let heading = Angle::between_points(target, ball.position.xy());
            let offset = heading * Vector2::new(-100.0, 0.0);
            let position = ball.position.xy() + offset;
            skill!(ctx, GoToPosition::new(position).with_heading(heading));
            skill!(ctx, Wait::new_secs_f64(1.0));
        }

        PlayerControlInput::default()
    }
}
