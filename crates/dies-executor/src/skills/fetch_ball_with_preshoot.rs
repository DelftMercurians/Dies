use std::{sync::Arc, time::Duration, time::Instant};
use dies_core::{Angle, Vector2, BALL_RADIUS, PLAYER_RADIUS};
use crate::{control::Velocity, PlayerControlInput};

use crate::skills::{
    FetchBall, GoToPosition, Shoot, SkillCtx, SkillProgress, SkillResult,
};
use crate::control::{
    find_best_preshoot_heading, find_best_preshoot_target, find_best_preshoot_target_target,
    find_best_shoot_target, PassingStore, ShootTarget
};

#[derive(Clone)]
pub struct FetchBallWithPreshoot {
    preshoot_position: Vector2,
    preshoot_heading: Option<dies_core::Angle>,
    shoot_target: ShootTarget,
    state: FetchBallWithPreshootState,
}

#[derive(Clone)]
enum FetchBallWithPreshootState {
    None,
    GoToPreshoot(GoToPosition),
    FetchBall(FetchBall),
    Shoot(Shoot),
}

impl FetchBallWithPreshoot {
    pub fn new(preshoot_position: Vector2, shoot_target: ShootTarget) -> Self {
        Self {
            preshoot_position,
            preshoot_heading: None,
            shoot_target,
            state: FetchBallWithPreshootState::None,
        }
    }

    pub fn with_heading(mut self, heading: dies_core::Angle) -> Self {
        self.preshoot_heading = Some(heading);
        self
    }

    pub fn state(&self) -> String {
        match &self.state {
            FetchBallWithPreshootState::None => "None".to_string(),
            FetchBallWithPreshootState::GoToPreshoot(_) => {
                format!("GoToPreshoot: {:?}", self.preshoot_position)
            }
            FetchBallWithPreshootState::FetchBall(_) => "FetchBall".to_string(),
            FetchBallWithPreshootState::Shoot(_) => format!("Shoot: {:?}", self.shoot_target),
        }
    }

    pub fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        // go to preshoot:
        if let Some(ball) = ctx.world.ball.as_ref() {
            let mut input = PlayerControlInput::new();
            input.with_dribbling(0.6);

            let ball_pos = ball.position.xy();
            let ball_speed = ball.velocity.xy().norm();
            let player_pos = ctx.player.position;

            // let's compute the preshoot position 'correctly':
            // if we are at the ball's position and have a ball, where would we shoot?
            let shooting_target = find_best_preshoot_target_target(&PassingStore::new(ctx.player.id, Arc::new(ctx.world.clone()))).position().unwrap();
            let prep_target = ball_pos - (shooting_target - ball_pos).normalize() * 200.0;

            if (prep_target - player_pos).magnitude() < 50.0 {
                return SkillProgress::Done(SkillResult::Success);
            }

            // now, let's just go to prep_target
            let target_vel = (prep_target - player_pos) * 2.0;
            let target_vel = if target_vel.magnitude() > 2000.0 {
                target_vel.normalize() * 2000.0
            } else { target_vel };

            input.velocity = Velocity::global(
                target_vel,
            );

            SkillProgress::Continue(input)
        } else {
            // Wait for the ball to appear
            SkillProgress::Continue(PlayerControlInput::default())
        }
    }
}
