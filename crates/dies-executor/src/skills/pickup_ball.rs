use dies_core::{Angle, Vector2};

use crate::{control::Velocity, PlayerControlInput};
use super::{SkillCtx, SkillProgress};

const DEFAULT_POS_TOLERANCE: f64 = 50.0;
const DEFAULT_YAW_TOLERANCE: f64 = 5.0;
const DEFAULT_DISTANCE_LIMIT: f64 = 200.0;
//const DEFAULT_VEL_TOLERANCE: f64 = 20.0;


/// A skill that picks up the ball from a given angle
/// (angle given is the angle robot should be heading at the moment of pickup)
#[derive(Clone)]
pub struct PickUpBall {
    approach_angle: Angle,
    distance_limit: f64,
    pos_tolerance: f64,
    yaw_tolerance: f64,
    go_capture: bool,
}

impl PickUpBall {
    pub fn new(approach_angle: Angle) -> Self {
        Self {
            approach_angle,
            distance_limit: DEFAULT_DISTANCE_LIMIT,
            pos_tolerance: DEFAULT_POS_TOLERANCE,
            yaw_tolerance: DEFAULT_YAW_TOLERANCE,
            go_capture: false,
        }
    }

    pub fn with_distance_limit(mut self, limit: f64) -> Self {
        self.distance_limit = limit;
        self
    }

    pub fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        if let Some(ball) = ctx.world.ball.as_ref() {
            let mut input = PlayerControlInput::new();

            let ball_pos = ball.position.xy();
            let player_pos = ctx.player.position;
            let ball_heading = Angle::between_points(player_pos, ball_pos);
            let player_heading = Angle::from_degrees(ctx.player.yaw.degrees());
            
            let offset = Vector2::new(
                self.approach_angle.radians().cos() * self.distance_limit,
                self.approach_angle.radians().sin() * self.distance_limit,
            );
            let approach_pos = ball_pos - offset;

            input.with_position(approach_pos);
            input.with_yaw(ball_heading);
            input.avoid_ball = true;
            //input.avoid_ball_care = 500.0; //???? doesnt do anything
            //Avoiding the ball to ensure proper pickup still sabotaged by huge breakbeam range
            
            // log::info!(
            //     "p_heading={:.2}, a_angle={:.2},\n player_pos={:?}, approach_pos={:?}",
            //     player_heading.degrees(), self.approach_angle.degrees(), player_pos, approach_pos
            // );

            let player_approach_heading = player_heading - self.approach_angle;
            //^2 used as an abs value
            if(player_approach_heading.degrees()*player_approach_heading.degrees() 
            < self.yaw_tolerance*self.yaw_tolerance || self.go_capture == true)  {
                input.avoid_ball = false;
                input.with_position(ball_pos);
                input.with_dribbling(0.5);
                self.go_capture = true;
                log::info!("PickUpBall: Inside yaw tolerance ");

                // log::info!(
                //     "PickUpBall aligned: player_approach_heading={:.2}, player_heading={:.2}, ball_heading={:.2}",
                //     player_approach_heading, player_heading, ball_heading
                // );
            }
            
            //TODO: Determine whether the robot should rotate to given angle after pickup
            let breakbeam = ctx.player.breakbeam_ball_detected;
            if(breakbeam){
                log::info!("PickUpBall: Ball captured successfully!");
                return SkillProgress::success();
            }
            return SkillProgress::Continue(input);
        } else {
            // Wait for the ball to appear
            SkillProgress::Continue(PlayerControlInput::default())
        }
    }
}

