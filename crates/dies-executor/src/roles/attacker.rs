use super::RoleCtx;
use crate::{
    roles::{skills::FetchBall, Role},
    skill, KickerControlInput, PlayerControlInput,
};
use dies_core::{Angle, BallData, FieldGeometry, PlayerId, Vector2, WorldData};

/// A role that moves the player to the intersection of the ball's path with the goal
/// line, acting as a wall to block the ball from reaching the goal.
pub struct Attacker {
    position: Vector2,
    passer_id: Option<PlayerId>,
    shooter_id: Option<PlayerId>,
    passer_kicked: bool,
    starting_pos: Vector2,
}

impl Attacker {
    /// Create a new Waller role with the given offset from the intersection point.
    pub fn new(position: Vector2) -> Self {
        Self {
            position,
            starting_pos: Vector2::new(0.0, 0.0),
            passer_id: None,
            shooter_id: None,
            passer_kicked: false,
        }
    }

    fn aim_at_goal(&self, player_pos: Vector2, world: &FieldGeometry) -> Angle {
        let opp_goal_center = Vector2::new(-world.field_length / 2.0, 0.0);

        Angle::between_points(player_pos, opp_goal_center)
    }

    fn aim_at_player(&self, passer_pos: Vector2, reciever_pos: Vector2) -> Angle {
        Angle::between_points(passer_pos, reciever_pos)
    }

    fn aim_at_ball(&self, player_pos: Vector2, ball_pos: Vector2) -> Angle {
        Angle::between_points(player_pos, ball_pos)
    }

    fn closest_player_to_ball(&self, world: &WorldData, ball: &BallData) -> Option<PlayerId> {
        let mut min_distance = f64::MAX;
        let mut closest_player_id = None;
        for player in world.own_players.iter() {
            let distance = distance(player.position, ball.position.xy());
            if distance < min_distance {
                min_distance = distance;
                closest_player_id = Some(player.id);
            }
        }
        closest_player_id
    }

    fn closest_player_to_passer(
        &self,
        world: &WorldData,
        passer_pos: Vector2,
        passer_id: PlayerId,
    ) -> Option<PlayerId> {
        let mut min_distance = f64::MAX;
        let mut closest_player_id = None;
        for player in world.own_players.iter() {
            if player.id == passer_id {
                continue;
            }
            let distance = distance(player.position, passer_pos);
            if distance < min_distance {
                min_distance = distance;
                closest_player_id = Some(player.id);
            }
        }
        closest_player_id
    }
}

impl Role for Attacker {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        if let (Some(ball), Some(geom)) = (ctx.world.ball.as_ref(), ctx.world.field_geom.as_ref()) {
            // Passer and Shooter only needs to be assigned once if not it creates many bugs
            if self.passer_id.is_none() {
                self.passer_id = self.closest_player_to_ball(ctx.world, ball);
                self.shooter_id = self.closest_player_to_passer(
                    ctx.world,
                    ctx.world
                        .own_players
                        .iter()
                        .find(|p| p.id == self.passer_id.unwrap())
                        .unwrap()
                        .position,
                    self.passer_id.unwrap(),
                );
            }
            // Getting the position of the passer and the shooter
            let passer_pos = ctx
                .world
                .own_players
                .iter()
                .find(|p| p.id == self.passer_id.unwrap())
                .unwrap()
                .position;
            let shooter_pos = ctx
                .world
                .own_players
                .iter()
                .find(|p| p.id == self.shooter_id.unwrap())
                .unwrap()
                .position;
            let mut input = PlayerControlInput::new();

            // If the player moved more than 1m then it should look for another player to pass the ball
            if distance(ctx.player.position, ball.position.xy()) < 500.0 {
                if self.starting_pos == Vector2::new(0.0, 0.0) {
                    self.starting_pos = ctx.player.position;
                }

                // dies_core::debug_string(format!("p{}", ctx.player.id))

                if distance(self.starting_pos, ctx.player.position) >= 950.0 {
                    println!("Moved more than 950 units");
                    // input.with_position(ctx.player.position);
                    let target_angle = self.aim_at_player(ctx.player.position, shooter_pos);
                    input.with_yaw(target_angle);
                    input.with_dribbling(1.0);
                    if (target_angle - ctx.player.yaw).abs() < 0.1 {
                        input.with_kicker(KickerControlInput::Kick);
                        self.starting_pos = Vector2::new(0.0, 0.0);
                    }
                    return input;
                }
            }

            // If the player is the passer
            if ctx.player.id == self.passer_id.unwrap() {
                let target_pos = ball.position.xy();
                let target_angle = self.aim_at_ball(ctx.player.position, target_pos);
                let dribble_speed = 1.0;

                // When the ball is close to the shooter, the passer moves back to its position (to make sure it doesn't endlessly follow the ball)
                if distance(shooter_pos, ball.position.xy()) < 600.0 {
                    input.with_position(self.position);
                    input.with_dribbling(0.0);
                    return input;
                }

                // If the ball is far from the passer, the passer moves to the ball and arms the kicker ready to pass
                if distance(ctx.player.position, ball.position.xy()) > 280.0 {
                    skill!(ctx, FetchBall::new());
                }

                // Once the passer has the ball and it is in position it aims for the shooter
                if distance(ctx.player.position, self.position) < 100.0 {
                    dies_core::debug_cross(
                        "ShooterPos",
                        shooter_pos,
                        dies_core::DebugColor::Purple,
                    );
                    let target_angle = self.aim_at_player(passer_pos, shooter_pos);
                    input.with_yaw(target_angle);
                    input.with_dribbling(1.0);

                    // If the passer is facing the shooter it passes the ball
                    if (target_angle - ctx.player.yaw).abs() < 0.1 {
                        println!("Passing the ball");
                        self.passer_kicked = true;
                        input.with_kicker(KickerControlInput::Kick);
                    }
                    return input;
                }
            }

            // If the player is the shooter it aims at the ball and moves to its designated position
            if ctx.player.id == self.shooter_id.unwrap() {
                dies_core::debug_cross("PasserPos", passer_pos, dies_core::DebugColor::Orange);
                let target_angle = self.aim_at_ball(shooter_pos, ball.position.xy());
                let dribble_speed = 1.0;
                input.with_yaw(target_angle);
                input.with_dribbling(dribble_speed);
                // If the shooter is close to the ball (meaning the passer passed the ball), it moves toward the ball to "catch" it
                // This should be changed to the parameter passer_has_kicked once it works!!
                if distance(shooter_pos, ball.position.xy()) < 800.0 {
                    while distance(shooter_pos, ball.position.xy()) > 280.0 {
                        println!("Moving to ball");
                        // Input to move to the ball and dribble.
                        input.with_position(ball.position.xy());
                        input.with_yaw(self.aim_at_ball(shooter_pos, ball.position.xy()));
                        input.with_dribbling(dribble_speed);
                        input.with_kicker(KickerControlInput::Arm);
                        return input;
                    }
                }
                // If the shooter has the ball, is in the designated spot it faces the goal
                if distance(shooter_pos, self.position) < 100.0
                    && distance(shooter_pos, ball.position.xy()) < 280.0
                {
                    let target_angle = self.aim_at_goal(shooter_pos, geom);
                    input.with_yaw(target_angle);
                    input.with_dribbling(1.0);

                    // If the shooter is facing the goal it kicks the ball
                    if (target_angle - ctx.player.yaw).abs() < 0.1 {
                        println!("GOLLLLLLLLLLL");
                        input.with_kicker(KickerControlInput::Kick);
                    }
                    return input;
                }
            }

            // This is the default input for all players, going to their designated position and facing the ball
            input.with_dribbling(1.0);
            input.with_position(self.position);
            input.with_yaw(self.aim_at_ball(ctx.player.position, ball.position.xy()));
            input
        } else {
            PlayerControlInput::new()
        }
    }
}

/// Calculate the distance between two points.
fn distance(a: Vector2, b: Vector2) -> f64 {
    (a - b).norm()
}
