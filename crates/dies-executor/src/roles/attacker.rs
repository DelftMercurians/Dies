use std::f64::consts::FRAC_PI_2;

use super::{passer, receiver, RoleCtx, SkillResult};
use crate::{
    invoke_skill,
    roles::{
        skills::{ApproachBall, Face, FetchBall, GoToPosition, Kick},
        Role,
    },
    skill, KickerControlInput, PlayerControlInput,
};
use dies_core::{
    Angle, BallData, FieldGeometry, GameState, PlayerData, PlayerId, Vector2, WorldData,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttackerSection {
    Mid,
    Left,
    Right,
}

impl AttackerSection {
    pub fn y_bounds(&self, field: &FieldGeometry) -> (f64, f64) {
        match self {
            AttackerSection::Mid => (-field.field_width / 6.0, field.field_width / 6.0),
            AttackerSection::Left => (-field.field_width / 2.0 + 200.0, -200.0),
            AttackerSection::Right => (200.0, field.field_width / 2.0 - 200.0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttackerState {
    Positioning,
    FetchingBall,
    Dribbling,
    Passing(PlayerId),
    Shooting,
}

/// A role that moves the player to the intersection of the ball's path with the goal
/// line, acting as a wall to block the ball from reaching the goal.
pub struct Attacker {
    section: AttackerSection,
    state: AttackerState,
    next_state: Option<AttackerState>,
    dribbling_start: Option<Vector2>,
    position_cache: PositionCache,
    has_passed_to_receiver: Option<PlayerId>,
}

impl Attacker {
    /// Create a new Waller role with the given offset from the intersection point.
    pub fn new(initial_state: AttackerState, section: AttackerSection) -> Self {
        Self {
            section,
            state: initial_state,
            next_state: None,
            dribbling_start: None,
            position_cache: PositionCache::new(10),
            has_passed_to_receiver: None,
        }
    }

    pub fn section(&self) -> AttackerSection {
        self.section
    }

    pub fn passed_to_receiver(&mut self) -> Option<PlayerId> {
        self.has_passed_to_receiver.take()
    }

    pub fn receive(&mut self) {
        if matches!(self.state, AttackerState::Positioning) {
            self.next_state = Some(AttackerState::FetchingBall);
        }
    }

    pub fn start_positioning(&mut self) {
        self.next_state = Some(AttackerState::Positioning);
    }

    pub fn has_ball(&self) -> bool {
        matches!(
            self.state,
            AttackerState::Dribbling | AttackerState::Passing(_) | AttackerState::Shooting
        )
    }

    pub fn fetching_ball(&self) -> bool {
        !matches!(self.state, AttackerState::Positioning)
    }
}

impl Role for Attacker {
    fn update(&mut self, mut ctx: RoleCtx<'_>) -> PlayerControlInput {
        let mut input = PlayerControlInput::new();
        if ctx.world.current_game_state.game_state == GameState::Stop {
            ctx.reset_skills();
            let mut input = PlayerControlInput::new();
            input.with_speed_limit(1300.0);
            input.avoid_ball = true;
            if let Some(ball) = ctx.world.ball.as_ref() {
                let ball_pos = ball.position.xy();
                let dist = (ball_pos - ctx.player.position.xy()).norm();
                if dist < 560.0 {
                    // Move away from the ball
                    let target =
                        ball_pos.xy() + (ctx.player.position - ball_pos.xy()).normalize() * 650.0;
                    input.with_position(target);
                }
            }
            return input;
        }

        if let (Some(ball), Some(geom)) = (ctx.world.ball.as_ref(), ctx.world.field_geom.as_ref()) {
            let ball_angle = Angle::between_points(ctx.player.position, ball.position.xy());
            let ball_pos = ball.position.xy();
            let ball_dist = distance(ctx.player.position, ball_pos);
            let ball_speed = ball.velocity.norm();
            let goal_pos = Vector2::new(
                4500.0,
                f64::max(f64::min(ctx.player.position.y, 400.0), -400.0),
            );

            if let Some(next_state) = self.next_state.take() {
                if next_state != self.state {
                    println!("Attacker: {:?} -> {:?}", self.state, next_state);
                    self.state = next_state;
                    self.dribbling_start = None;
                    self.has_passed_to_receiver = None;
                    self.position_cache.reset();
                    ctx.reset_skills();
                }
            }

            let new_state = match self.state {
                AttackerState::Positioning => {
                    // Positioning ourselves to best receive the ball
                    let target_pos = self.position_cache.get_or_insert_with(|| {
                        find_best_striker_position(ctx.world, &self.section, geom, ctx.player)
                    });
                    input.with_position(target_pos);
                    input.with_yaw(ball_angle);

                    // If the ball is close and slow enough, start fetching it
                    if ball_speed < 300.0 {
                        AttackerState::FetchingBall
                    } else {
                        AttackerState::Positioning
                    }
                }
                AttackerState::FetchingBall => loop {
                    if ball.position.x < -1000.0 {
                        break AttackerState::Positioning;
                    }

                    match skill!(ctx, FetchBall::new()) {
                        crate::roles::SkillResult::Success => {
                            break AttackerState::Dribbling;
                        }
                        _ => {}
                    }
                },
                AttackerState::Dribbling => {
                    let starting_pos = *self.dribbling_start.get_or_insert(ctx.player.position);

                    let (best_pos, best_receiver, best_score) =
                        find_best_passer_position(starting_pos, 600.0, ctx.world, geom, ctx.player);
                    if distance(starting_pos, ctx.player.position) > 600.0 {
                        println!("Dribbling too far, passing");
                        if let Some(receiver) = best_receiver {
                            if score_line_of_sight(
                                &ctx.world,
                                ctx.player.position,
                                goal_pos,
                                geom,
                                &ctx.player,
                            ) > 100.0
                            {
                                AttackerState::Shooting
                            } else if best_score > 50.0 {
                                AttackerState::Passing(receiver)
                            } else if ctx.player.position.x > 1000.0 {
                                AttackerState::Shooting
                            } else {
                                AttackerState::Passing(receiver)
                            }
                        } else {
                            AttackerState::Shooting
                        }
                    } else {
                        if !ctx.player.breakbeam_ball_detected {
                            println!("Lost ball, fetching");
                            AttackerState::FetchingBall
                        } else {
                            match skill!(ctx, GoToPosition::new(best_pos).with_ball()) {
                                crate::roles::SkillResult::Success => AttackerState::Shooting,
                                _ => AttackerState::Dribbling,
                            }
                        }
                    }
                }
                AttackerState::Passing(receiver) => loop {
                    if !ctx.player.breakbeam_ball_detected {
                        println!("Lost ball, fetching");
                        break AttackerState::FetchingBall;
                    } else {
                        match invoke_skill!(ctx, Face::towards_own_player(receiver).with_ball()) {
                            crate::roles::SkillProgress::Continue(mut input) => {
                                input.with_dribbling(1.0);
                                return input;
                            }
                            _ => {}
                        }
                        if let SkillResult::Success = skill!(ctx, Kick::new()) {
                            self.has_passed_to_receiver = Some(receiver);
                            break AttackerState::Positioning;
                        }
                        match skill!(ctx, ApproachBall::new()) {
                            crate::roles::SkillResult::Success => continue,
                            _ => break AttackerState::Positioning,
                        }
                    }
                },
                AttackerState::Shooting => {
                    if !ctx.player.breakbeam_ball_detected {
                        println!("Lost ball, fetching");
                        AttackerState::FetchingBall
                    } else {
                        match invoke_skill!(
                            ctx,
                            Face::towards_position(Vector2::new(
                                4500.0,
                                f64::max(f64::min(ctx.player.position.y, 400.0), -400.0)
                            ))
                            .with_ball()
                        ) {
                            crate::roles::SkillProgress::Continue(mut input) => {
                                input.with_dribbling(1.0);
                                return input;
                            }
                            _ => {}
                        }
                        if let SkillResult::Success = skill!(ctx, Kick::new()) {
                            AttackerState::Positioning
                        } else {
                            AttackerState::Shooting
                        }
                    }
                }
            };

            if new_state != self.state {
                self.next_state = Some(new_state);
            }

            input
        } else {
            PlayerControlInput::new()
        }
    }
}

/// A cache that holds a position and a counter that is incremented every time the position is
/// accessed. The position is refreshed every `refresh_interval` accesses.
struct PositionCache {
    position: Option<Vector2>,
    counter: u32,
    refresh_interval: u32,
}

impl PositionCache {
    fn new(refresh_interval: u32) -> Self {
        Self {
            position: None,
            counter: 0,
            refresh_interval,
        }
    }

    fn get_or_insert_with(&mut self, f: impl FnOnce() -> Vector2) -> Vector2 {
        self.counter += 1;
        if let Some(position) = self.position {
            if self.counter >= self.refresh_interval {
                self.counter = 0;
                self.position = Some(f());
            }
            position
        } else {
            let pos = f();
            self.position = Some(pos);
            pos
        }
    }

    fn reset(&mut self) {
        self.position = None;
        self.counter = 0;
    }
}

/// Calculate the distance between two points.
fn distance(a: Vector2, b: Vector2) -> f64 {
    (a - b).norm()
}

/// Find a position with the best line of sight to the goal withing the given section.
fn find_best_striker_position(
    world: &WorldData,
    section: &AttackerSection,
    field: &FieldGeometry,
    player: &PlayerData,
) -> Vector2 {
    let (min_y, max_y) = section.y_bounds(field);
    let min_x = 100.0;
    let max_x = field.field_length / 2.0 - 100.0;

    let mut best_position = Vector2::new(0.0, 0.0);
    let mut best_score = -1e10;

    for x in (min_x as i32..max_x as i32).step_by(40) {
        for y in (min_y as i32..max_y as i32).step_by(40) {
            let position = Vector2::new(x as f64, y as f64);
            if !is_pos_valid(position, field) {
                continue;
            }

            let ball_score = score_line_of_sight(
                world,
                position,
                world
                    .ball
                    .as_ref()
                    .map(|b| b.position.xy())
                    .unwrap_or_default(),
                field,
                player,
            );
            let goal_score = score_line_of_sight(
                world,
                position,
                Vector2::new(field.field_length / 2.0, 0.0),
                field,
                player,
            );
            let goal_dist_score =
                3_000_000.0 / (position - Vector2::new(field.field_length / 2.0, 0.0)).norm();
            let mut ball_dist = 0.0;
            if let Some(ball) = world.ball.as_ref() {
                ball_dist = (ball.position.xy() + ball.velocity.xy() * 0.5 - position).norm();
            }
            let score = ball_score * 0.5 + goal_score + goal_dist_score - ball_dist * 20.0;
            if score > best_score {
                best_score = score;
                best_position = position;
            }
        }
    }

    best_position
}

fn find_best_passer_position(
    starting_pos: Vector2,
    max_radius: f64,
    world: &WorldData,
    field: &FieldGeometry,
    player: &PlayerData,
) -> (Vector2, Option<PlayerId>, f64) {
    let mut best_position = Vector2::new(0.0, 0.0);
    let mut best_score = 0.0;
    let mut best_striker = None;

    let attackers = world
        .own_players
        .iter()
        .filter(|p| p.position.x > 0.0)
        .collect::<Vec<_>>();

    let min_theta = -FRAC_PI_2;
    let max_theta = FRAC_PI_2;
    for theta in (min_theta as i32..max_theta as i32).step_by(10) {
        let theta = theta as f64;
        for radius in (0..max_radius as i32).step_by(20) {
            let x = starting_pos.x + (radius as f64) * theta.cos();
            let y = starting_pos.y + (radius as f64) * theta.sin();
            let position = Vector2::new(x, y);
            if !is_pos_valid(position, field) {
                continue;
            }

            let striker_score = attackers
                .iter()
                .map(|p| {
                    (
                        p.id,
                        score_line_of_sight(world, position, p.position, field, player),
                    )
                })
                .max_by_key(|&x| x.1 as i64);
            let goal_score = score_line_of_sight(
                world,
                position,
                Vector2::new(field.field_length / 2.0, 0.0),
                field,
                player,
            );
            if let Some((striker_id, score)) = striker_score {
                let score = score + goal_score;
                if score > best_score {
                    best_score = score;
                    best_position = position;
                    best_striker = Some(striker_id);
                }
            }
        }
    }

    (best_position, best_striker, best_score)
}

fn is_pos_valid(pos: Vector2, field: &FieldGeometry) -> bool {
    const MARGIN: f64 = 100.0;
    // check if pos inside penalty area
    if pos.x.abs() > field.field_length / 2.0 - field.penalty_area_depth - MARGIN
        && pos.y.abs() < field.penalty_area_width / 2.0 + MARGIN
    {
        return false;
    }
    true
}

/// Compute a "badness" score for a line of sight between two points based on the minumum
/// distance to the line of sight from the closest enemy player.
///
/// The score is higher if the line of sight is further from the enemy players.
fn score_line_of_sight(
    world: &WorldData,
    from: Vector2,
    to: Vector2,
    field: &FieldGeometry,
    player: &PlayerData,
) -> f64 {
    let mut min_distance = f64::MAX;
    for player in world.opp_players.iter() {
        let distance = distance_to_line(from, to, player.position);
        if distance < min_distance {
            min_distance = distance;
        }
    }
    if to.x > 3800.0 {
        min_distance = 0.0;
    }
    min_distance - (from.y.abs() / 4.0).max(40.0) - (from.x.abs() / 4.0).max(40.0) - (player.position - from).magnitude().max(100.0)
}

fn distance_to_line(a: Vector2, b: Vector2, p: Vector2) -> f64 {
    let n = (b - a).normalize();
    let ap = p - a;
    let proj = ap.dot(&n);
    let proj = proj.max(0.0).min((b - a).norm());
    (ap - proj * n).norm()
}
