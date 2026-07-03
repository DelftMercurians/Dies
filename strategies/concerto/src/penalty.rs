//! Penalty-kick taking (rules §5.3.5).
//!
//! The executor enforces every positioning rule of the penalty family — all
//! non-kicker/non-keeper robots are parked at the field end, speed-capped and
//! clamped ≥1m behind the ball, keyed on the `penalty_kicker` role — so
//! concerto's whole job here is: pin one taker, stage it behind the ball during
//! `PreparePenalty`, and fire a single reflex strike the moment `Penalty`
//! (normal start) arrives. The planner is deliberately bypassed: the mark is
//! farther from goal than [`config::KICKER_SHOT_RANGE`], and every planner
//! fallback (kick-ahead / recycle / pass) is illegal with all teammates parked
//! behind the ball. `strike_from_possession` (acquire → aim → reflex-fire, one
//! continuous touch, no carry) also keeps the excessive-dribbling and
//! only-toward-goal constraints safe by construction.

use dies_strategy_api::prelude::*;
use dies_strategy_api::World;

use crate::config;
use crate::geometry;

/// Latched per-penalty state, reset whenever the game state leaves the penalty
/// family (see `ConcertoStrategy::update`).
#[derive(Default)]
pub struct PenaltyState {
    /// The latched aim point on the opponent goal line. Held under
    /// [`config::PENALTY_AIM_STICK`] hysteresis so the Strike aiming loop does
    /// not re-orbit on per-frame keeper jitter.
    aim: Option<Vector2>,
}

impl PenaltyState {
    pub fn reset(&mut self) {
        if self.aim.take().is_some() {
            debug::remove("penalty_phase");
            debug::remove("penalty_aim");
        }
    }
}

/// The penalty mark we shoot from: on the goal-to-goal line,
/// `goal_line_to_penalty_mark` (6m Div B) from the opponent goal center. Note
/// this lies in OUR half on a Div-B field (9m long) — nothing here may assume
/// the mark is near the opponent goal. Only used as a fallback origin while the
/// ball is undetected; the live ball position always wins.
pub fn opp_penalty_mark(world: &World) -> Vector2 {
    let d = world
        .field()
        .map(|f| f.goal_line_to_penalty_mark)
        .unwrap_or(6000.0);
    Vector2::new(world.field_length() / 2.0 - d, 0.0)
}

/// Pick the aim point: the best open shot window (keeper-shadow aware, same
/// model as open play), falling back to the corner farther from the keeper when
/// no window resolves. The keeper is line-bound until the ball is in play, so
/// its full on-line shadow radius applies.
fn fresh_aim(world: &World) -> Vector2 {
    let origin = world
        .ball_position()
        .unwrap_or_else(|| opp_penalty_mark(world));
    let goal = world.opp_goal_center();
    let keeper_id = geometry::opp_keeper(world);
    if let Some(shot) = geometry::best_shot(
        origin,
        goal.x,
        world.goal_width(),
        world.opp_players(),
        keeper_id,
        config::SHOT_ROBOT_RADIUS,
        config::SHOT_KEEPER_RADIUS,
        config::BALL_RADIUS,
        config::SHOT_KEEPER_BIAS,
    ) {
        return shot.target;
    }
    let keeper_y = keeper_id
        .and_then(|id| world.opp_players().iter().find(|p| p.id == id))
        .map(|p| p.position.y)
        .unwrap_or(0.0);
    let y = (world.goal_width() / 2.0 - config::PENALTY_AIM_POST_MARGIN).max(0.0);
    Vector2::new(goal.x, if keeper_y > 0.0 { -y } else { y })
}

fn penalty_aim(world: &World, prev: Option<Vector2>) -> Vector2 {
    let fresh = fresh_aim(world);
    match prev {
        Some(p) if (fresh - p).norm() < config::PENALTY_AIM_STICK => p,
        _ => fresh,
    }
}

/// Drive the penalty taker for this frame.
///
/// `PreparePenalty`: stand [`config::PENALTY_PREP_STANDOFF`] behind the ball
/// along the attacking axis, already facing the latched aim (may approach, must
/// not touch). `Penalty` (normal start) and `PenaltyRun`: strike at the latched
/// aim — a rebound during `PenaltyRun` re-engages the same strike (double touch
/// does not apply to penalties, rules §8.2).
pub fn update_taker(
    state: &mut PenaltyState,
    world: &World,
    ctx: &mut TeamContext,
    kicker: PlayerId,
    game_state: GameState,
) {
    let aim = penalty_aim(world, state.aim);
    state.aim = Some(aim);
    debug::cross("penalty_aim", aim);

    let Some(p) = ctx.player(kicker) else {
        return;
    };
    p.set_role("penalty_kicker");
    if game_state == GameState::PreparePenalty {
        debug::string("penalty_phase", "prepare");
        let ball = world
            .ball_position()
            .unwrap_or_else(|| opp_penalty_mark(world));
        let attack_dir = (world.opp_goal_center() - ball)
            .try_normalize(1e-6)
            .unwrap_or_else(|| Vector2::new(1.0, 0.0));
        p.go_to(ball - attack_dir * config::PENALTY_PREP_STANDOFF)
            .facing(aim);
    } else {
        debug::string("penalty_phase", "strike");
        p.strike_from_possession(aim);
    }
}
