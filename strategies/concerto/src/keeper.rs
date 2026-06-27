//! Goalkeeper — dedicated positioning + active clearing, outside Formation.
//!
//! Two modes:
//! - **Guard**: sit on a small arc around the goal mouth, on the bisector of the
//!   shot cone (the two tangents from the ball to the posts), tracking the ball.
//! - **Clear**: when a ball settles inside our defense area, pick it up and kick
//!   it out toward the opponent half (off the centre line), then return to Guard.
//!
//! The keeper runs an aggressive, ORCA-free, planner-free control profile (see
//! [`keeper_control_override`]). Disabling ORCA is what stops field-boundary
//! avoidance from ever overriding the keeper's commanded velocity.

use dies_strategy_api::prelude::*;
use dies_strategy_api::World;

use crate::config;

/// Keeper behaviour mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KeeperMode {
    /// Hold the arc and track the ball.
    Guard,
    /// A ball has settled in the box — pick it up and clear it.
    Clear,
}

/// Persistent keeper state across frames.
pub struct KeeperState {
    mode: KeeperMode,
}

impl Default for KeeperState {
    fn default() -> Self {
        Self::new()
    }
}

impl KeeperState {
    pub fn new() -> Self {
        Self {
            mode: KeeperMode::Guard,
        }
    }
}

/// The keeper's aggressive control profile: high position/braking gains, a tight
/// speed cap, and — crucially — ORCA and the planner disabled so nothing deflects
/// or reroutes its line motion (no field-boundary avoidance).
pub fn keeper_control_override() -> ControlOverride {
    ControlOverride::new()
        .aggressiveness(config::KEEPER_AGGRESSIVENESS)
        .brake_gain(config::KEEPER_BRAKE_GAIN)
        .speed_limit(config::KEEPER_SPEED_LIMIT)
        .avoid_robots(false)
        .use_planner(false)
}

/// Drive the keeper for this frame: set its role + control override and issue the
/// Guard/Clear skill commands.
pub fn update(state: &mut KeeperState, world: &World, keeper: &mut PlayerHandle) {
    keeper.set_role("goalkeeper");
    keeper.set_control_override(keeper_control_override());

    // ── Mode transitions ────────────────────────────────────────────────
    match state.mode {
        KeeperMode::Guard => {
            if should_clear(world) {
                state.mode = KeeperMode::Clear;
            }
        }
        KeeperMode::Clear => {
            if !clear_still_valid(world, keeper) {
                state.mode = KeeperMode::Guard;
            }
        }
    }

    // ── Act ─────────────────────────────────────────────────────────────
    match state.mode {
        KeeperMode::Guard => {
            let target = keeper_arc_target(
                world,
                config::KEEPER_ARC_RADIUS,
                config::KEEPER_ARC_MAX_ANGLE,
            );
            let face = world
                .ball_position()
                .unwrap_or_else(|| world.opp_goal_center());
            keeper.go_to(target).facing(face);
        }
        KeeperMode::Clear => {
            let ball = world.ball_position().unwrap_or_else(|| keeper.position());
            let target = clear_target(world, ball);
            let heading = facing_angle(ball, target);
            if keeper.has_ball() {
                let _ = keeper.reflex_shoot(target);
            } else {
                let _ = keeper.pickup_ball(heading);
            }
        }
    }
}

/// Compute the keeper's Guard target: the point on the arc of radius `radius`
/// around the goal centre that lies on the bisector of the shot cone, clamped to
/// `±max_angle` off straight-out.
pub fn keeper_arc_target(world: &World, radius: f64, max_angle: f64) -> Vector2 {
    let g = world.own_goal_center();
    let half_goal = world.goal_width() / 2.0;

    let ball = match world.ball_position() {
        Some(b) => b,
        // No ball: square up at the centre of the arc.
        None => return clamp_to_arc(g + Vector2::new(radius, 0.0), g, radius, max_angle),
    };

    let post_l = g + Vector2::new(0.0, half_goal);
    let post_r = g + Vector2::new(0.0, -half_goal);

    // Interior bisector of the cone (ball→left-post, ball→right-post).
    let bisector = match (
        (post_l - ball).try_normalize(1.0e-6),
        (post_r - ball).try_normalize(1.0e-6),
    ) {
        (Some(l), Some(r)) => (l + r).try_normalize(1.0e-6),
        _ => None,
    };

    let candidate = bisector
        .and_then(|u| ray_circle_front(ball, u, g, radius))
        .unwrap_or_else(|| {
            // Degenerate: fall back to the goal→ball direction on the arc.
            let d = (ball - g)
                .try_normalize(1.0e-6)
                .unwrap_or(Vector2::new(1.0, 0.0));
            g + d * radius
        });

    clamp_to_arc(candidate, g, radius, max_angle)
}

/// First intersection of the ray `from + t·dir` (t>0) with the circle `(centre,
/// radius)` — the one nearer `from` (the front of the circle). `None` if the ray
/// misses or only hits behind the origin.
fn ray_circle_front(from: Vector2, dir: Vector2, centre: Vector2, radius: f64) -> Option<Vector2> {
    let d = from - centre;
    let b = dir.dot(&d);
    let c = d.norm_squared() - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let t = -b - disc.sqrt();
    if t <= 0.0 {
        return None;
    }
    Some(from + dir * t)
}

/// Project a point onto the arc: force radius `radius` from `centre` and clamp the
/// angle off straight-out (+x) to `±max_angle`.
fn clamp_to_arc(p: Vector2, centre: Vector2, radius: f64, max_angle: f64) -> Vector2 {
    let rel = p - centre;
    let theta = rel.y.atan2(rel.x).clamp(-max_angle, max_angle);
    centre + Vector2::new(radius * theta.cos(), radius * theta.sin())
}

/// Whether to enter Clear: ball present, essentially stopped, firmly inside the
/// box, and the keeper is allowed to play it. No opponent check — the rules keep
/// every other robot out of the defense area, so a ball in the box is ours.
fn should_clear(world: &World) -> bool {
    if !world.is_ball_in_play() {
        return false;
    }
    let Some(ball) = world.ball_position() else {
        return false;
    };
    let speed = world.ball_velocity().map(|v| v.norm()).unwrap_or(0.0);
    speed < config::CLEAR_SPEED_LIMIT && ball_firmly_inside(world, ball, config::CLEAR_INNER_MARGIN)
}

/// Whether an in-progress Clear should continue. Aborts back to Guard if the ball
/// has left the box, the keeper is about to leave the box, or play is interrupted.
fn clear_still_valid(world: &World, keeper: &PlayerHandle) -> bool {
    if !world.is_ball_in_play() {
        return false;
    }
    let Some(ball) = world.ball_position() else {
        return false;
    };
    // Ball gone from the box (with no margin) → done / left.
    if !world.own_penalty_area().contains(ball) {
        return false;
    }
    // Safety: never let the keeper leave the box.
    keeper_safely_inside(world, keeper.position(), config::CLEAR_EXIT_MARGIN)
}

/// Ball is inside the penalty area by `margin` on the field-facing edges (front +
/// sides), so the whole pickup maneuver stays inside. The back edge is the goal
/// line — no margin needed there.
fn ball_firmly_inside(world: &World, ball: Vector2, margin: f64) -> bool {
    let area = world.own_penalty_area();
    // Side edges use the full `margin`; the front (field-facing) edge uses a much
    // smaller margin so the keeper claims balls on the front line that no field
    // robot can legally reach. The back edge is the goal line — no margin there.
    ball.x >= area.min.x
        && ball.x <= area.max.x - config::CLEAR_FRONT_MARGIN
        && ball.y >= area.min.y + margin
        && ball.y <= area.max.y - margin
}

/// Keeper is comfortably inside the box: at least `margin` from the field-facing
/// edges (front + sides). The goal-line edge is not constrained.
fn keeper_safely_inside(world: &World, pos: Vector2, margin: f64) -> bool {
    let area = world.own_penalty_area();
    pos.x <= area.max.x - margin && pos.y >= area.min.y + margin && pos.y <= area.max.y - margin
}

/// Clearing target: upfield (toward the opponent half) and pushed toward the
/// touchline on the ball's side, never straight up the middle in front of goal.
fn clear_target(world: &World, ball: Vector2) -> Vector2 {
    let side = if ball.y >= 0.0 { 1.0 } else { -1.0 };
    let max_y = (world.field_width() / 2.0 - 200.0).max(config::CLEAR_TARGET_MIN_Y);
    let y = side * config::CLEAR_TARGET_MIN_Y.min(max_y);
    Vector2::new(config::CLEAR_TARGET_X, y)
}

/// Heading from `from` toward `to`.
fn facing_angle(from: Vector2, to: Vector2) -> Angle {
    let d = to - from;
    Angle::from_radians(d.y.atan2(d.x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dies_core::FieldGeometry;
    use dies_strategy_api::prelude::{BallState, PlayerState};
    use dies_strategy_protocol::{GameState, PlayerId, Possession, WorldSnapshot};

    fn world_with_ball(ball: Vector2) -> World {
        World::new(WorldSnapshot {
            timestamp: 0.0,
            dt: 0.016,
            field_geom: Some(FieldGeometry::default()),
            ball: Some(BallState {
                position: ball,
                velocity: Vector2::zeros(),
                detected: true,
            }),
            own_players: vec![PlayerState::new(
                PlayerId::new(1),
                Vector2::new(-4000.0, 0.0),
                Vector2::zeros(),
                Angle::from_radians(0.0),
            )],
            opp_players: vec![],
            game_state: GameState::Run,
            us_operating: true,
            our_keeper_id: Some(PlayerId::new(1)),
            freekick_kicker: None,
            possession: Possession::Loose,
            possession_stale: false,
            ball_contest: None,
        })
    }

    const R: f64 = 400.0;
    const MAX_A: f64 = std::f64::consts::FRAC_PI_4;

    #[test]
    fn central_ball_puts_keeper_square_on() {
        let w = world_with_ball(Vector2::new(0.0, 0.0));
        let t = keeper_arc_target(&w, R, MAX_A);
        let g = w.own_goal_center();
        // On the arc, straight out in front of the goal centre.
        assert!(((t - g).norm() - R).abs() < 1.0, "radius: {:?}", t);
        assert!(t.y.abs() < 1.0, "should be centred: {:?}", t);
        assert!(t.x > g.x, "should be in front of the goal");
    }

    #[test]
    fn stays_on_arc_and_within_clamp() {
        let g = Vector2::new(-4500.0, 0.0);
        for &by in &[-2000.0, -800.0, 0.0, 800.0, 2000.0] {
            for &bx in &[-3000.0, -1000.0, 1000.0] {
                let w = world_with_ball(Vector2::new(bx, by));
                let t = keeper_arc_target(&w, R, MAX_A);
                // Exactly on the arc.
                assert!(((t - g).norm() - R).abs() < 1.0, "radius off: {:?}", t);
                // Within the angular clamp.
                let theta = (t - g).y.atan2((t - g).x);
                assert!(theta.abs() <= MAX_A + 1.0e-6, "angle {} > max", theta);
            }
        }
    }

    #[test]
    fn no_ball_falls_back_to_centre() {
        let mut snap = world_with_ball(Vector2::zeros()).raw_snapshot().clone();
        snap.ball = None;
        let w = World::new(snap);
        let t = keeper_arc_target(&w, R, MAX_A);
        let g = w.own_goal_center();
        assert!(t.y.abs() < 1.0 && (t.x - (g.x + R)).abs() < 1.0, "{:?}", t);
    }

    #[test]
    fn ball_on_a_side_pulls_keeper_toward_that_side() {
        let w = world_with_ball(Vector2::new(0.0, 1500.0));
        let t = keeper_arc_target(&w, R, MAX_A);
        assert!(t.y > 0.0, "keeper should shift toward +y: {:?}", t);
    }

    #[test]
    fn clear_only_when_stopped_and_firmly_inside() {
        // Firmly inside, stopped → clear.
        let inside = Vector2::new(-4000.0, 0.0);
        assert!(should_clear(&world_with_ball(inside)));
        // Outside the box → no.
        assert!(!should_clear(&world_with_ball(Vector2::new(0.0, 0.0))));
    }

    #[test]
    fn clear_claims_ball_near_the_front_line() {
        // A stopped ball just inside the front edge (the old 250mm margin left a
        // dead zone here where field robots can't reach and the keeper wouldn't
        // commit). The small front margin must now claim it.
        let front_edge = -4500.0 + 1000.0; // own_penalty_area front (default geom)
        let ball = Vector2::new(front_edge - config::CLEAR_FRONT_MARGIN - 10.0, 0.0);
        assert!(should_clear(&world_with_ball(ball)), "ball at {ball:?}");
    }

    #[test]
    fn clear_target_is_off_centre_and_upfield_on_ball_side() {
        let w = world_with_ball(Vector2::new(-4000.0, 300.0));
        let target = clear_target(&w, Vector2::new(-4000.0, 300.0));
        assert!(
            target.y >= config::CLEAR_TARGET_MIN_Y,
            "off centre: {:?}",
            target
        );
        let neg = clear_target(&w, Vector2::new(-4000.0, -300.0));
        assert!(neg.y <= -config::CLEAR_TARGET_MIN_Y, "mirror: {:?}", neg);
    }
}
