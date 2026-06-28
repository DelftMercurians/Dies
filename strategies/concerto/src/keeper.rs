//! Goalkeeper — dedicated positioning + active clearing, outside Formation.
//!
//! Two modes:
//! - **Guard**: sit on a small arc around the goal mouth, on the bisector of the
//!   shot cone (the two tangents from the ball to the posts), tracking the ball.
//!   When a fast ball is incoming on a path that crosses the goal mouth, the
//!   keeper instead sits on the shot line (see [`shot_intercept`]) so it is in
//!   the ball's path whichever corner the shot is aimed at.
//! - **Clear**: when a ball settles inside our defense area, pick it up and kick
//!   it out toward the opponent half (off the centre line), then return to Guard.
//!
//! In Guard the keeper uses the [`go_to_bounded`](PlayerHandle::go_to_bounded)
//! skill: aggressive direct-velocity control with the planner and ORCA bypassed,
//! kept inside the guard arc by a no-overshoot velocity envelope (see
//! [`keeper_guard_zone`]). The aggressive control profile lives in the executor
//! skill, so no per-frame control overrides cross the IPC.

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

/// The guard region the keeper must not leave: an arc band in front of the goal.
/// The `go_to_bounded` skill's velocity envelope brakes the keeper before it
/// crosses any edge (the outer radius, or the ±angular ends near the posts), so
/// even the aggressive profile can never swing it out of position.
pub fn keeper_guard_zone(world: &World) -> MotionBounds {
    MotionBounds::Arc(ArcZone {
        center: world.own_goal_center(),
        min_radius: 0.0,
        max_radius: config::KEEPER_ARC_RADIUS + config::KEEPER_ZONE_RADIUS_SLACK,
        half_angle: config::KEEPER_ARC_MAX_ANGLE,
    })
}

/// Drive the keeper for this frame: set its role and issue the Guard/Clear skill
/// commands.
pub fn update(state: &mut KeeperState, world: &World, keeper: &mut PlayerHandle) {
    keeper.set_role("goalkeeper");

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
            keeper
                .go_to_bounded(target, keeper_guard_zone(world))
                .facing(face);
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

    // Fast incoming shot on target: stand on the shot line where it reaches the
    // keeper's depth, rather than the cone bisector.
    if let Some(p) = shot_intercept(world, g, half_goal, radius, max_angle) {
        return p;
    }

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

/// Shot-line intercept: if the ball is travelling fast (≥
/// [`KEEPER_INTERCEPT_SPEED`](config::KEEPER_INTERCEPT_SPEED)) toward our goal on
/// a path that crosses the goal mouth (within
/// [`KEEPER_INTERCEPT_MOUTH_MARGIN`](config::KEEPER_INTERCEPT_MOUTH_MARGIN) of a
/// post), return the point on the keeper's arc lying on that trajectory — where
/// the shot reaches the keeper's standoff depth in front of goal. This puts the
/// keeper in the path of the shot regardless of which corner it is aimed at,
/// instead of the cone-bisector position. Returns `None` for a slow ball, a ball
/// moving away from goal, or an off-target trajectory, in which case the caller
/// falls back to the bisector.
fn shot_intercept(
    world: &World,
    g: Vector2,
    half_goal: f64,
    radius: f64,
    max_angle: f64,
) -> Option<Vector2> {
    let ball = world.ball_position()?;
    let vel = world.ball_velocity()?;
    let speed = vel.norm();
    if speed < config::KEEPER_INTERCEPT_SPEED {
        return None;
    }
    let dir = vel / speed;
    // Must be heading toward our goal (own goal is at -x).
    if dir.x >= 0.0 {
        return None;
    }
    // Where does the trajectory cross the goal line (x = g.x)? On target only if
    // within the goal mouth plus the post margin.
    let t_goal = (g.x - ball.x) / dir.x;
    if t_goal <= 0.0 {
        return None;
    }
    let y_cross = ball.y + dir.y * t_goal;
    if y_cross.abs() > half_goal + config::KEEPER_INTERCEPT_MOUTH_MARGIN {
        return None;
    }
    // Intercept the shot line at the keeper's standoff depth in front of goal.
    // If the ball is already inside that depth it's a point-blank shot — defer to
    // the bisector / clear logic.
    let depth_x = g.x + radius;
    let t_depth = (depth_x - ball.x) / dir.x;
    if t_depth <= 0.0 {
        return None;
    }
    Some(clamp_to_arc(ball + dir * t_depth, g, radius, max_angle))
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
    speed < config::CLEAR_SPEED_LIMIT && ball_inside_box(world, ball)
}

/// Whether an in-progress Clear should continue. Aborts back to Guard if the ball
/// has left the box, the keeper charges out the front edge, or play is interrupted.
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
    // Safety: the keeper may go right up to the side/goal-line edges to reach a
    // corner ball, but must not charge out the front edge and abandon the goal.
    keeper_inside_front(world, keeper.position(), config::CLEAR_EXIT_MARGIN)
}

/// A stopped ball inside the penalty area is the keeper's to clear — no field
/// robot may enter the box. Require only a small `CLEAR_BALL_MARGIN` inside the
/// field-facing edges (front + sides) so we don't chase a ball straddling the
/// line; the back edge is the goal line (no margin). A large margin would strand
/// balls in the box corners where neither keeper nor field robot can reach them.
fn ball_inside_box(world: &World, ball: Vector2) -> bool {
    let area = world.own_penalty_area();
    let m = config::CLEAR_BALL_MARGIN;
    ball.x >= area.min.x
        && ball.x <= area.max.x - m
        && ball.y >= area.min.y + m
        && ball.y <= area.max.y - m
}

/// Keeper hasn't charged out past the front (field-facing) edge of the box. The
/// side and goal-line edges are unconstrained: reaching a ball in the box corners
/// requires the keeper near them, and the ball-left-box check already ends the
/// clear once the ball is out.
fn keeper_inside_front(world: &World, pos: Vector2, margin: f64) -> bool {
    let area = world.own_penalty_area();
    pos.x <= area.max.x - margin
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
        world_with_ball_vel(ball, Vector2::zeros())
    }

    fn world_with_ball_vel(ball: Vector2, vel: Vector2) -> World {
        World::new(WorldSnapshot {
            timestamp: 0.0,
            dt: 0.016,
            field_geom: Some(FieldGeometry::default()),
            ball: Some(BallState {
                position: ball,
                velocity: vel,
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
    fn fast_corner_shot_pulls_keeper_onto_the_shot_line() {
        // Central ball, but a fast shot aimed at the +y corner. The cone bisector
        // for a central ball is square-on (y≈0); the shot-line intercept must
        // instead shift the keeper toward +y to sit in the ball's path.
        let g = Vector2::new(-4500.0, 0.0);
        let target = g + Vector2::new(0.0, 400.0); // just inside the +y post
        let dir = (target - Vector2::zeros()).normalize();
        let w = world_with_ball_vel(Vector2::zeros(), dir * 1500.0);
        let t = keeper_arc_target(&w, R, MAX_A);
        // On the arc.
        assert!(((t - g).norm() - R).abs() < 1.0, "radius off: {:?}", t);
        // Shifted toward the shot side, unlike the square-on bisector position.
        assert!(
            t.y > 100.0,
            "should sit on the shot line toward +y: {:?}",
            t
        );
    }

    #[test]
    fn slow_ball_keeps_the_bisector_position() {
        // A slow central ball must NOT trigger the intercept — keeper stays square.
        let w = world_with_ball_vel(Vector2::zeros(), Vector2::new(-100.0, 200.0));
        let t = keeper_arc_target(&w, R, MAX_A);
        assert!(t.y.abs() < 1.0, "slow ball should stay square-on: {:?}", t);
    }

    #[test]
    fn fast_ball_moving_away_keeps_the_bisector_position() {
        // Fast ball but travelling toward the opponent goal (+x): no intercept,
        // keeper holds the bisector (square-on for a central ball).
        let w = world_with_ball_vel(Vector2::zeros(), Vector2::new(1500.0, 0.0));
        let t = keeper_arc_target(&w, R, MAX_A);
        assert!(
            t.y.abs() < 1.0,
            "ball moving away should stay square: {:?}",
            t
        );
    }

    #[test]
    fn fast_off_target_shot_keeps_the_bisector_position() {
        // Fast ball heading goalward but well wide of the mouth (crosses the goal
        // line far outside the posts + margin): not a shot on target, so the
        // keeper tracks the bisector rather than chasing the wide trajectory.
        let g = Vector2::new(-4500.0, 0.0);
        // Aim at a point far above the +y post on the goal line.
        let aim = g + Vector2::new(0.0, 2500.0);
        let dir = (aim - Vector2::zeros()).normalize();
        let w = world_with_ball_vel(Vector2::zeros(), dir * 1500.0);
        let t = keeper_arc_target(&w, R, MAX_A);
        // Bisector for a central ball is square-on; intercept would have pushed it
        // hard toward the +y clamp. Assert we did NOT take the intercept.
        assert!(
            t.y.abs() < 1.0,
            "off-target shot should keep bisector: {:?}",
            t
        );
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
        // commit). The small uniform margin must now claim it.
        let front_edge = -4500.0 + 1000.0; // own_penalty_area front (default geom)
        let ball = Vector2::new(front_edge - config::CLEAR_BALL_MARGIN - 10.0, 0.0);
        assert!(should_clear(&world_with_ball(ball)), "ball at {ball:?}");
    }

    #[test]
    fn clear_claims_ball_in_the_box_corner() {
        // Ball deep in the box near a side edge (the original "stuck in the corner"
        // deadlock): field robots are held out, so the keeper must claim it. Old
        // 250mm side margin excluded it; the small uniform margin must not.
        let side_edge = -1000.0; // own_penalty_area side (default geom, width 2000)
        let ball = Vector2::new(-4100.0, side_edge + config::CLEAR_BALL_MARGIN + 10.0);
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
