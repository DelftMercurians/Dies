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
        half_angle: config::KEEPER_ARC_MAX_ANGLE + config::KEEPER_ZONE_ANGLE_SLACK,
    })
}

/// Draw the keeper's guard arc — the arc of radius [`KEEPER_ARC_RADIUS`] around
/// the goal centre, spanning the effective angular clamp (the tighter of
/// [`KEEPER_ARC_MAX_ANGLE`] and the stay-inside-the-mouth limit) — as a debug
/// shape. There is no arc primitive in the debug protocol, so it is drawn as a
/// short polyline of line segments.
///
/// [`KEEPER_ARC_RADIUS`]: config::KEEPER_ARC_RADIUS
/// [`KEEPER_ARC_MAX_ANGLE`]: config::KEEPER_ARC_MAX_ANGLE
fn draw_guard_arc(world: &World) {
    let g = world.own_goal_center();
    let radius = config::KEEPER_ARC_RADIUS;
    let half_goal = world.goal_width() / 2.0;
    let y_lim = (half_goal - config::KEEPER_MOUTH_MARGIN).max(0.0);
    // Same effective angular clamp as `clamp_to_mouth`.
    let lat_angle = (y_lim / radius).clamp(0.0, 1.0).asin();
    let lim = config::KEEPER_ARC_MAX_ANGLE.min(lat_angle);

    const SEGMENTS: usize = 24;
    let point = |theta: f64| g + Vector2::new(radius * theta.cos(), radius * theta.sin());
    let mut prev = point(-lim);
    for i in 1..=SEGMENTS {
        let theta = -lim + (2.0 * lim) * (i as f64 / SEGMENTS as f64);
        let next = point(theta);
        debug::line_colored(&format!("keeper_arc.{i}"), prev, next, DebugColor::Cyan);
        prev = next;
    }
}

/// Drive the keeper for this frame: set its role and issue the Guard/Clear skill
/// commands.
pub fn update(state: &mut KeeperState, world: &World, keeper: &mut PlayerHandle) {
    keeper.set_role("goalkeeper");
    draw_guard_arc(world);

    // ── Penalty defense (rules §5.3.6) ──────────────────────────────────
    // Until the penalty kick is taken, the keeper must stand *on the goal line*
    // between the posts — not on its guard arc. `Penalty` flips to `PenaltyRun`
    // on ball movement (the kick), so gating on PreparePenalty|Penalty releases
    // the keeper into Guard (and its shot-line intercept) the moment the shot
    // is live.
    if penalty_line_required(world) {
        state.mode = KeeperMode::Guard; // resume in Guard once the kick is taken
        let face = world
            .ball_position()
            .unwrap_or_else(|| world.opp_goal_center());
        keeper
            .go_to_bounded(world.own_goal_center(), keeper_guard_zone(world))
            .facing(face);
        return;
    }

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
                config::KEEPER_MOUTH_MARGIN,
                config::KEEPER_NEARPOST_BIAS,
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
            let _ = keeper.strike_through(target);
            // if keeper.has_ball() {
            // } else {
            //     // Acquire via the unified ball-handling skill (Hold mode); the
            //     // reflex_shoot above takes over once the breakbeam trips.
            //     let _ = keeper.handle_ball(
            //         BallAction::Hold { heading },
            //         AcquirePosition::Heading(heading),
            //     );
            // }
        }
    }
}

/// Compute the keeper's Guard target: the point on the arc of radius `radius`
/// around the goal centre that lies on the shot-cone bisector biased toward the
/// near post, kept laterally within the goal mouth (`mouth_margin` inside each
/// post) and within `±max_angle` off straight-out.
///
/// The near-post bias (`nearpost_bias`, scaled by how oblique the ball is) shifts
/// the keeper toward the post on the ball's side: for a corner shot the makeable
/// shot is at the near post, while the far post is geometrically very hard, so we
/// deny the near post and concede the far. The mouth clamp (Option B) replaces the
/// old fixed angular clamp, which pinned the keeper near goal-centre and left the
/// near post open against oblique shots.
pub fn keeper_arc_target(
    world: &World,
    radius: f64,
    max_angle: f64,
    mouth_margin: f64,
    nearpost_bias: f64,
) -> Vector2 {
    let g = world.own_goal_center();
    let half_goal = world.goal_width() / 2.0;
    // Lateral limit for the keeper centre: stay this far inside the posts.
    let y_lim = (half_goal - mouth_margin).max(0.0);

    let ball = match world.ball_position() {
        Some(b) => b,
        // No ball: square up at the centre of the arc.
        None => return clamp_to_mouth(g + Vector2::new(radius, 0.0), g, radius, max_angle, y_lim),
    };

    // Fast incoming shot on target: stand on the shot line where it reaches the
    // keeper's depth, rather than the cone bisector.
    if let Some(p) = shot_intercept(world, g, half_goal, radius, max_angle, y_lim) {
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

    // Near-post bias, scaled by obliqueness: `obliq` is the lateral component of
    // the goal→ball direction (0 = straight on, 1 = level with the goal line). The
    // near post is the one on the ball's side.
    let aim = bisector.map(|bis| {
        let to_ball = (ball - g)
            .try_normalize(1.0e-6)
            .unwrap_or(Vector2::new(1.0, 0.0));
        let obliq = to_ball.y.abs().min(1.0);
        let near_post = g + Vector2::new(0.0, half_goal * to_ball.y.signum());
        match (near_post - ball).try_normalize(1.0e-6) {
            Some(to_near) => {
                let w = (nearpost_bias * obliq).clamp(0.0, 1.0);
                (bis * (1.0 - w) + to_near * w)
                    .try_normalize(1.0e-6)
                    .unwrap_or(bis)
            }
            None => bis,
        }
    });

    let candidate = aim
        .and_then(|u| ray_circle_front(ball, u, g, radius))
        .unwrap_or_else(|| {
            // Degenerate: fall back to the goal→ball direction on the arc.
            let d = (ball - g)
                .try_normalize(1.0e-6)
                .unwrap_or(Vector2::new(1.0, 0.0));
            g + d * radius
        });

    // Cross-shot damping: a ball crossing the mouth fast (not a goalward shot —
    // the intercept above already handled those) is a cross/switch. Pull the guard
    // angle back toward square-on so the keeper stays compact and isn't wrong-footed
    // by a strike back across the face. Keeps it more central (never off its line).
    let damped = damp_cross(candidate, world, g, radius);

    clamp_to_mouth(damped, g, radius, max_angle, y_lim)
}

/// Pull a guard candidate's angle back toward square-on (straight out from goal)
/// proportionally to how fast the ball is crossing the mouth laterally, when the
/// ball is fast and deep in our half. Returns `candidate` unchanged for a slow or
/// up-field ball. The radius is preserved (we only reduce the angular excursion),
/// so the keeper stays on its arc — this only ever moves it *toward* centre, never
/// further out or off the line.
fn damp_cross(candidate: Vector2, world: &World, g: Vector2, radius: f64) -> Vector2 {
    let Some(ball) = world.ball_position() else {
        return candidate;
    };
    if ball.x > config::KEEPER_CROSS_DAMP_BALL_X {
        return candidate;
    }
    let lateral_speed = world.ball_velocity().map(|v| v.y.abs()).unwrap_or(0.0);
    let w = config::KEEPER_CROSS_DAMP_MAX
        * crate::geometry::smoothstep(
            config::KEEPER_CROSS_DAMP_SPEED_LO,
            config::KEEPER_CROSS_DAMP_SPEED_HI,
            lateral_speed,
        );
    if w <= 1.0e-6 {
        return candidate;
    }
    let rel = candidate - g;
    let theta = rel.y.atan2(rel.x) * (1.0 - w);
    g + Vector2::new(radius * theta.cos(), radius * theta.sin())
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
    y_lim: f64,
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
    Some(clamp_to_mouth(
        ball + dir * t_depth,
        g,
        radius,
        max_angle,
        y_lim,
    ))
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

/// Project a point onto the keeper's guard arc, kept in front of the goal and
/// inside the mouth: force radius `radius` from `centre`, then clamp the angle off
/// straight-out (+x, team-relative) to whichever is tighter — `±max_angle` (the
/// stay-in-front sanity bound) or the angle whose lateral offset equals `y_lim`
/// (the stay-within-the-mouth clamp, Option B).
fn clamp_to_mouth(p: Vector2, centre: Vector2, radius: f64, max_angle: f64, y_lim: f64) -> Vector2 {
    let rel = p - centre;
    // Tightest angular bound: the explicit sanity cap, or the lateral mouth limit
    // (the angle at which `radius·sin θ == y_lim`).
    let lat_angle = (y_lim / radius).clamp(0.0, 1.0).asin();
    let lim = max_angle.min(lat_angle);
    let theta = rel.y.atan2(rel.x).clamp(-lim, lim);
    centre + Vector2::new(radius * theta.cos(), radius * theta.sin())
}

/// Whether the keeper must hold the goal line for a penalty against us: from the
/// prepare signal until the kick is taken. `GameState::Penalty` transitions to
/// `PenaltyRun` on ball movement, so these two states are exactly the
/// before-the-kick window.
fn penalty_line_required(world: &World) -> bool {
    !world.us_operating()
        && matches!(
            world.game_state(),
            GameState::PreparePenalty | GameState::Penalty
        )
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

/// Clearing target: the opponent goal mouth on the ball's flank side. A clear is
/// a full-power strike that rolls the length of the field, so under the Div-B
/// aimless-kick rule it must be goal-bound like the hoof — a wing clear that
/// crosses halfway and rolls out wide of their goal hands the opponent a free
/// kick back at OUR kick position. Inside the mouth an untouched clear scores or
/// forces their keeper to play it.
fn clear_target(world: &World, ball: Vector2) -> Vector2 {
    let side = if ball.y >= 0.0 { 1.0 } else { -1.0 };
    Vector2::new(world.opp_goal_center().x, side * config::CLEAR_AIM_Y)
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
            pre_stage: false,
            our_keeper_id: Some(PlayerId::new(1)),
            freekick_kicker: None,
            double_touch_barred: None,
            possession: Possession::Loose,
            possession_stale: false,
            ball_contest: None,
        })
    }

    const R: f64 = config::KEEPER_ARC_RADIUS;
    const MAX_A: f64 = config::KEEPER_ARC_MAX_ANGLE;
    const MARGIN: f64 = config::KEEPER_MOUTH_MARGIN;
    const BIAS: f64 = config::KEEPER_NEARPOST_BIAS;

    /// Guard target with the production constants.
    fn guard_target(w: &World) -> Vector2 {
        keeper_arc_target(w, R, MAX_A, MARGIN, BIAS)
    }

    /// Distance from `p` to segment `a→b` (a shot path).
    fn seg_dist(p: Vector2, a: Vector2, b: Vector2) -> f64 {
        let ab = b - a;
        let t = ((p - a).dot(&ab) / ab.norm_squared()).clamp(0.0, 1.0);
        (p - (a + ab * t)).norm()
    }

    /// Regression guard for the 2026-06-28 corner free-kick goal: a strike from the
    /// corner (-4000,-2000) crossing the line at y=-241. The keeper body (≈90mm)
    /// plus ball radius must be within reach of the shot line. The old ±45° clamp
    /// parked the keeper near goal-centre and left this wide open; the new mouth
    /// clamp covers it.
    #[test]
    fn oblique_corner_shot_is_covered() {
        let ball = Vector2::new(-4000.0, -2000.0);
        let w = world_with_ball(ball);
        let cross = Vector2::new(-4500.0, -241.0);
        let contact = 90.0 + config::BALL_RADIUS;

        let kn = guard_target(&w);
        assert!(
            seg_dist(kn, ball, cross) < contact,
            "new keeper must block the incident shot: keeper {:?} dist {:.0}",
            kn,
            seg_dist(kn, ball, cross)
        );

        // The old behaviour (±45° clamp, no bias) left it open — documents the fix.
        let kb = keeper_arc_target(&w, R, std::f64::consts::FRAC_PI_4, MARGIN, 0.0);
        assert!(
            seg_dist(kb, ball, cross) > contact,
            "old clamp should have conceded (regression baseline): keeper {:?} dist {:.0}",
            kb,
            seg_dist(kb, ball, cross)
        );
    }

    /// On an oblique ball the keeper hugs toward the near post — well past the old
    /// 45°/`radius·sin45°≈283mm` lateral cap that pinned it near centre.
    #[test]
    fn reaches_toward_near_post_on_oblique_ball() {
        let kn = guard_target(&world_with_ball(Vector2::new(-4000.0, -2000.0)));
        assert!(kn.y < -340.0, "should hug toward the near post: {:?}", kn);
    }

    #[test]
    fn central_ball_puts_keeper_square_on() {
        let w = world_with_ball(Vector2::new(0.0, 0.0));
        let t = guard_target(&w);
        let g = w.own_goal_center();
        // On the arc, straight out in front of the goal centre.
        assert!(((t - g).norm() - R).abs() < 1.0, "radius: {:?}", t);
        assert!(t.y.abs() < 1.0, "should be centred: {:?}", t);
        assert!(t.x > g.x, "should be in front of the goal");
    }

    #[test]
    fn stays_on_arc_and_within_clamp() {
        let g = Vector2::new(-4500.0, 0.0);
        let half_goal = FieldGeometry::default().goal_width / 2.0;
        let y_lim = half_goal - MARGIN;
        // Tightest lateral angle: whichever of the sanity cap / mouth limit binds.
        let lim = MAX_A.min((y_lim / R).clamp(0.0, 1.0).asin());
        for &by in &[-2000.0, -800.0, 0.0, 800.0, 2000.0] {
            for &bx in &[-3000.0, -1000.0, 1000.0] {
                let w = world_with_ball(Vector2::new(bx, by));
                let t = guard_target(&w);
                // Exactly on the arc.
                assert!(((t - g).norm() - R).abs() < 1.0, "radius off: {:?}", t);
                // Within the angular clamp.
                let theta = (t - g).y.atan2((t - g).x);
                assert!(theta.abs() <= lim + 1.0e-6, "angle {} > lim {}", theta, lim);
                // And never laterally outside the mouth.
                assert!(t.y.abs() <= y_lim + 1.0, "outside mouth: {:?}", t);
            }
        }
    }

    #[test]
    fn no_ball_falls_back_to_centre() {
        let mut snap = world_with_ball(Vector2::zeros()).raw_snapshot().clone();
        snap.ball = None;
        let w = World::new(snap);
        let t = guard_target(&w);
        let g = w.own_goal_center();
        assert!(t.y.abs() < 1.0 && (t.x - (g.x + R)).abs() < 1.0, "{:?}", t);
    }

    #[test]
    fn ball_on_a_side_pulls_keeper_toward_that_side() {
        let w = world_with_ball(Vector2::new(0.0, 1500.0));
        let t = guard_target(&w);
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
        let t = guard_target(&w);
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
        let t = guard_target(&w);
        assert!(t.y.abs() < 1.0, "slow ball should stay square-on: {:?}", t);
    }

    #[test]
    fn fast_ball_moving_away_keeps_the_bisector_position() {
        // Fast ball but travelling toward the opponent goal (+x): no intercept,
        // keeper holds the bisector (square-on for a central ball).
        let w = world_with_ball_vel(Vector2::zeros(), Vector2::new(1500.0, 0.0));
        let t = guard_target(&w);
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
        let t = guard_target(&w);
        // Bisector for a central ball is square-on; intercept would have pushed it
        // hard toward the +y clamp. Assert we did NOT take the intercept.
        assert!(
            t.y.abs() < 1.0,
            "off-target shot should keep bisector: {:?}",
            t
        );
    }

    #[test]
    fn fast_cross_across_the_face_keeps_keeper_compact() {
        // Regression for the worked-ball-across-the-box concession: a ball deep in
        // our half on the +y side, moving fast *laterally* (across the mouth, not
        // goalward), must not drag the keeper all the way to the +y side — a strike
        // back across the face would then beat it into the open -y corner. The
        // cross-damping pulls the guard angle back toward square-on.
        let ball = Vector2::new(-3200.0, 1400.0);
        let crossing = world_with_ball_vel(ball, Vector2::new(-200.0, 3000.0)); // fast across
        let still = world_with_ball(ball); // same position, ball at rest
        let kc = guard_target(&crossing);
        let ks = guard_target(&still);
        // Both shade toward +y, but the fast cross stays markedly more central.
        assert!(kc.y > 0.0 && ks.y > 0.0, "both shade to the ball side");
        assert!(
            kc.y < ks.y - 50.0,
            "fast cross must keep the keeper more central: crossing y={:.0} vs still y={:.0}",
            kc.y,
            ks.y
        );
        // It must stay on its arc (only the angle is reduced, not the radius).
        let g = crossing.own_goal_center();
        assert!(
            ((kc - g).norm() - R).abs() < 1.0,
            "still on the arc: {:?}",
            kc
        );
    }

    #[test]
    fn slow_cross_is_tracked_normally() {
        // A slowly-rolling ball across the face (an opponent dribbling it) is NOT a
        // cross-shot threat, so the keeper tracks it normally (no damping).
        let ball = Vector2::new(-3200.0, 1400.0);
        let slow = world_with_ball_vel(ball, Vector2::new(0.0, 200.0));
        let still = world_with_ball(ball);
        assert!(
            (guard_target(&slow).y - guard_target(&still).y).abs() < 1.0,
            "slow ball must be tracked like a static one"
        );
    }

    #[test]
    fn upfield_cross_does_not_damp() {
        // A fast lateral ball in the attacking half is no immediate shot threat and
        // must not damp (the keeper would otherwise sit needlessly square-on).
        let ball = Vector2::new(1000.0, 1400.0);
        let fast = world_with_ball_vel(ball, Vector2::new(0.0, 3000.0));
        let still = world_with_ball(ball);
        assert!(
            (guard_target(&fast).y - guard_target(&still).y).abs() < 1.0,
            "up-field cross must not be damped"
        );
    }

    #[test]
    fn penalty_defense_holds_the_goal_line() {
        let mut snap = world_with_ball(Vector2::new(-2500.0, 0.0))
            .raw_snapshot()
            .clone();
        // Opponent penalty, before the kick: keeper must be on the line.
        snap.us_operating = false;
        snap.game_state = GameState::PreparePenalty;
        assert!(penalty_line_required(&World::new(snap.clone())));
        snap.game_state = GameState::Penalty;
        assert!(penalty_line_required(&World::new(snap.clone())));
        // Kick taken (ball moved) → PenaltyRun → normal Guard resumes.
        snap.game_state = GameState::PenaltyRun;
        assert!(!penalty_line_required(&World::new(snap.clone())));
        // Our own penalty: the keeper is free to guard normally.
        snap.game_state = GameState::PreparePenalty;
        snap.us_operating = true;
        assert!(!penalty_line_required(&World::new(snap)));
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
    fn clear_target_is_goal_bound_on_ball_side() {
        // The clear must aim inside the opponent goal mouth (off-centre on the
        // ball's flank), so a full-length roll-out can't be a Div-B aimless kick.
        let w = world_with_ball(Vector2::new(-4000.0, 300.0));
        let target = clear_target(&w, Vector2::new(-4000.0, 300.0));
        assert!(
            (target.x - w.opp_goal_center().x).abs() < 1e-6,
            "goal-bound: {target:?}"
        );
        assert!(
            (target.y - config::CLEAR_AIM_Y).abs() < 1e-6
                && target.y.abs() < w.goal_width() / 2.0,
            "inside the mouth on the ball side: {target:?}"
        );
        let neg = clear_target(&w, Vector2::new(-4000.0, -300.0));
        assert!((neg.y + config::CLEAR_AIM_Y).abs() < 1e-6, "mirror: {neg:?}");
    }
}
