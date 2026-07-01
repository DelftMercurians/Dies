//! Goalie positioning benchmark — baseline (old 45° angular clamp) vs the new
//! mouth-clamp + near-post-bias guard model.
//!
//! Run: `cargo run -p concerto --example goalie_bench`
//!
//! Two metrics, both deterministic and isolated from defenders / strategy noise:
//!
//! 1. **Static coverage** (the headline). For a stationary threat ball, place the
//!    keeper at its guard target and measure the largest *open* lane an adversarial
//!    shooter can exploit through the goal mouth — the attacker's best shot. Lower
//!    is better. Reported split into the near-post lane and far-post lane so the
//!    near-post-bias tradeoff is visible. Three variants per scenario:
//!      - `base`    : old model (bisector ∩ arc, clamped to ±45°)
//!      - `clampB`  : new mouth clamp, **no** near-post bias (isolates the clamp fix)
//!      - `new`     : new mouth clamp **+** config near-post bias (production)
//!
//! 2. **Kinematic shot sim** (regression sanity). A handful of scenarios where the
//!    keeper pre-positions, then a shot is struck at the adversarial gap; the keeper
//!    is stepped with the real `GoToBounded` profile (KP 10, 2500 mm/s, 8000 mm/s²)
//!    and confined to its guard zone. Reports SAVE / GOAL for base vs new.
//!
//! Everything is in the keeper's team-relative frame: own goal at -x = (-4500, 0),
//! posts at (-4500, ±500), threats come from the -x defensive half.

use dies_core::{Angle, FieldGeometry, Vector2};
use dies_strategy_api::prelude::{BallState, PlayerState};
use dies_strategy_api::World;
use dies_strategy_protocol::{GameState, PlayerId, Possession, WorldSnapshot};

use concerto::config;
use concerto::keeper::keeper_arc_target;

// ── Physical constants ────────────────────────────────────────────────────────
const BODY: f64 = 90.0; // keeper robot radius (mm)
const BALL: f64 = config::BALL_RADIUS; // 21.5 mm
const BLOCK: f64 = BODY + BALL; // contact radius: keeper covers a shot within this

// Keeper kinematic profile (mirrors GoToBoundedSkill in the executor).
const KP: f64 = 10.0;
const SPEED: f64 = 2500.0;
const ACCEL: f64 = 8000.0;
const DT: f64 = 1.0 / 60.0;

// Baseline angular clamp (the old KEEPER_ARC_MAX_ANGLE) for the frozen reference.
const BASE_MAX_ANGLE: f64 = std::f64::consts::FRAC_PI_4; // 45°

// ── World construction ─────────────────────────────────────────────────────────
fn make_world(ball: Vector2, ball_vel: Vector2) -> World {
    World::new(WorldSnapshot {
        timestamp: 0.0,
        dt: DT,
        field_geom: Some(FieldGeometry::default()),
        ball: Some(BallState {
            position: ball,
            velocity: ball_vel,
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
        us_operating: false,
        pre_stage: false,
        our_keeper_id: Some(PlayerId::new(1)),
        freekick_kicker: None,
        possession: Possession::Loose,
        possession_stale: false,
        ball_contest: None,
    })
}

// ── Frozen baseline keeper geometry (copy of the pre-change keeper.rs) ──────────
fn base_target(world: &World, radius: f64, max_angle: f64) -> Vector2 {
    let g = world.own_goal_center();
    let half_goal = world.goal_width() / 2.0;
    let ball = match world.ball_position() {
        Some(b) => b,
        None => return clamp_to_arc(g + Vector2::new(radius, 0.0), g, radius, max_angle),
    };
    if let Some(p) = base_shot_intercept(world, g, half_goal, radius, max_angle) {
        return p;
    }
    let post_l = g + Vector2::new(0.0, half_goal);
    let post_r = g + Vector2::new(0.0, -half_goal);
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
            let d = (ball - g)
                .try_normalize(1.0e-6)
                .unwrap_or(Vector2::new(1.0, 0.0));
            g + d * radius
        });
    clamp_to_arc(candidate, g, radius, max_angle)
}

fn base_shot_intercept(
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
    if dir.x >= 0.0 {
        return None;
    }
    let t_goal = (g.x - ball.x) / dir.x;
    if t_goal <= 0.0 {
        return None;
    }
    let y_cross = ball.y + dir.y * t_goal;
    if y_cross.abs() > half_goal + config::KEEPER_INTERCEPT_MOUTH_MARGIN {
        return None;
    }
    let depth_x = g.x + radius;
    let t_depth = (depth_x - ball.x) / dir.x;
    if t_depth <= 0.0 {
        return None;
    }
    Some(clamp_to_arc(ball + dir * t_depth, g, radius, max_angle))
}

fn clamp_to_arc(p: Vector2, centre: Vector2, radius: f64, max_angle: f64) -> Vector2 {
    let rel = p - centre;
    let theta = rel.y.atan2(rel.x).clamp(-max_angle, max_angle);
    centre + Vector2::new(radius * theta.cos(), radius * theta.sin())
}

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

// ── Coverage metric ────────────────────────────────────────────────────────────
/// Perpendicular distance from point `p` to segment `a→b`.
fn point_seg_dist(p: Vector2, a: Vector2, b: Vector2) -> f64 {
    let ab = b - a;
    let len2 = ab.norm_squared();
    if len2 < 1.0e-9 {
        return (p - a).norm();
    }
    let t = ((p - a).dot(&ab) / len2).clamp(0.0, 1.0);
    (p - (a + ab * t)).norm()
}

/// Open lanes through the goal mouth that an adversarial shot from `ball` can use
/// past a keeper at `keeper`. Returns `(largest_gap, near_gap, far_gap)` in mm,
/// where near/far are relative to the ball's side. A lane at goal-line `y` is open
/// if the shot segment `ball→(goal_x, y)` clears the keeper disc (`BLOCK`).
fn coverage(ball: Vector2, keeper: Vector2, goal_x: f64, half_goal: f64) -> (f64, f64, f64) {
    let y_lo = -half_goal + BALL;
    let y_hi = half_goal - BALL;
    let step = 1.0;
    let near_sign = if ball.y >= 0.0 { 1.0 } else { -1.0 };

    let mut runs: Vec<(f64, f64)> = Vec::new(); // (start_y, end_y) of open runs
    let mut run_start: Option<f64> = None;
    let mut y = y_lo;
    while y <= y_hi + 1.0e-9 {
        let target = Vector2::new(goal_x, y);
        let open = point_seg_dist(keeper, ball, target) > BLOCK;
        match (open, run_start) {
            (true, None) => run_start = Some(y),
            (false, Some(s)) => {
                runs.push((s, y - step));
                run_start = None;
            }
            _ => {}
        }
        y += step;
    }
    if let Some(s) = run_start {
        runs.push((s, y_hi));
    }

    let mut largest = 0.0_f64;
    let mut near = 0.0_f64;
    let mut far = 0.0_f64;
    for (s, e) in runs {
        let w = e - s;
        let mid = 0.5 * (s + e);
        largest = largest.max(w);
        if mid * near_sign >= 0.0 {
            near = near.max(w);
        } else {
            far = far.max(w);
        }
    }
    (largest, near, far)
}

// ── Scenarios ──────────────────────────────────────────────────────────────────
struct Scn {
    name: &'static str,
    ball: Vector2,
}

fn static_scenarios() -> Vec<Scn> {
    vec![
        Scn {
            name: "corner-freekick (the incident)",
            ball: Vector2::new(-4000.0, -2000.0),
        },
        Scn {
            name: "deep-corner (tight)",
            ball: Vector2::new(-4200.0, -2200.0),
        },
        Scn {
            name: "wide box-edge",
            ball: Vector2::new(-3500.0, -1500.0),
        },
        Scn {
            name: "wing mid-range",
            ball: Vector2::new(-2500.0, -1800.0),
        },
        Scn {
            name: "half-wide in penalty",
            ball: Vector2::new(-3500.0, -800.0),
        },
        Scn {
            name: "slight off-centre",
            ball: Vector2::new(-3000.0, -600.0),
        },
        Scn {
            name: "central top-of-box",
            ball: Vector2::new(-3000.0, 0.0),
        },
        Scn {
            name: "central distance",
            ball: Vector2::new(-1500.0, 0.0),
        },
        Scn {
            name: "near goal-line angle",
            ball: Vector2::new(-4350.0, -1400.0),
        },
        Scn {
            name: "mirror +y corner",
            ball: Vector2::new(-4000.0, 2000.0),
        },
        Scn {
            name: "mirror +y wide",
            ball: Vector2::new(-3500.0, 1500.0),
        },
    ]
}

fn new_target(world: &World, bias: f64) -> Vector2 {
    keeper_arc_target(
        world,
        config::KEEPER_ARC_RADIUS,
        config::KEEPER_ARC_MAX_ANGLE,
        config::KEEPER_MOUTH_MARGIN,
        bias,
    )
}

fn run_static() -> (f64, f64, f64, usize) {
    let geom = FieldGeometry::default();
    let half_goal = geom.goal_width / 2.0;
    let goal_x = -geom.field_length / 2.0;

    println!("\n=== STATIC COVERAGE — largest open lane an adversarial shooter can use (mm; lower better) ===");
    println!(
        "{:<32} {:>12} {:>22} {:>22} {:>22}",
        "scenario",
        "ball(x,y)",
        "base  [largest|near|far]",
        "clampB[largest|near|far]",
        "new   [largest|near|far]"
    );

    let (mut sum_base, mut sum_clamp, mut sum_new) = (0.0, 0.0, 0.0);
    let mut improved = 0usize;
    let scns = static_scenarios();
    for s in &scns {
        let w = make_world(s.ball, Vector2::zeros());
        let kb = base_target(&w, config::KEEPER_ARC_RADIUS, BASE_MAX_ANGLE);
        let kc = new_target(&w, 0.0);
        let kn = new_target(&w, config::KEEPER_NEARPOST_BIAS);

        let (lb, nb, fb) = coverage(s.ball, kb, goal_x, half_goal);
        let (lc, nc, fc) = coverage(s.ball, kc, goal_x, half_goal);
        let (ln, nn, fnf) = coverage(s.ball, kn, goal_x, half_goal);

        sum_base += lb;
        sum_clamp += lc;
        sum_new += ln;
        if ln <= lb + 1.0 {
            improved += 1;
        }

        println!(
            "{:<32} {:>5.0},{:>5.0} {:>7.0}|{:>5.0}|{:>5.0}   {:>7.0}|{:>5.0}|{:>5.0}   {:>7.0}|{:>5.0}|{:>5.0}",
            s.name, s.ball.x, s.ball.y, lb, nb, fb, lc, nc, fc, ln, nn, fnf
        );
    }
    let n = scns.len();
    println!(
        "{:<32} {:>12} {:>22.0} {:>22.0} {:>22.0}",
        "TOTAL largest-gap", "", sum_base, sum_clamp, sum_new
    );
    (sum_base, sum_clamp, sum_new, improved)
}

/// The actual shot that scored in the incident: a strike crossing the goal line at
/// y ≈ -241 (team-relative). Is it blocked by each pre-positioned keeper?
fn run_incident_shot() {
    let geom = FieldGeometry::default();
    let half_goal = geom.goal_width / 2.0;
    let goal_x = -geom.field_length / 2.0;
    let ball = Vector2::new(-4000.0, -2000.0);
    let cross = Vector2::new(goal_x, -241.0);

    let w = make_world(ball, Vector2::zeros());
    let kb = base_target(&w, config::KEEPER_ARC_RADIUS, BASE_MAX_ANGLE);
    let kn = new_target(&w, config::KEEPER_NEARPOST_BIAS);
    let db = point_seg_dist(kb, ball, cross);
    let dn = point_seg_dist(kn, ball, cross);

    println!("\n=== INCIDENT SHOT — strike from (-4000,-2000) crossing the line at y=-241 ===");
    println!("  (a save needs the keeper within {BLOCK:.0} mm of the shot line; half_goal={half_goal:.0})");
    println!(
        "  base : keeper ({:>6.0},{:>6.0})  dist-to-shot {:>6.0} mm  -> {}",
        kb.x,
        kb.y,
        db,
        if db < BLOCK { "SAVE" } else { "GOAL" }
    );
    println!(
        "  new  : keeper ({:>6.0},{:>6.0})  dist-to-shot {:>6.0} mm  -> {}",
        kn.x,
        kn.y,
        dn,
        if dn < BLOCK { "SAVE" } else { "GOAL" }
    );
}

// ── Kinematic shot sim ─────────────────────────────────────────────────────────
struct Zone {
    radius: f64,
    max_angle: f64,
}

fn step_keeper(
    pos: Vector2,
    vel: Vector2,
    target: Vector2,
    g: Vector2,
    zone: &Zone,
) -> (Vector2, Vector2) {
    let to = target - pos;
    let d = to.norm();
    let desired = if d > 1.0e-6 {
        (to / d) * (KP * d).min(SPEED)
    } else {
        Vector2::zeros()
    };
    let mut dv = desired - vel;
    let max_dv = ACCEL * DT;
    if dv.norm() > max_dv {
        dv = dv.normalize() * max_dv;
    }
    let new_vel = vel + dv;
    let mut new_pos = pos + new_vel * DT;
    // Confine to the guard zone (mirrors the no-overshoot envelope).
    let rel = new_pos - g;
    let r = rel
        .norm()
        .min(zone.radius + config::KEEPER_ZONE_RADIUS_SLACK);
    let theta = rel.y.atan2(rel.x).clamp(-(zone.max_angle), zone.max_angle);
    new_pos = g + Vector2::new(r * theta.cos(), r * theta.sin());
    (new_pos, new_vel)
}

struct DynScn {
    name: &'static str,
    settle_ball: Vector2,
    /// If `Some`, the ball is already moving (a redirect) and struck at this vel.
    moving: Option<Vector2>,
    shot_speed: f64,
    settle_secs: f64,
}

/// Returns (outcome, min_dist).
fn simulate(scn: &DynScn, is_new: bool) -> (&'static str, f64) {
    let geom = FieldGeometry::default();
    let half_goal = geom.goal_width / 2.0;
    let goal_x = -geom.field_length / 2.0;
    let g = Vector2::new(goal_x, 0.0);

    let zone = if is_new {
        Zone {
            radius: config::KEEPER_ARC_RADIUS,
            max_angle: config::KEEPER_ARC_MAX_ANGLE + config::KEEPER_ZONE_ANGLE_SLACK,
        }
    } else {
        Zone {
            radius: config::KEEPER_ARC_RADIUS,
            max_angle: BASE_MAX_ANGLE,
        }
    };
    let target_of = |w: &World| -> Vector2 {
        if is_new {
            new_target(w, config::KEEPER_NEARPOST_BIAS)
        } else {
            base_target(w, config::KEEPER_ARC_RADIUS, BASE_MAX_ANGLE)
        }
    };

    // Start the keeper at its guard target for the settle ball.
    let w0 = make_world(scn.settle_ball, Vector2::zeros());
    let mut kp = target_of(&w0);
    let mut kv = Vector2::zeros();

    // Settle phase: ball stationary, keeper pre-positions.
    let settle_ticks = (scn.settle_secs / DT) as usize;
    for _ in 0..settle_ticks {
        let w = make_world(scn.settle_ball, Vector2::zeros());
        let t = target_of(&w);
        let (p, v) = step_keeper(kp, kv, t, g, &zone);
        kp = p;
        kv = v;
    }

    // Release the shot.
    let (mut ball, mut bvel) = match scn.moving {
        Some(v) => (scn.settle_ball, v),
        None => {
            // Adversarial: aim at the centre of the largest open lane.
            let aim_y = best_gap_center(scn.settle_ball, kp, goal_x, half_goal);
            let dir = (Vector2::new(goal_x, aim_y) - scn.settle_ball).normalize();
            (scn.settle_ball, dir * scn.shot_speed)
        }
    };

    let mut min_dist = f64::INFINITY;
    let max_ticks = (1.5 / DT) as usize;
    for _ in 0..max_ticks {
        let w = make_world(ball, bvel);
        let t = target_of(&w);
        let (p, v) = step_keeper(kp, kv, t, g, &zone);
        kp = p;
        kv = v;
        ball += bvel * DT;
        let d = (kp - ball).norm();
        min_dist = min_dist.min(d);
        if d < BLOCK {
            return ("SAVE", min_dist);
        }
        if ball.x <= goal_x {
            if ball.y.abs() < half_goal {
                return ("GOAL", min_dist);
            }
            return ("WIDE", min_dist);
        }
    }
    ("OUT", min_dist)
}

fn best_gap_center(ball: Vector2, keeper: Vector2, goal_x: f64, half_goal: f64) -> f64 {
    let y_lo = -half_goal + BALL;
    let y_hi = half_goal - BALL;
    let step = 1.0;
    let mut best_w = -1.0;
    let mut best_c = 0.0;
    let mut run_start: Option<f64> = None;
    let mut y = y_lo;
    while y <= y_hi + 1.0e-9 {
        let open = point_seg_dist(keeper, ball, Vector2::new(goal_x, y)) > BLOCK;
        match (open, run_start) {
            (true, None) => run_start = Some(y),
            (false, Some(s)) => {
                let w = (y - step) - s;
                if w > best_w {
                    best_w = w;
                    best_c = 0.5 * (s + y - step);
                }
                run_start = None;
            }
            _ => {}
        }
        y += step;
    }
    if let Some(s) = run_start {
        let w = y_hi - s;
        if w > best_w {
            best_c = 0.5 * (s + y_hi);
        }
    }
    best_c
}

fn run_dynamic() {
    let scns = vec![
        DynScn {
            name: "incident free-kick (settle 1.5s, strike 4000)",
            settle_ball: Vector2::new(-4000.0, -2000.0),
            moving: None,
            shot_speed: 4000.0,
            settle_secs: 1.5,
        },
        DynScn {
            name: "central drive (settle 1.0s, strike 4500)",
            settle_ball: Vector2::new(-2500.0, 0.0),
            moving: None,
            shot_speed: 4500.0,
            settle_secs: 1.0,
        },
        DynScn {
            // On-target redirect: crosses the goal line at (-4500, 300).
            name: "fast on-target redirect (moving 3500)",
            settle_ball: Vector2::new(-4100.0, -1600.0),
            moving: Some(
                (Vector2::new(-4500.0, 300.0) - Vector2::new(-4100.0, -1600.0)).normalize()
                    * 3500.0,
            ),
            shot_speed: 3500.0,
            settle_secs: 0.0,
        },
    ];
    println!("\n=== KINEMATIC SHOT SIM (regression sanity; real GoToBounded profile) ===");
    println!("{:<48} {:>14} {:>14}", "scenario", "base", "new");
    for s in &scns {
        let (ob, db) = simulate(s, false);
        let (on, dn) = simulate(s, true);
        println!(
            "{:<48} {:>6} ({:>4.0}) {:>6} ({:>4.0})",
            s.name, ob, db, on, dn
        );
    }
}

/// Aggregate largest-gap across the static battery for a given near-post bias —
/// plus, separately, the worst-case regression vs baseline on any single scenario.
fn run_bias_sweep() {
    let geom = FieldGeometry::default();
    let half_goal = geom.goal_width / 2.0;
    let goal_x = -geom.field_length / 2.0;
    let scns = static_scenarios();

    // Baseline largest-gap per scenario, for the regression column.
    let base: Vec<f64> = scns
        .iter()
        .map(|s| {
            let w = make_world(s.ball, Vector2::zeros());
            let k = base_target(&w, config::KEEPER_ARC_RADIUS, BASE_MAX_ANGLE);
            coverage(s.ball, k, goal_x, half_goal).0
        })
        .collect();

    println!("\n=== NEAR-POST BIAS SWEEP (static battery; largest-gap, lower better) ===");
    println!(
        "{:>6} {:>14} {:>22}",
        "bias", "total largest", "worst single regression"
    );
    for &bias in &[0.0, 0.05, 0.10, 0.15, 0.20, 0.25] {
        let mut total = 0.0;
        let mut worst = 0.0_f64; // most a single scenario got worse than baseline
        for (i, s) in scns.iter().enumerate() {
            let w = make_world(s.ball, Vector2::zeros());
            let k = new_target(&w, bias);
            let g = coverage(s.ball, k, goal_x, half_goal).0;
            total += g;
            worst = worst.max(g - base[i]);
        }
        println!("{bias:>6.2} {total:>14.0} {worst:>22.0}");
    }
}

fn main() {
    println!("Goalie positioning benchmark — baseline vs mouth-clamp + near-post bias");
    println!(
        "config: radius={:.0}  max_angle={:.0}°  mouth_margin={:.0}  near_post_bias={:.2}",
        config::KEEPER_ARC_RADIUS,
        config::KEEPER_ARC_MAX_ANGLE.to_degrees(),
        config::KEEPER_MOUTH_MARGIN,
        config::KEEPER_NEARPOST_BIAS,
    );

    let (base, clampb, new, improved) = run_static();
    run_bias_sweep();
    run_incident_shot();
    run_dynamic();

    let n = static_scenarios().len();
    println!("\n=== SUMMARY ===");
    println!(
        "  total largest-gap (mm):  base {base:.0}  ->  clamp-only {clampb:.0}  ->  new {new:.0}"
    );
    println!(
        "  reduction vs base:       clamp-only {:.1}%   new {:.1}%",
        100.0 * (base - clampb) / base,
        100.0 * (base - new) / base
    );
    println!("  scenarios where new <= base: {improved}/{n}");
}
