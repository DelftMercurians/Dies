use dies_strategy_api::prelude::*;

use crate::config;

/// Momentum-aware estimate of the time (seconds) for a robot at `pos` moving with
/// `vel` to reach `target`, given top speed `v_max` and acceleration `a_max`.
///
/// Unlike Euclidean distance, this credits velocity already pointing at the target
/// and penalises velocity that must be reversed or bled off sideways. A robot
/// driving toward a target has genuinely low cost to continue and high cost to
/// turn around — the physical basis for assignment/selection stability (no
/// artificial stay-bonus).
pub fn redirect_time(pos: Vector2, vel: Vector2, target: Vector2, v_max: f64, a_max: f64) -> f64 {
    let d = target - pos;
    let dist = d.norm();
    if dist < 1e-3 {
        return 0.0;
    }
    let dir = d / dist;

    let v_along = vel.dot(&dir); // + toward target, - away
    let t_cruise = dist / v_max;

    let v_against = (-v_along).max(0.0); // wrong-way speed to reverse
    let v_cross = (vel - dir * v_along).norm(); // sideways speed to bleed off
    let t_redirect = (2.0 * v_against + v_cross) / a_max;

    // Small head-start credit for velocity already toward the target.
    let t_credit = v_along.max(0.0).min(v_max) / (2.0 * a_max);

    (t_cruise + t_redirect - t_credit).max(0.0)
}

/// Infer the opponent keeper: the opponent nearest its own goal (the goal we
/// attack). Strategies aren't told the opponent's designated keeper, but the
/// robot parked deepest in front of the mouth is it in practice — and if no
/// opponent is near the goal, the nearest one casts no central shadow anyway, so
/// the shot model degrades gracefully.
pub fn opp_keeper(world: &World) -> Option<PlayerId> {
    let goal = world.opp_goal_center();
    world
        .opp_players()
        .iter()
        .min_by(|a, b| {
            let da = (a.position - goal).norm();
            let db = (b.position - goal).norm();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|p| p.id)
}

/// An aim point in the opponent goal mouth plus the angular width of the open
/// window it sits in (the shot's aiming tolerance, as seen from the ball).
#[derive(Debug, Clone, Copy)]
pub struct ShotAim {
    /// Point on the goal line to shoot at (the open window's midpoint).
    pub target: Vector2,
    /// Angular width of the open window from the ball, in radians. Distance-aware:
    /// the same gap subtends less angle the farther the ball is, so gating on this
    /// scales the required clearance with range for free.
    pub angle: f64,
}

/// Project every opponent between `ball` and the goal line as a circular shadow
/// onto the goal line (similar triangles from the ball) and return the free
/// y-gaps left within the mouth, plus the keeper's projected y. Shared by
/// [`best_shot`] (window picking) and [`hoof_ray_target`] (goal-term for hoof
/// rays). `None` when there is no shot geometry at all (ball at/behind the goal
/// line, or the mouth narrower than the ball).
#[allow(clippy::too_many_arguments)]
fn goal_free_gaps(
    ball: Vector2,
    goal_x: f64,
    goal_width: f64,
    opponents: &[PlayerState],
    keeper_id: Option<PlayerId>,
    robot_radius: f64,
    keeper_radius: f64,
    ball_radius: f64,
) -> Option<(Vec<(f64, f64)>, Option<f64>)> {
    let dx = goal_x - ball.x;
    if dx <= 1e-3 {
        return None; // ball at or behind the goal line — no shot geometry
    }
    let post = goal_width / 2.0;
    let lo = -post + ball_radius;
    let hi = post - ball_radius;
    if hi <= lo {
        return None;
    }

    // Project opponents to blocked y-intervals on the goal line.
    let mut blocked: Vec<(f64, f64)> = Vec::new();
    let mut keeper_y: Option<f64> = None;
    for opp in opponents {
        let ox = opp.position.x - ball.x;
        if ox <= 1e-3 || opp.position.x > goal_x + robot_radius {
            continue; // behind the ball, or behind the goal line
        }
        let is_keeper = Some(opp.id) == keeper_id;
        let r = if is_keeper {
            keeper_radius
        } else {
            robot_radius
        } + ball_radius;
        let scale = dx / ox; // similar triangles: perpendicular extent grows with range
        let yc = ball.y + (opp.position.y - ball.y) * scale;
        let half = r * scale;
        if is_keeper {
            keeper_y = Some(yc);
        }
        let b_lo = (yc - half).max(lo);
        let b_hi = (yc + half).min(hi);
        if b_hi > b_lo {
            blocked.push((b_lo, b_hi));
        }
    }

    // Merge shadows and collect the free gaps within [lo, hi].
    blocked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut free: Vec<(f64, f64)> = Vec::new();
    let mut cursor = lo;
    for (b_lo, b_hi) in blocked {
        if b_lo > cursor {
            free.push((cursor, b_lo));
        }
        cursor = cursor.max(b_hi);
    }
    if cursor < hi {
        free.push((cursor, hi));
    }
    Some((free, keeper_y))
}

/// Find the widest *open* window in the opponent goal mouth as seen from `ball`,
/// and aim at its midpoint — the open-goal model that replaces shooting only at
/// the goal centre.
///
/// Every opponent between the ball and the goal line is projected as a circular
/// shadow onto the goal line (via similar triangles from the ball) and subtracted
/// from the mouth. The aim window must clear the posts and every shadow by the ball
/// radius. The keeper is special-cased: it gets an inflated shadow (it dives/slides
/// to cover during the ball's flight, blocking far more than its static footprint)
/// and the scoring biases the aim toward the corner *furthest* from it, so when
/// both corners are open we shoot where the keeper must travel furthest.
///
/// Returns the chosen aim point and its angular width, or `None` if no gap in the
/// mouth is open at all (fully crowded goal).
#[allow(clippy::too_many_arguments)]
pub fn best_shot(
    ball: Vector2,
    goal_x: f64,
    goal_width: f64,
    opponents: &[PlayerState],
    keeper_id: Option<PlayerId>,
    robot_radius: f64,
    keeper_radius: f64,
    ball_radius: f64,
    keeper_bias: f64,
) -> Option<ShotAim> {
    let dx = goal_x - ball.x;
    let (free, keeper_y) = goal_free_gaps(
        ball,
        goal_x,
        goal_width,
        opponents,
        keeper_id,
        robot_radius,
        keeper_radius,
        ball_radius,
    )?;

    // Pick the window with the best score: open width plus a bias for distance from
    // the keeper's projected position. Aim at its midpoint.
    let mut best: Option<(f64, f64, f64)> = None; // (score, w_lo, w_hi)
    for (w_lo, w_hi) in free {
        let width = w_hi - w_lo;
        if width <= 1e-6 {
            continue;
        }
        let mid = 0.5 * (w_lo + w_hi);
        let keeper_dist = keeper_y.map(|ky| (mid - ky).abs()).unwrap_or(0.0);
        let score = width + keeper_bias * keeper_dist;
        if best.map(|(s, _, _)| score > s).unwrap_or(true) {
            best = Some((score, w_lo, w_hi));
        }
    }

    let (_, w_lo, w_hi) = best?;
    let mid_y = 0.5 * (w_lo + w_hi);
    let ang_lo = (w_lo - ball.y).atan2(dx);
    let ang_hi = (w_hi - ball.y).atan2(dx);
    Some(ShotAim {
        target: Vector2::new(goal_x, mid_y),
        angle: (ang_hi - ang_lo).abs(),
    })
}

/// Effective keeper shot-shadow radius as a function of how far the keeper has
/// advanced off its goal line (mm). On its line → full inflated shadow (it can
/// dive/slide to cover the mouth during the ball's flight); fully committed
/// forward → plain field-robot radius (it can no longer recover to a corner);
/// linear in between. See [`config::KEEPER_CHARGE_NEAR`]/[`config::KEEPER_CHARGE_FAR`].
pub fn keeper_charge_radius(advance: f64) -> f64 {
    let full = config::SHOT_KEEPER_RADIUS;
    let t = ((advance - config::KEEPER_CHARGE_NEAR)
        / (config::KEEPER_CHARGE_FAR - config::KEEPER_CHARGE_NEAR))
        .clamp(0.0, 1.0);
    full + t * (config::SHOT_ROBOT_RADIUS - full)
}

/// Charge-aware shadow radius for the inferred opponent keeper in the shot model.
pub fn keeper_shot_radius(world: &World) -> f64 {
    let full = config::SHOT_KEEPER_RADIUS;
    let goal = world.opp_goal_center();
    let Some(kid) = opp_keeper(world) else {
        return full;
    };
    let Some(keeper) = world.opp_players().iter().find(|p| p.id == kid) else {
        return full;
    };
    // Advancement off the goal line: the goal we attack is at max +x, so a
    // keeper that has charged toward the ball has a *smaller* x than its goal.
    let advance = (goal.x - keeper.position.x).max(0.0);
    keeper_charge_radius(advance)
}

/// THE shot model: [`best_shot`] wired to concerto's config parameters and the
/// charge-aware keeper shadow. The single scoring function shared by the
/// planner's shot gate / finish aim and Formation's supporter placement, so
/// supporters stand exactly where the planner would rate a shot highly.
pub fn best_shot_from(world: &World, from: Vector2) -> Option<ShotAim> {
    best_shot(
        from,
        world.opp_goal_center().x,
        world.goal_width(),
        world.opp_players(),
        opp_keeper(world),
        config::SHOT_ROBOT_RADIUS,
        keeper_shot_radius(world),
        config::BALL_RADIUS,
        config::SHOT_KEEPER_BIAS,
    )
}

/// Pick the most dangerous *open* finishing pocket for the central box-runner.
///
/// The box-runner is the advanced central body the planner cuts the ball back to.
/// Standing it on a single fixed point in front of the box means it sits wherever
/// the formation puts it even when the defence is covering that exact spot — a
/// cutback to a finisher with no shooting window is wasted. Instead, sample a small
/// fan of candidate pockets across the front of the opponent box on the side away
/// from the ball (so the cutback still cuts *across* the keeper) and pick the one
/// with the widest *open* goal-mouth subtense — the keeper-aware shot window from
/// [`best_shot`] — weighted by how uncontested the spot is ([`receiver_clearance`]).
///
/// Net effect: our central finisher drifts into the shooting lane the defence
/// leaves open instead of a static point, turning the cutback the planner already
/// targets into a real shot. Candidates never go closer to goal than `base` (which
/// already sits just outside the penalty area), so it cannot wander into the box
/// keepout. Returns `base` unchanged when no candidate offers a real window, so it
/// degrades gracefully to the old fixed behaviour.
#[allow(clippy::too_many_arguments)]
pub fn best_finishing_pocket(
    base: Vector2,
    ball: Vector2,
    opp_goal_x: f64,
    goal_width: f64,
    opponents: &[PlayerState],
    keeper_id: Option<PlayerId>,
    robot_radius: f64,
    keeper_radius: f64,
    ball_radius: f64,
    keeper_bias: f64,
    half_wid: f64,
    y_offset: f64,
    // Hysteresis: last tick's pocket (if any) and a *multiplicative* incumbency
    // bonus. The candidate nearest `prev` has its score scaled by up to
    // `1 + stickiness`, so the pocket sticks unless a challenger is decisively
    // better — the box-runner analogue of the supporters' target hysteresis, but
    // relative because the pocket score (`angle × clearance`) has no fixed scale.
    // Pass `None` / `0.0` for the stateless behaviour.
    prev: Option<Vector2>,
    stickiness: f64,
    sticky_radius: f64,
) -> Vector2 {
    // Away from the ball so the cutback crosses the keeper; deterministic when the
    // ball is dead-central (matches the box-runner's own default in formation.rs).
    let away = if ball.y > 0.0 { -1.0 } else { 1.0 };
    // Depths: at the box front (`base.x`) and a step back from it — never closer to
    // goal, so we stay clear of the penalty-area keepout.
    let depths = [base.x, base.x - 300.0];
    // Lateral fan on the away side, plus a tucked-in option toward centre.
    let y_mags = [y_offset * 0.4, y_offset, y_offset * 1.7];
    let mut best: Option<(Vector2, f64)> = None;
    for &dx in &depths {
        for &ym in &y_mags {
            let cy = (away * ym).clamp(-half_wid + 400.0, half_wid - 400.0);
            let cand = Vector2::new(dx, cy);
            let angle = best_shot(
                cand,
                opp_goal_x,
                goal_width,
                opponents,
                keeper_id,
                robot_radius,
                keeper_radius,
                ball_radius,
                keeper_bias,
            )
            .map(|s| s.angle)
            .unwrap_or(0.0);
            if angle <= 0.0 {
                continue;
            }
            let mut score = angle * receiver_clearance(cand, opponents);
            // Multiplicative incumbency: a pocket sitting on last tick's choice is
            // boosted so the argmax holds it against a marginally-better neighbour.
            if let Some(p) = prev {
                if stickiness > 0.0 {
                    let closeness = 1.0 - smoothstep(0.0, sticky_radius, (cand - p).norm());
                    score *= 1.0 + stickiness * closeness;
                }
            }
            if best.map(|(_, s)| score > s).unwrap_or(true) {
                best = Some((cand, score));
            }
        }
    }
    best.map(|(p, _)| p).unwrap_or(base)
}

/// Pick an advancement-hoof direction by scoring *rays*, not landing points.
///
/// This scorer needs no travel estimate (the old resting-point scorer depended
/// on a roll-distance model that is always wrong on a real carpet): candidates
/// are restricted to the cone from the ball to the opponent GOAL MOUTH, posts
/// inset by `mouth_inset` as execution margin. The mouth cone — not the full
/// back line — is what makes every outcome benign under the Div-B aimless-kick
/// rule (a ball from our half that crosses halfway and then their goal line
/// outside the goal untouched = opponent free kick back at our kick position):
/// undershoot settles in their half, an untouched full-length ray is a goal,
/// and a blocked ray is a keeper/defender touch (rebound, or a corner for us).
/// A wide back-line exit — the old cone's "benign overshoot" — is exactly the
/// penalized case, so it is no longer a candidate at all.
///
/// Rays are scored goal > our forward > safety > centre:
/// - exiting through an *open* goal-mouth window ([`goal_free_gaps`], keeper
///   shadow inflated) — prefer the ray that can actually score;
/// - passing near one of `mates` (our players in the opponent half — the
///   permanent outlet / supporters), a within-mouth tiebreak for rebounds;
/// - a mild penalty for opponents sitting on the ray (an en-route touch is
///   rule-safe but costs possession midfield);
/// - a small centre-bias tiebreak (all rays exit at the same depth).
///
/// Returns the ray's goal-line exit point to aim at (the driver only uses it
/// for heading — where the ball stops is deliberately not modelled), or `None`
/// when the cone is degenerate (ball essentially on the opponent back line —
/// the strike-zone branch owns that region).
#[allow(clippy::too_many_arguments)]
pub fn hoof_ray_target(
    ball: Vector2,
    opponents: &[PlayerState],
    mates: &[Vector2],
    opp_goal_x: f64,
    goal_width: f64,
    keeper_id: Option<PlayerId>,
    robot_radius: f64,
    keeper_radius: f64,
    ball_radius: f64,
    half_wid: f64,
    goal_w: f64,
    mate_w: f64,
    mate_radius: f64,
    open_penalty: f64,
    ray_corridor: f64,
    center_w: f64,
    mouth_inset: f64,
) -> Option<Vector2> {
    let depth = opp_goal_x - ball.x;
    if depth <= mouth_inset {
        return None; // on/behind their back line — no forward cone left
    }
    // Cone from the ball to the (inset) goal posts. Every ray in it exits the
    // field through the goal mouth: |y| ≤ goal_width/2 − inset.
    let y_lim = (goal_width / 2.0 - mouth_inset).max(1.0);
    let a_lo = (-y_lim - ball.y).atan2(depth);
    let a_hi = (y_lim - ball.y).atan2(depth);

    // Open goal-mouth gaps, computed once (keeper shadow inflated as for a shot).
    let gaps = goal_free_gaps(
        ball,
        opp_goal_x,
        goal_width,
        opponents,
        keeper_id,
        robot_radius,
        keeper_radius,
        ball_radius,
    )
    .map(|(free, _)| free)
    .unwrap_or_default();

    let score_exit = |exit_y: f64| -> f64 {
        let exit = Vector2::new(opp_goal_x, exit_y);
        let dir = (exit - ball).normalize();

        let in_mouth = gaps
            .iter()
            .any(|&(g_lo, g_hi)| exit_y > g_lo && exit_y < g_hi);
        let goal_term = if in_mouth { goal_w } else { 0.0 };

        // Long ball to the target man: nearest-to-ray forward teammate, by
        // perpendicular distance, provided it projects forward along the ray.
        let mate_term = mates
            .iter()
            .filter(|m| (*m - ball).dot(&dir) > 0.0)
            .map(|m| {
                let to_m = m - ball;
                let perp = (to_m.x * dir.y - to_m.y * dir.x).abs();
                mate_w * (1.0 - smoothstep(0.0, mate_radius, perp))
            })
            .fold(0.0, f64::max);

        let open = lane_openness(ball, exit, opponents, ray_corridor);
        goal_term + mate_term - open_penalty * (1.0 - open) + center_w * (half_wid - exit_y.abs())
    };

    // Candidates: a uniform angular sweep of the cone, PLUS the midpoint of every
    // open goal-mouth window. The windows can be slivers far narrower than the
    // sweep spacing (a keeper leaves ~170mm inside each post), so they must be
    // explicit candidates — and the midpoint is the right aim anyway (the same
    // choice `best_shot` makes). The mouth cone is narrow from long range
    // (~5° at full field), so the sweep step is fine-grained.
    let steps = ((a_hi - a_lo) / 1.5_f64.to_radians()).ceil().max(1.0) as usize;
    let sweep = (0..=steps).map(|i| {
        let theta = a_lo + (a_hi - a_lo) * (i as f64) / (steps as f64);
        ball.y + theta.tan() * depth
    });
    // Window midpoints are inside the mouth but can fall between a post and the
    // inset bound; clamp them into the cone so the execution margin holds.
    let windows = gaps
        .iter()
        .map(move |&(g_lo, g_hi)| (0.5 * (g_lo + g_hi)).clamp(-y_lim, y_lim));

    sweep
        .chain(windows)
        .map(|y| (y, score_exit(y)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(y, _)| Vector2::new(opp_goal_x, y))
}

/// A scored keeper-clear ray: the point to strike through and how open the
/// near portion of its lane is (the caller's hold-vs-fire gate).
#[derive(Debug, Clone, Copy)]
pub struct ClearRay {
    pub target: Vector2,
    pub openness: f64,
}

/// Length of a panic-fan ray's aim target past the ball. Only the heading
/// matters to the strike; this just has to be long enough to be directionally
/// stable.
const CLEAR_PANIC_RAY_LEN: f64 = 5000.0;

/// Panic-fan geometry: rays swept from this angle off straight-downfield (+x)
/// out to nearly lateral, on both sides, in `CLEAR_PANIC_FAN_STEP` increments.
const CLEAR_PANIC_FAN_LO: f64 = 30.0 * std::f64::consts::PI / 180.0;
const CLEAR_PANIC_FAN_HI: f64 = 80.0 * std::f64::consts::PI / 180.0;
const CLEAR_PANIC_FAN_STEP: f64 = 10.0 * std::f64::consts::PI / 180.0;

/// Score sink for a clear ray that fires into a nearby opponent — larger than any
/// clean ray's score, so a vetoed ray is chosen only when every candidate is
/// vetoed (a forced panic clear with the keeper fully surrounded).
const CLEAR_VETO_PENALTY: f64 = 1.0e6;

/// Pick the keeper's clear ray by scoring candidates against ALL robots — the
/// opponents *and* our own field players (`own_others`, keeper excluded). The
/// own-robot term is the point of this scorer: the old fixed-target clear was
/// blind to our own wall parked on the ball→goal ray and blasted into its backs.
///
/// Candidates are the Div-B-safe goal-mouth cone (sweep + open-window midpoints,
/// as in [`hoof_ray_target`]); with `panic` set — the Keeper-Held-Ball deadline
/// looming — a wide lateral fan is added and a sideways-preference term
/// (`panic_side_w · |dir.y|`) joins the score: a blocked central ray ricochets
/// back toward our goal, a blocked sideways ray deflects laterally, and a clean
/// sideways exit over the touch line is only a throw-in (no foul counter, never
/// aimless).
///
/// Openness is measured over the first `open_range` mm of the ray only: a
/// blocker well downfield is a rule-safe en-route touch (defuses aimless), not a
/// bounce-back risk, and must not keep the keeper holding forever. The returned
/// [`ClearRay::openness`] is that near-lane openness, so the caller can gate
/// hold-vs-fire on it; the goal-window bonus and side term only rank candidates.
#[allow(clippy::too_many_arguments)]
pub fn best_clear_ray(
    ball: Vector2,
    opponents: &[PlayerState],
    own_others: &[Vector2],
    opp_goal_x: f64,
    goal_width: f64,
    opp_keeper_id: Option<PlayerId>,
    robot_radius: f64,
    keeper_radius: f64,
    ball_radius: f64,
    ray_corridor: f64,
    open_range: f64,
    goal_w: f64,
    mouth_inset: f64,
    panic: bool,
    panic_side_w: f64,
    veto_reach: f64,
    veto_corridor: f64,
) -> Option<ClearRay> {
    // Open goal-mouth windows (their-keeper shadow inflated, as for a shot).
    let gaps = goal_free_gaps(
        ball,
        opp_goal_x,
        goal_width,
        opponents,
        opp_keeper_id,
        robot_radius,
        keeper_radius,
        ball_radius,
    )
    .map(|(free, _)| free)
    .unwrap_or_default();

    let near_openness = |target: Vector2| -> f64 {
        let full = target - ball;
        let len = full.norm();
        if len < 1e-6 {
            return 0.0;
        }
        let near_end = ball + full * (open_range.min(len) / len);
        lane_openness(ball, near_end, opponents, ray_corridor).min(lane_openness_pts(
            ball,
            near_end,
            own_others,
            ray_corridor,
        ))
    };

    let score = |target: Vector2, openness: f64| -> f64 {
        let dir = (target - ball).try_normalize(1e-6).unwrap_or(Vector2::x());
        // Goal-window bonus: does the ray exit the opponent goal line inside an
        // open mouth window? (Panic-fan rays point sideways and never do.)
        let in_mouth = if dir.x > 1e-6 {
            let exit_y = ball.y + dir.y / dir.x * (opp_goal_x - ball.x);
            gaps.iter().any(|&(lo, hi)| exit_y > lo && exit_y < hi)
        } else {
            false
        };
        let goal_term = if in_mouth { goal_w } else { 0.0 };
        let side_term = if panic {
            panic_side_w * dir.y.abs()
        } else {
            0.0
        };
        goal_term + openness + side_term
    };

    // Primary candidates: the mouth cone (posts inset as execution margin) —
    // a fine sweep plus every open window's midpoint, as in `hoof_ray_target`.
    let mut candidates: Vec<Vector2> = Vec::new();
    let depth = opp_goal_x - ball.x;
    if depth > mouth_inset {
        let y_lim = (goal_width / 2.0 - mouth_inset).max(1.0);
        let a_lo = (-y_lim - ball.y).atan2(depth);
        let a_hi = (y_lim - ball.y).atan2(depth);
        let steps = ((a_hi - a_lo) / 1.5_f64.to_radians()).ceil().max(1.0) as usize;
        for i in 0..=steps {
            let theta = a_lo + (a_hi - a_lo) * (i as f64) / (steps as f64);
            candidates.push(Vector2::new(opp_goal_x, ball.y + theta.tan() * depth));
        }
        for &(g_lo, g_hi) in &gaps {
            candidates.push(Vector2::new(
                opp_goal_x,
                (0.5 * (g_lo + g_hi)).clamp(-y_lim, y_lim),
            ));
        }
    }

    // Panic fan: strongly lateral rays on both sides.
    if panic {
        let mut theta = CLEAR_PANIC_FAN_LO;
        while theta <= CLEAR_PANIC_FAN_HI + 1e-9 {
            for sign in [-1.0, 1.0] {
                let dir = Vector2::new(theta.cos(), sign * theta.sin());
                candidates.push(ball + dir * CLEAR_PANIC_RAY_LEN);
            }
            theta += CLEAR_PANIC_FAN_STEP;
        }
    }

    candidates
        .into_iter()
        .map(|target| {
            // Hard "never clear into a nearby opponent" veto — applied hardest at
            // the keeper: a ball fired into an opponent at our keeper's feet
            // rebounds toward our OWN goal, the worst outcome on the field. A
            // vetoed ray reports zero near-openness (so a non-panic keeper HOLDS
            // rather than fire it) and sinks below every clean ray (so even a
            // forced panic clear takes any lane that misses the opponent, e.g. a
            // sideways fan ray, and only fires into one when fully surrounded).
            let vetoed =
                fires_into_close_opponent(ball, target, opponents, veto_reach, veto_corridor);
            let openness = if vetoed { 0.0 } else { near_openness(target) };
            let mut s = score(target, openness);
            if vetoed {
                s -= CLEAR_VETO_PENALTY;
            }
            (target, openness, s)
        })
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(target, openness, _)| ClearRay { target, openness })
}

/// Pick an open, forward support/outlet position on one flank.
///
/// Static flank points leave supporters stranded behind opponents, so the ball
/// can never reach them. This samples a small grid *ahead of the ball* on the
/// given flank (`sign` = ±1 for the +y / -y side) and returns the candidate with
/// the most open ball→candidate lane, mildly preferring more forward positions.
/// The result is always a valid outlet the carrier could actually pass into.
#[allow(clippy::too_many_arguments)]
pub fn best_support_pos(
    ball: Vector2,
    opponents: &[PlayerState],
    sign: f64,
    half_len: f64,
    half_wid: f64,
    corridor: f64,
    opp_pen_depth: f64,
    opp_pen_half_width: f64,
    // Attacking mode (ball in the opponent half): place supporters advanced and
    // wide so a deep carrier has a goal-line cross/cutback outlet.
    attacking: bool,
    flank_y_fracs: [f64; 3],
    goal_line_setback: f64,
    // Positions already claimed by other supporters this tick. Candidates within
    // `min_separation` of any of these are excluded — and additionally candidates
    // within `min_lateral_sep` of a taken spot's y — so supporters can never
    // resolve to the same outlet AND are structurally spread across the field
    // width (two supporters never stack on one flank).
    taken: &[Vector2],
    min_separation: f64,
    min_lateral_sep: f64,
    // Hysteresis: last tick's target for this slot (if any) and the incumbency
    // bonus. The candidate nearest `prev` gets up to `+stickiness` added to its
    // score, so a challenger must be decisively more open to displace the
    // incumbent — this is what stops the target thrashing between near-tied grid
    // cells under opponent jitter. Pass `None` / `0.0` for the stateless behaviour.
    // `sticky_radius` is the distance over which the bonus decays to zero.
    prev: Option<Vector2>,
    stickiness: f64,
    sticky_radius: f64,
    // Shot quality (0..1) at a candidate spot — the planner's own shot model
    // ([`best_shot_from`], normalized by the caller), weighted by `shot_w`, so
    // supporters stand where a received pass has a real shooting window instead
    // of merely where the lane is clear.
    shot_quality: &dyn Fn(Vector2) -> f64,
    shot_w: f64,
) -> Vector2 {
    const PAD: f64 = 400.0;
    // Conservative grid stays modest and central; attacking mode reaches toward the
    // goal line and flanks wide (outer bands clear the box keepout at the corners).
    let (x_targets, y_mags): ([f64; 3], [f64; 3]) = if attacking {
        (
            [
                ball.x + 1500.0,
                ball.x + 3000.0,
                half_len - goal_line_setback,
            ],
            flank_y_fracs,
        )
    } else {
        (
            [ball.x + 1000.0, ball.x + 2000.0, ball.x + 3000.0],
            [0.18, 0.32, 0.46],
        )
    };

    // Sample BOTH flanks plus the centre line. Previously each lateral magnitude was
    // multiplied by `sign`, so the central slot (sign 0) collapsed every candidate to
    // y = 0 — a single point in front of the box with zero freedom to slide off a
    // marker. A supporter pinned there sat fully covered for seconds. Mirroring the
    // magnitudes gives every slot real lateral options; the side bias below (not the
    // candidate set) is what keeps the slots spread across the field.
    let mut y_fracs: Vec<f64> = Vec::with_capacity(2 * y_mags.len() + 1);
    y_fracs.push(0.0);
    for &m in &y_mags {
        y_fracs.push(m);
        y_fracs.push(-m);
    }
    // Tie-break that nudges slot 0 to +y and slot 1 to −y so the supporters don't
    // stack, scaled to lose to any real openness/clearance gap (so a slot still
    // crosses the field when its own side is covered). Centre slot (sign 0) is free.
    const SIDE_BIAS: f64 = 1e-4;

    // `best` only considers candidates clear of already-taken supporter spots;
    // `fallback` ignores that constraint so we still return a sane point if every
    // candidate is crowded out (more taken spots than the grid can separate).
    let mut best: Option<(Vector2, f64)> = None;
    let mut fallback: Option<(Vector2, f64)> = None;
    for &xt in &x_targets {
        let cx = xt.clamp(-half_len + PAD, half_len - PAD);
        for &fy in &y_fracs {
            let cy = (half_wid * fy).clamp(-half_wid + PAD, half_wid - PAD);
            // Keep the outlet out of the opponent penalty area (no robot but the
            // keeper may enter it) — otherwise the supporter pins against the box
            // keepout and stalls at `no_path`.
            let cand = clamp_out_of_opp_box(
                Vector2::new(cx, cy),
                half_len,
                opp_pen_depth,
                opp_pen_half_width,
            );
            // A clear throwing lane is necessary but not sufficient: a defender
            // sitting *on* the receive point intercepts the moment the ball arrives.
            // Score openness (lane clear) AND receiver clearance (spot uncontested),
            // both in [0, 1], so a supporter peels off its marker into open space.
            // A short lane is trivially open, so without the reach gate the supporter
            // could "get open" by hugging the carrier — gate openness by distance
            // from the ball so only a real, pass-length outlet counts as open.
            const MIN_OUTLET_SEPARATION: f64 = 1200.0;
            let reach = smoothstep(0.0, MIN_OUTLET_SEPARATION, (cand - ball).norm());
            let open = lane_openness(ball, cand, opponents, corridor) * reach;
            let clearance = receiver_clearance(cand, opponents);
            // Incumbency bonus: full weight for a candidate sitting on last tick's
            // target, decaying to zero by `sticky_radius`, so the argmax holds the
            // current outlet unless a challenger is more open by `stickiness`.
            let sticky = match prev {
                Some(p) if stickiness > 0.0 => {
                    stickiness * (1.0 - smoothstep(0.0, sticky_radius, (cand - p).norm()))
                }
                _ => 0.0,
            };
            let score = open
                + clearance
                + shot_w * shot_quality(cand)
                + config::SUPPORT_FWD_X_W * cand.x
                + SIDE_BIAS * sign * cand.y
                + sticky;
            if fallback.is_none() || score > fallback.unwrap().1 {
                fallback = Some((cand, score));
            }
            if taken.iter().any(|t| {
                (cand - t).norm() < min_separation || (cand.y - t.y).abs() < min_lateral_sep
            }) {
                continue;
            }
            if best.is_none() || score > best.unwrap().1 {
                best = Some((cand, score));
            }
        }
    }
    best.or(fallback).map(|(p, _)| p).unwrap_or_else(|| {
        clamp_out_of_opp_box(
            Vector2::new(
                (ball.x + 2000.0).clamp(-half_len + PAD, half_len - PAD),
                sign * half_wid * 0.3,
            ),
            half_len,
            opp_pen_depth,
            opp_pen_half_width,
        )
    })
}

/// Clearance in [0, 1] around a candidate receive point: 0 if an opponent stands on
/// top of it, rising smoothly to 1 once the nearest opponent is at least
/// `RECEIVER_CLEARANCE_RADIUS` away. Complements [`lane_openness`]: openness asks
/// "is the throwing lane clear?", clearance asks "can I actually receive here, or is
/// a marker sitting on the spot to intercept on arrival?". Radius ~3 robot radii so a
/// defender within tackling distance tanks the spot.
pub fn receiver_clearance(at: Vector2, opponents: &[PlayerState]) -> f64 {
    const RECEIVER_CLEARANCE_RADIUS: f64 = 700.0;
    let mut min_d = f64::INFINITY;
    for opp in opponents {
        min_d = min_d.min((opp.position - at).norm());
    }
    smoothstep(0.0, RECEIVER_CLEARANCE_RADIUS, min_d)
}

/// Push a point in front of the opponent penalty area if it falls inside the box
/// inflated by `BOX_CLEARANCE` (covers the planner keepout + a buffer). Only the
/// goal-mouth y-band is constrained; points off to the side keep their x.
fn clamp_out_of_opp_box(
    p: Vector2,
    half_len: f64,
    opp_pen_depth: f64,
    opp_pen_half_width: f64,
) -> Vector2 {
    const BOX_CLEARANCE: f64 = 300.0;
    let box_front = half_len - opp_pen_depth - BOX_CLEARANCE;
    if p.x > box_front && p.y.abs() < opp_pen_half_width + BOX_CLEARANCE {
        Vector2::new(box_front, p.y)
    } else {
        p
    }
}

/// Returns `true` if `point` lies roughly between `from` and `to` along the from→to axis.
///
/// Projects `point` onto the infinite line through `from` and `to` and checks whether
/// the parameter *t* falls in [0, 1].
///
/// Reserved for the M3 conservative steal gate (is a defender between threat and goal?).
#[allow(dead_code)]
pub fn is_between(point: Vector2, from: Vector2, to: Vector2) -> bool {
    let dir = to - from;
    let len_sq = dir.x * dir.x + dir.y * dir.y;

    if len_sq < 1e-6 {
        return false;
    }

    let to_point = point - from;
    let t = (to_point.x * dir.x + to_point.y * dir.y) / len_sq;

    (0.0..=1.0).contains(&t)
}

/// Smooth Hermite step: 0 below `edge0`, 1 above `edge1`, C¹-continuous between.
/// Used everywhere a hard threshold would otherwise cause formation discontinuities.
pub fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    if (edge1 - edge0).abs() < 1e-9 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Time (s) and miss distance (mm) of the ball's closest approach to `pos`,
/// assuming straight-line ball motion. Negative time = the approach is in the
/// past (ball receding). `None` if the ball is (near) stationary.
pub fn ball_closest_approach(ball: Vector2, vel: Vector2, pos: Vector2) -> Option<(f64, f64)> {
    let speed2 = vel.norm_squared();
    if speed2 < 1e-6 {
        return None;
    }
    let t = (pos - ball).dot(&vel) / speed2;
    let miss = (pos - (ball + vel * t)).norm();
    Some((t, miss))
}

/// Continuous threat a field position poses to our goal, in [0, 1].
///
/// Combines proximity to our goal (closer = higher) with a directional term
/// (square-on to the goal mouth is worse than wide). Smooth in `pos`.
pub fn threat(pos: Vector2, own_goal: Vector2, goal_near: f64, goal_far: f64) -> f64 {
    let to_goal = own_goal - pos;
    let dist = to_goal.norm();
    let prox = 1.0 - smoothstep(goal_near, goal_far, dist);
    if dist < 1e-6 {
        return prox;
    }
    // Directional cosine toward our goal (pos on the -x side aiming at goal = 1).
    let aim = (to_goal / dist).dot(&Vector2::new(-1.0, 0.0)).max(0.0);
    prox * (0.5 + 0.5 * aim)
}

/// Opponents within this radius of the lane origin are contesting the ball at
/// the carrier's feet, not occupying the downfield lane. An on-ball defender sits
/// at the shared origin of *every* forward lane and would otherwise tank the
/// openness of all of them equally — masking a genuinely open outlet so the
/// carrier never passes and hoofs the ball away instead. Excluding it keeps
/// `lane_openness` measuring what it is meant to: how clear the lane is downfield.
/// Sized to ~2.5 robot radii — two robots plus the ball contesting one point.
const LANE_CONTEST_RADIUS: f64 = 250.0;

/// Continuous openness in [0, 1] of the lane from `from` to `to` w.r.t. opponents.
///
/// 1 if no opponent is near the segment, decaying toward 0 as the nearest in-corridor
/// opponent approaches the line.
pub fn lane_openness(from: Vector2, to: Vector2, opponents: &[PlayerState], corridor: f64) -> f64 {
    lane_openness_pts_iter(from, to, opponents.iter().map(|o| o.position), corridor)
}

/// [`lane_openness`] over bare positions — used where the blockers aren't
/// opponents (e.g. our own wall in the keeper's clear-ray scorer).
pub fn lane_openness_pts(from: Vector2, to: Vector2, blockers: &[Vector2], corridor: f64) -> f64 {
    lane_openness_pts_iter(from, to, blockers.iter().copied(), corridor)
}

fn lane_openness_pts_iter(
    from: Vector2,
    to: Vector2,
    blockers: impl Iterator<Item = Vector2>,
    corridor: f64,
) -> f64 {
    let dir = to - from;
    let len = dir.norm();
    if len < 1e-6 {
        return 1.0;
    }
    let mut min_perp = f64::INFINITY;
    for pos in blockers {
        let to_opp = pos - from;
        // Skip on-ball contesters at the lane origin: they block the first few
        // hundred mm of every lane equally and say nothing about downfield clarity.
        if to_opp.norm() < LANE_CONTEST_RADIUS {
            continue;
        }
        let t = to_opp.dot(&dir) / (len * len);
        if !(0.0..=1.0).contains(&t) {
            continue;
        }
        let perp = (to_opp.x * dir.y - to_opp.y * dir.x).abs() / len;
        min_perp = min_perp.min(perp);
    }
    if min_perp.is_infinite() {
        return 1.0;
    }
    // 0 when on the line, 1 when at/over the corridor edge.
    smoothstep(0.0, corridor, min_perp)
}

/// True if aiming from `ball` toward `target` would drive the ball almost
/// immediately into an opponent: one within `reach` of the ball, ahead along the
/// aim ray, and within `corridor` perpendicular of it. The universal restart-aim
/// veto — a restart must never be fired straight into a nearby presser (worst in
/// our own half, where the blocked ball becomes their shot on our goal).
pub fn fires_into_close_opponent(
    ball: Vector2,
    target: Vector2,
    opponents: &[PlayerState],
    reach: f64,
    corridor: f64,
) -> bool {
    let dir = target - ball;
    let len = dir.norm();
    if len < 1e-6 {
        return false;
    }
    let dir = dir / len;
    opponents.iter().any(|o| {
        let to_o = o.position - ball;
        let along = to_o.dot(&dir);
        if along <= 0.0 || along > reach {
            return false; // behind the ball, or beyond the "close" reach
        }
        let perp = (to_o.x * dir.y - to_o.y * dir.x).abs();
        perp < corridor
    })
}

/// Distance from `from` (assumed inside the field rectangle `[-half_len,half_len]
/// × [-half_wid,half_wid]`) to where the unit ray `dir` crosses the boundary.
/// `INFINITY` for a degenerate (zero) direction. Used to reject advance headings
/// that would put the ball out of play almost immediately.
pub fn ray_exit_distance(from: Vector2, dir: Vector2, half_len: f64, half_wid: f64) -> f64 {
    let mut t = f64::INFINITY;
    if dir.x.abs() > 1e-9 {
        let wall = if dir.x > 0.0 { half_len } else { -half_len };
        let tx = (wall - from.x) / dir.x;
        if tx > 0.0 {
            t = t.min(tx);
        }
    }
    if dir.y.abs() > 1e-9 {
        let wall = if dir.y > 0.0 { half_wid } else { -half_wid };
        let ty = (wall - from.y) / dir.y;
        if ty > 0.0 {
            t = t.min(ty);
        }
    }
    t
}

/// Push `p` laterally out of the stadium of half-width `half_width` around the
/// segment `a→b` (to the side of the axis it is already on; on-axis breaks
/// toward +normal deterministically). Points already outside are returned
/// unchanged. Used by the wall to vacate the keeper's clear corridor.
pub fn project_out_of_corridor(p: Vector2, a: Vector2, b: Vector2, half_width: f64) -> Vector2 {
    let ab = b - a;
    let len2 = ab.norm_squared();
    if len2 < 1e-9 {
        return p;
    }
    let t = ((p - a).dot(&ab) / len2).clamp(0.0, 1.0);
    let closest = a + ab * t;
    let off = p - closest;
    if off.norm() >= half_width {
        return p;
    }
    let axis = ab / len2.sqrt();
    let n = Vector2::new(-axis.y, axis.x);
    let side = if off.dot(&n) >= 0.0 { 1.0 } else { -1.0 };
    closest + n * (half_width * side)
}

/// Push `p` out of the keeper's clear "keyhole": the work disc of `disc_radius`
/// around `ball` (room to swing around the ball and line up from any side —
/// for a box-corner ball the wall otherwise bunches at the keepout line right
/// next to it and boxes the keeper out of its own approach) union the corridor
/// stadium toward `goal` (the strike ray). Corridor first, then disc: a radial
/// push from the ball scales a point's lateral offset to any line through the
/// ball by ≥1, so escaping the disc never re-enters the corridor. A point
/// exactly on the ball breaks perpendicular to the axis.
pub fn project_out_of_keyhole(
    p: Vector2,
    ball: Vector2,
    goal: Vector2,
    disc_radius: f64,
    corridor_half_width: f64,
) -> Vector2 {
    let p = project_out_of_corridor(p, ball, goal, corridor_half_width);
    let off = p - ball;
    let d = off.norm();
    if d >= disc_radius {
        return p;
    }
    let dir = if d > 1e-6 {
        off / d
    } else {
        let axis = (goal - ball)
            .try_normalize(1e-6)
            .unwrap_or(Vector2::new(1.0, 0.0));
        Vector2::new(-axis.y, axis.x)
    };
    ball + dir * disc_radius
}

/// The point a radial `standoff` in front of our goal along the ball→goal-centre
/// ray. Capped at 0.8·(ball distance) so it never sits behind a ball closer than
/// the standoff. This is the centre of the shadow wall (see `shadow_arc`) and is
/// reused to place the anchor at its own, shorter standoff.
pub fn goal_ray_point(ball: Vector2, own_goal: Vector2, standoff: f64) -> Vector2 {
    let to_ball = ball - own_goal;
    let dist = to_ball.norm();
    let u = to_ball
        .try_normalize(1e-6)
        .unwrap_or_else(|| Vector2::new(1.0, 0.0));
    own_goal + u * standoff.min(dist * 0.8)
}

/// Place `k` shadow defenders as a contiguous wall centred on the direct
/// ball→goal-centre line, on a standoff line in front of our goal.
///
/// The straight shot at the goal centre is the highest-percentage threat and the
/// hardest for the keeper to react to, so the wall is built centre-out: with an
/// odd `k` a robot sits on the centre line; with an even `k` two robots straddle
/// it, spaced under a robot width (`spacing`) so they overlap and leave no central
/// gap. Further robots fan symmetrically outward. Wide-angle shots past the wall
/// edges are conceded to the keeper. Positions vary continuously with the ball and
/// are ordered left→right so slot identity stays stable as the wall shifts.
pub fn shadow_arc(
    ball: Vector2,
    own_goal: Vector2,
    k: usize,
    standoff: f64,
    spacing: f64,
) -> Vec<Vector2> {
    if k == 0 {
        return Vec::new();
    }
    // Stand off the goal *radially* along the ball→goal-centre line, so the wall
    // always sits between the ball and the goal — even for a wide/corner ball.
    // (A fixed-x standoff line breaks down there: the ball→goal ray, projected
    // onto a line in *front* of a ball that's already deep in the corner, lands
    // far off the field, and the unreachable wall roles go unfilled — the goal
    // is then left to the keeper alone.) Cap the radius at 0.8·dist so the wall
    // never ends up behind the ball when it's closer than the standoff.
    let to_ball = ball - own_goal;
    let dist = to_ball.norm();
    let u = to_ball
        .try_normalize(1e-6)
        .unwrap_or_else(|| Vector2::new(1.0, 0.0));
    let center = own_goal + u * standoff.min(dist * 0.8);

    // Spread the wall perpendicular to the shot line (so it faces the ball),
    // ordered along +perp so slot identity stays stable as the wall pivots with
    // the ball. For a square-on ball this reduces to the old vertical wall.
    let perp = Vector2::new(-u.y, u.x);
    (0..k)
        .map(|i| {
            let offset = (i as f64 - (k as f64 - 1.0) / 2.0) * spacing;
            center + perp * offset
        })
        .collect()
}

/// The rounded-rectangle offset curve around a box (the defense area), parametrized
/// by signed arc-length `s`. `s = 0` is the front-centre point `(front_x + offset,
/// cy)`; `+s` runs toward `+y`. The curve is the flat front edge, a quarter-circle
/// of radius `offset` at each front corner (centred on the box corner), then the two
/// side edges — clamped at the goal line so it never wraps behind the goal. This is
/// the path the goal wall slides along; the matching [`MotionBounds::RoundedRect`]
/// band keeps robots pinned to it in the depth direction.
pub fn rect_offset_point(pen: &Rect, offset: f64, s: f64) -> Vector2 {
    let front_x = pen.max.x;
    let half_wid = (pen.max.y - pen.min.y) / 2.0;
    let cy = (pen.max.y + pen.min.y) / 2.0;
    let depth = pen.max.x - pen.min.x;
    let quarter = std::f64::consts::FRAC_PI_2 * offset;
    let l = half_wid + quarter + depth;
    let s = s.clamp(-l, l);
    let sign = if s >= 0.0 { 1.0 } else { -1.0 };
    let a = s.abs();
    let (x, ymag) = if a <= half_wid {
        // Flat front edge.
        (front_x + offset, a)
    } else if a <= half_wid + quarter {
        // Rounded corner: sweep from facing +x to facing +y.
        let ang = (a - half_wid) / offset;
        (front_x + offset * ang.cos(), half_wid + offset * ang.sin())
    } else {
        // Side edge, receding toward the goal line.
        let t = a - half_wid - quarter;
        (front_x - t, half_wid + offset)
    };
    Vector2::new(x, cy + sign * ymag)
}

/// Inverse of [`rect_offset_point`]: the signed arc-length `s` of the curve point
/// nearest `p`. Continuous across the front→corner→side transitions and the x-axis,
/// so the wall's base point (and hence slot identity) doesn't jump as the ball
/// sweeps. Used both to pick the wall's base_s from the ball ray and to measure each
/// member's along-curve position for the rigid-translation pacing.
pub fn project_to_rect_offset(pen: &Rect, offset: f64, p: Vector2) -> f64 {
    let front_x = pen.max.x;
    let half_wid = (pen.max.y - pen.min.y) / 2.0;
    let cy = (pen.max.y + pen.min.y) / 2.0;
    let depth = pen.max.x - pen.min.x;
    let quarter = std::f64::consts::FRAC_PI_2 * offset;
    let l = half_wid + quarter + depth;

    let dy = p.y - cy;
    let sign = if dy >= 0.0 { 1.0 } else { -1.0 };
    let ay = dy.abs();

    let s = if ay <= half_wid {
        // In front of (or beside) the flat front edge → arc-length along it.
        sign * ay
    } else if p.x >= front_x {
        // Off a front corner → angle swept from the corner.
        let ang = (ay - half_wid)
            .atan2((p.x - front_x).max(0.0))
            .clamp(0.0, std::f64::consts::FRAC_PI_2);
        sign * (half_wid + offset * ang)
    } else {
        // Beside a side edge → distance receded toward the goal line.
        let t = (front_x - p.x).clamp(0.0, depth);
        sign * (half_wid + quarter + t)
    };
    s.clamp(-l, l)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_offset_roundtrip_and_corner_wrap() {
        // Standard SSL own defense area: x∈[-4500,-3500], y∈[-1000,1000].
        let pen = Rect::new(
            Vector2::new(-4500.0, -1000.0),
            Vector2::new(-3500.0, 1000.0),
        );
        let off = 550.0;
        // Front-centre is at (front_x + offset, 0) at s = 0.
        let p0 = rect_offset_point(&pen, off, 0.0);
        assert!((p0 - Vector2::new(-3500.0 + off, 0.0)).norm() < 1e-6);
        // point→s→point is identity for points on the curve (front, corner, side).
        for s in [-1600.0, -900.0, -300.0, 0.0, 400.0, 1050.0, 1800.0] {
            let p = rect_offset_point(&pen, off, s);
            let s2 = project_to_rect_offset(&pen, off, p);
            let p2 = rect_offset_point(&pen, off, s2);
            assert!(
                (p - p2).norm() < 1.0,
                "roundtrip s={s}: {:?} vs {:?}",
                p,
                p2
            );
        }
        // A base near the +y front corner fans onto BOTH the front edge and the
        // side edge (wraps the corner) instead of a straight chord — every point
        // stays at ~`off` from the box (never deeper, so never inside it).
        let base = project_to_rect_offset(&pen, off, Vector2::new(-3500.0, 1000.0)); // corner
        for i in [-1.0, 0.0, 1.0] {
            let pt = rect_offset_point(&pen, off, base + i * 170.0);
            // Distance outside the box must be ~off (rounded corner, no clipping in).
            let nx = pt.x.clamp(pen.min.x, pen.max.x);
            let ny = pt.y.clamp(pen.min.y, pen.max.y);
            let d = (pt - Vector2::new(nx, ny)).norm();
            assert!(
                (d - off).abs() < 1.0,
                "fanned pt off-box dist {d}, want {off}"
            );
        }
    }

    #[test]
    fn closest_approach_head_on_offset_and_receding() {
        let ball = Vector2::new(0.0, 0.0);
        let vel = Vector2::new(1000.0, 0.0);
        // Head-on: reaches the point in 2 s, miss 0.
        let (t, d) = ball_closest_approach(ball, vel, Vector2::new(2000.0, 0.0)).unwrap();
        assert!((t - 2.0).abs() < 1e-9 && d < 1e-9);
        // Laterally offset: same time, miss = the offset.
        let (t, d) = ball_closest_approach(ball, vel, Vector2::new(2000.0, 500.0)).unwrap();
        assert!((t - 2.0).abs() < 1e-9 && (d - 500.0).abs() < 1e-9);
        // Behind the ball's travel: approach is in the past.
        let (t, _) = ball_closest_approach(ball, vel, Vector2::new(-1000.0, 0.0)).unwrap();
        assert!(t < 0.0);
        // Stationary ball: no approach.
        assert!(ball_closest_approach(ball, Vector2::new(0.0, 0.0), ball).is_none());
    }

    #[test]
    fn keeper_charge_radius_shrinks_with_advancement() {
        // On its line (no advancement) → full inflated shadow.
        assert!((keeper_charge_radius(0.0) - crate::config::SHOT_KEEPER_RADIUS).abs() < 1e-6);
        assert!(
            (keeper_charge_radius(crate::config::KEEPER_CHARGE_NEAR)
                - crate::config::SHOT_KEEPER_RADIUS)
                .abs()
                < 1e-6
        );
        // Fully committed forward → shrinks to a plain field-robot shadow.
        assert!(
            (keeper_charge_radius(crate::config::KEEPER_CHARGE_FAR)
                - crate::config::SHOT_ROBOT_RADIUS)
                .abs()
                < 1e-6
        );
        // Halfway between near and far → midpoint radius, monotone shrinking.
        let mid = keeper_charge_radius(
            (crate::config::KEEPER_CHARGE_NEAR + crate::config::KEEPER_CHARGE_FAR) / 2.0,
        );
        let expected = (crate::config::SHOT_KEEPER_RADIUS + crate::config::SHOT_ROBOT_RADIUS) / 2.0;
        assert!((mid - expected).abs() < 1e-6);
    }

    #[test]
    fn redirect_time_prefers_continuing_over_reversing() {
        let pos = Vector2::new(0.0, 0.0);
        let target = Vector2::new(1000.0, 0.0);
        let toward = redirect_time(pos, Vector2::new(2000.0, 0.0), target, 3000.0, 3000.0);
        let away = redirect_time(pos, Vector2::new(-2000.0, 0.0), target, 3000.0, 3000.0);
        let still = redirect_time(pos, Vector2::new(0.0, 0.0), target, 3000.0, 3000.0);
        assert!(
            toward < still,
            "moving toward should be cheaper than stationary"
        );
        assert!(
            still < away,
            "stationary should be cheaper than moving away"
        );
    }

    #[test]
    fn smoothstep_is_monotonic_and_clamped() {
        assert_eq!(smoothstep(0.0, 1.0, -1.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 2.0), 1.0);
        let a = smoothstep(0.0, 1.0, 0.25);
        let b = smoothstep(0.0, 1.0, 0.75);
        assert!(a < b);
    }

    #[test]
    fn shadow_wall_covers_the_direct_goal_line() {
        // Opponent kickoff: ball dead centre, our goal at -x. For every wall size,
        // some robot must sit within a robot radius of the direct ball→goal-centre
        // line (y=0 here) — otherwise a straight kick threads the gap.
        const ROBOT_RADIUS: f64 = 90.0;
        let own_goal = Vector2::new(-4500.0, 0.0);
        let ball = Vector2::new(0.0, 0.0);
        for k in 1..=3 {
            let wall = shadow_arc(ball, own_goal, k, 1500.0, 170.0);
            assert_eq!(wall.len(), k);
            let min_abs_y = wall.iter().map(|p| p.y.abs()).fold(f64::INFINITY, f64::min);
            assert!(
                min_abs_y < ROBOT_RADIUS,
                "k={k}: nearest shadow is {min_abs_y}mm off the goal line, leaving a central gap"
            );
            // Wall sits on the standoff line in front of goal.
            assert!(wall.iter().all(|p| (p.x - (-3000.0)).abs() < 1e-6));
        }
    }

    #[test]
    fn shadow_wall_is_symmetric_about_the_goal_line() {
        let own_goal = Vector2::new(-4500.0, 0.0);
        let ball = Vector2::new(0.0, 0.0);
        let wall = shadow_arc(ball, own_goal, 3, 1500.0, 170.0);
        let sum_y: f64 = wall.iter().map(|p| p.y).sum();
        assert!(
            sum_y.abs() < 1e-6,
            "wall should be centred, got y-sum {sum_y}"
        );
    }

    #[test]
    fn shadow_wall_stays_on_field_for_a_corner_ball() {
        // Free kick deep in our corner (the collapse scenario, team-relative: our
        // goal at -x). The old fixed-x standoff projected the wall ~3 m off the
        // field; the radial standoff must keep every robot on the field and
        // strictly between the ball and the goal.
        let own_goal = Vector2::new(-4500.0, 0.0);
        let ball = Vector2::new(-4000.0, 2000.0);
        let half_wid = 3000.0;
        for k in 1..=3 {
            let wall = shadow_arc(ball, own_goal, k, 1500.0, 170.0);
            assert_eq!(wall.len(), k);
            for p in &wall {
                assert!(
                    p.y.abs() <= half_wid,
                    "k={k}: shadow off the field at {p:?}"
                );
                // Closer to the goal than the ball is, and in front of the goal line.
                assert!(
                    (p - own_goal).norm() < (ball - own_goal).norm(),
                    "k={k}: shadow {p:?} is not between ball and goal"
                );
                assert!(
                    p.x > own_goal.x,
                    "k={k}: shadow {p:?} is behind the goal line"
                );
            }
        }
    }

    #[test]
    fn threat_is_higher_near_our_goal() {
        let own_goal = Vector2::new(-4500.0, 0.0);
        let near = threat(Vector2::new(-3500.0, 0.0), own_goal, 1500.0, 6000.0);
        let far = threat(Vector2::new(3000.0, 0.0), own_goal, 1500.0, 6000.0);
        assert!(near > far);
    }

    #[test]
    fn support_pos_avoids_standing_behind_opponent() {
        // An opponent sits straight ahead on the +y flank; the supporter must pick
        // an open lane instead of parking behind it.
        let ball = Vector2::new(0.0, 0.0);
        let blocker = PlayerState::new(
            PlayerId::new(9),
            Vector2::new(2000.0, 960.0),
            Vector2::new(0.0, 0.0),
            Angle::from_radians(0.0),
        );
        let pos = best_support_pos(
            ball,
            std::slice::from_ref(&blocker),
            1.0,
            4500.0,
            3000.0,
            500.0,
            1000.0,
            1000.0,
            false,
            [0.46, 0.62, 0.78],
            400.0,
            &[],
            1500.0,
            0.0,
            None,
            0.0,
            0.0,
            &|_| 0.0,
            0.0,
        );
        let openness = lane_openness(ball, pos, std::slice::from_ref(&blocker), 500.0);
        assert!(pos.y > 0.0, "support should stay on its flank");
        assert!(
            openness > 0.9,
            "support lane should be open, got {openness} at {pos:?}"
        );
    }

    #[test]
    fn support_pos_never_enters_opponent_penalty_area() {
        // Ball near the opponent end with the goal-mouth lanes wide open: without
        // box-awareness the outlet would land inside the opponent penalty area.
        let half_len = 4500.0;
        let pen_depth = 1000.0;
        let pen_half_width = 1000.0;
        let ball = Vector2::new(3000.0, 0.0);
        for &sign in &[1.0, -1.0] {
            let pos = best_support_pos(
                ball,
                &[],
                sign,
                half_len,
                3000.0,
                500.0,
                pen_depth,
                pen_half_width,
                true,
                [0.46, 0.62, 0.78],
                400.0,
                &[],
                1500.0,
                0.0,
                None,
                0.0,
                0.0,
                &|_| 0.0,
                0.0,
            );
            let in_box = pos.x > half_len - pen_depth && pos.y.abs() < pen_half_width;
            assert!(!in_box, "support landed inside the opp box at {pos:?}");
        }
    }

    #[test]
    fn attacking_support_flanks_the_goal_for_a_deep_carrier() {
        // Ball deep in the opponent half with open lanes: attacking mode should put
        // the supporter advanced (near the goal line) and wide (flanking the box),
        // not stranded level/behind near the centre.
        let half_len = 4500.0;
        let pos = best_support_pos(
            Vector2::new(3000.0, 0.0),
            &[],
            1.0,
            half_len,
            3000.0,
            500.0,
            1000.0, // pen depth
            1000.0, // pen half width
            true,
            [0.46, 0.62, 0.78],
            400.0,
            &[],
            1500.0,
            0.0,
            None,
            0.0,
            0.0,
            &|_| 0.0,
            0.0,
        );
        assert!(
            pos.x > 3000.0,
            "attacking supporter should be advanced past the ball, got {pos:?}"
        );
        assert!(
            pos.y > 1300.0,
            "attacking supporter should flank wide of the box, got {pos:?}"
        );
    }

    #[test]
    fn central_support_slot_dodges_a_marker_instead_of_pinning() {
        // Regression for the "fully-covered supporter parks in front of the box"
        // bug (frame 4776, yellow p4). Ball deep in the attacking corner, the
        // central slot (sign 0), and a defender sitting on the lane to the single
        // box-front point the slot used to collapse onto (~3200, 0). The slot must
        // now have the lateral freedom to relocate to an open, uncontested spot.
        let half_len = 4500.0;
        let ball = Vector2::new(3920.0, 2280.0);
        let marker = PlayerState::new(
            PlayerId::new(5),
            Vector2::new(3269.0, 318.0), // on the lane to the old pinned point
            Vector2::new(0.0, 0.0),
            Angle::from_radians(0.0),
        );
        let pos = best_support_pos(
            ball,
            std::slice::from_ref(&marker),
            0.0, // central slot — used to collapse all candidates to y = 0
            half_len,
            3000.0,
            500.0,
            1000.0,
            1000.0,
            true,
            [0.46, 0.62, 0.78],
            400.0,
            &[],
            1500.0,
            0.0,
            None,
            0.0,
            0.0,
            &|_| 0.0,
            0.0,
        );
        assert!(
            (pos - marker.position).norm() > 700.0,
            "supporter should peel off the marker, got {pos:?} \
             ({:.0}mm from marker)",
            (pos - marker.position).norm()
        );
        let open = lane_openness(ball, pos, std::slice::from_ref(&marker), 500.0);
        let clearance = receiver_clearance(pos, std::slice::from_ref(&marker));
        assert!(
            open > 0.9 && clearance > 0.9,
            "relocated spot should be open and uncontested, got open={open} clearance={clearance} at {pos:?}"
        );
    }

    #[test]
    fn supporters_never_resolve_to_the_same_outlet() {
        // With a wide-open field every slot's raw argmax is the same deep/wide point,
        // so without the `taken` exclusion all supporters would stack there. Placing
        // them greedily must structurally separate them by at least `min_separation`.
        let half_len = 4500.0;
        let ball = Vector2::new(2000.0, 0.0);
        let min_sep = 1500.0;
        let mut taken: Vec<Vector2> = Vec::new();
        for &sign in &[1.0, -1.0, 0.0] {
            let pos = best_support_pos(
                ball,
                &[], // no opponents → openness is uniform, argmax ties everywhere
                sign,
                half_len,
                3000.0,
                500.0,
                1000.0,
                1000.0,
                true,
                [0.46, 0.62, 0.78],
                400.0,
                &taken,
                min_sep,
                0.0,
                None,
                0.0,
                0.0,
                &|_| 0.0,
                0.0,
            );
            for prev in &taken {
                assert!(
                    (pos - prev).norm() >= min_sep,
                    "supporter at {pos:?} is within {min_sep} of {prev:?}"
                );
            }
            taken.push(pos);
        }
    }

    #[test]
    fn incumbency_bonus_holds_a_tied_outlet_but_yields_to_a_better_one() {
        // The thrashing fix: `best_support_pos` is a discrete argmax, so when two
        // grid outlets are near-tied the winner flips every tick under opponent
        // jitter, yanking the supporter's target across the field. Last tick's
        // target gets an incumbency bonus so the choice sticks — UNLESS a
        // challenger is decisively more open.
        let half_len = 4500.0;
        let ball = Vector2::new(3000.0, 0.0);
        let flanks = [0.46, 0.62, 0.78];

        // 1. Perfect tie (no opponents, central slot): the default tie-break picks
        //    one flank deterministically. Feed the *mirror* point back as `prev`;
        //    the bonus must flip the winner to the incumbent's side, proving the
        //    hysteresis — not a fixed bias — is what decides a tie.
        let base = best_support_pos(
            ball,
            &[],
            0.0,
            half_len,
            3000.0,
            500.0,
            1000.0,
            1000.0,
            true,
            flanks,
            400.0,
            &[],
            1500.0,
            0.0,
            None,
            0.0,
            0.0,
            &|_| 0.0,
            0.0,
        );
        let mirror = Vector2::new(base.x, -base.y);
        let held = best_support_pos(
            ball,
            &[],
            0.0,
            half_len,
            3000.0,
            500.0,
            1000.0,
            1000.0,
            true,
            flanks,
            400.0,
            &[],
            1500.0,
            0.0,
            Some(mirror),
            crate::config::SUPPORT_STICKINESS,
            crate::config::SUPPORT_STICKY_RADIUS,
            &|_| 0.0,
            0.0,
        );
        assert!(
            (held - mirror).norm() < (base - mirror).norm(),
            "incumbency should hold the previous outlet {mirror:?} against a tie, got {held:?}"
        );

        // 2. Decisive gap: an opponent parks on the incumbent outlet (clearance → 0)
        //    while the opposite flank is wide open. The ~1.0 clearance gap dwarfs the
        //    0.18 bonus, so the supporter must relocate rather than pin on a covered
        //    spot.
        let covered = base; // incumbent outlet from step 1
        let marker = PlayerState::new(
            PlayerId::new(9),
            covered,
            Vector2::new(0.0, 0.0),
            Angle::from_radians(0.0),
        );
        let moved = best_support_pos(
            ball,
            std::slice::from_ref(&marker),
            0.0,
            half_len,
            3000.0,
            500.0,
            1000.0,
            1000.0,
            true,
            flanks,
            400.0,
            &[],
            1500.0,
            0.0,
            Some(covered),
            crate::config::SUPPORT_STICKINESS,
            crate::config::SUPPORT_STICKY_RADIUS,
            &|_| 0.0,
            0.0,
        );
        assert!(
            (moved - covered).norm() > crate::config::SUPPORT_STICKY_RADIUS,
            "supporter should yield a covered outlet despite incumbency, stayed at {moved:?}"
        );
    }

    /// `best_clear_ray` with production-shaped params: Div-B geometry, ball in
    /// our box, no opponents unless given.
    fn clear_ray(
        ball: Vector2,
        own_others: &[Vector2],
        opponents: &[PlayerState],
        panic: bool,
    ) -> ClearRay {
        best_clear_ray(
            ball, opponents, own_others, 4500.0, 1000.0, None, 110.0, 280.0, 21.5, 300.0, 3000.0,
            0.5, 150.0, panic, 0.3, 1500.0, 550.0,
        )
        .expect("clear geometry from our box is never degenerate")
    }

    #[test]
    fn clear_ray_sees_own_wall_as_blocker() {
        // THE bug this scorer exists for: our own anchor parked on the
        // ball→opp-goal ray. Every goal-bound ray passes within a few mm of it
        // (the mouth cone is ±~2° from our box), so openness must collapse —
        // the keeper holds instead of blasting into the wall's back.
        let ball = Vector2::new(-4000.0, 0.0);
        let wall = [Vector2::new(-3350.0, 0.0)]; // anchor: box edge + margin
        let blocked = clear_ray(ball, &wall, &[], false);
        assert!(
            blocked.openness < 0.2,
            "own wall on the ray must tank openness: {}",
            blocked.openness
        );
        // Wall stepped out of the corridor (the formation's vacate move) →
        // the same ray is open and the target is goal-bound (inside the mouth).
        let parted = [Vector2::new(-3350.0, 600.0)];
        let open = clear_ray(ball, &parted, &[], false);
        assert!(
            open.openness > 0.9,
            "parted wall must leave the lane open: {}",
            open.openness
        );
        assert!(
            (open.target.x - 4500.0).abs() < 1e-6 && open.target.y.abs() < 500.0,
            "primary clear must be goal-bound: {:?}",
            open.target
        );
    }

    #[test]
    fn clear_ray_panic_prefers_open_sideways_over_blocked_central() {
        // Panic with the central lane still blocked: a sideways ray deflects
        // laterally (worst case throw-in), a central one ricochets back at our
        // goal — the side fan must win.
        let ball = Vector2::new(-4000.0, 0.0);
        let wall = [Vector2::new(-3350.0, 0.0)];
        let r = clear_ray(ball, &wall, &[], true);
        let dir = (r.target - ball).normalize();
        assert!(
            dir.y.abs() > 0.7,
            "blocked centre in panic must go sideways: dir {dir:?}"
        );
        assert!(r.openness > 0.9, "chosen side lane is open: {}", r.openness);
    }

    #[test]
    fn clear_ray_panic_still_takes_an_open_goal_ray() {
        // Panic alone must not force a sideways ball: with the corridor open,
        // the goal-window bonus outranks the side-preference term.
        let ball = Vector2::new(-4000.0, 0.0);
        let r = clear_ray(ball, &[], &[], true);
        assert!(
            (r.target.x - 4500.0).abs() < 1e-6 && r.target.y.abs() < 500.0,
            "open goal ray must win in panic too: {:?}",
            r.target
        );
    }

    #[test]
    fn clear_ray_ignores_downfield_blockers() {
        // An opponent deep in their half sitting on the goal-bound ray is a
        // rule-safe en-route touch, not a bounce-back risk: it must not hold
        // the keeper's fire gate hostage (openness is near-range only).
        let ball = Vector2::new(-4000.0, 0.0);
        let deep_opp = PlayerState::new(
            PlayerId::new(9),
            Vector2::new(2000.0, 0.0), // 6m downfield, dead on the ray
            Vector2::new(0.0, 0.0),
            Angle::from_radians(0.0),
        );
        let r = clear_ray(ball, &[], std::slice::from_ref(&deep_opp), false);
        assert!(
            r.openness > 0.9,
            "a 6m-downfield blocker must not gate the fire: {}",
            r.openness
        );
    }

    #[test]
    fn clear_ray_vetoes_firing_into_nearby_opponent() {
        // Hardest close-opponent rule: an opponent right in front of the keeper's
        // ball sits in every goal-bound mouth-cone ray (the cone is only ±~2° from
        // our box). Firing there rebounds into our OWN goal — so the veto collapses
        // near-openness and the keeper HOLDS (not-panic) rather than fire.
        let ball = Vector2::new(-4000.0, 0.0);
        let close_opp = PlayerState::new(
            PlayerId::new(9),
            Vector2::new(-3400.0, 0.0), // 600mm dead ahead, on the goal ray
            Vector2::new(0.0, 0.0),
            Angle::from_radians(0.0),
        );
        let r = clear_ray(ball, &[], std::slice::from_ref(&close_opp), false);
        assert!(
            r.openness < 0.01,
            "a nearby opponent on the ray must veto the fire (hold): {}",
            r.openness
        );
    }

    #[test]
    fn clear_ray_panic_steers_around_nearby_opponent() {
        // Same nearby opponent, but the held-ball deadline is up: the keeper MUST
        // clear. It must still pick a lane that misses the opponent — a strongly
        // sideways fan ray whose near lane is clean — never fire into its feet.
        let ball = Vector2::new(-4000.0, 0.0);
        let close_opp = PlayerState::new(
            PlayerId::new(9),
            Vector2::new(-3400.0, 0.0),
            Vector2::new(0.0, 0.0),
            Angle::from_radians(0.0),
        );
        let r = clear_ray(ball, &[], std::slice::from_ref(&close_opp), true);
        let dir = (r.target - ball).normalize();
        assert!(
            dir.y.abs() > 0.7,
            "panic clear must steer sideways around the opponent: dir {dir:?}"
        );
        assert!(
            r.openness > 0.9,
            "the chosen sideways lane must be clean: {}",
            r.openness
        );
    }

    #[test]
    fn project_out_of_corridor_moves_only_intruders() {
        let a = Vector2::new(-4000.0, 0.0);
        let b = Vector2::new(4500.0, 0.0);
        // On the axis: pushed out to exactly half-width (deterministic +n side).
        let on_axis = project_out_of_corridor(Vector2::new(-3350.0, 0.0), a, b, 450.0);
        assert!((on_axis.y.abs() - 450.0).abs() < 1e-6, "{on_axis:?}");
        assert!(
            (on_axis.x - -3350.0).abs() < 1e-6,
            "x preserved: {on_axis:?}"
        );
        // Inside, off-axis: pushed to its own side.
        let below = project_out_of_corridor(Vector2::new(-3000.0, -100.0), a, b, 450.0);
        assert!((below.y - -450.0).abs() < 1e-6, "keeps its side: {below:?}");
        // Already outside: untouched.
        let out = Vector2::new(-3000.0, 800.0);
        assert_eq!(project_out_of_corridor(out, a, b, 450.0), out);
        // Beyond the segment's cap: stadium distance applies, not infinite axis —
        // a point 300mm behind the cap is inside and gets pushed to the cap
        // circle; one 600mm behind is already outside and stays put.
        let behind = Vector2::new(-4300.0, 0.0);
        let pushed = project_out_of_corridor(behind, a, b, 450.0);
        assert!(
            ((pushed - a).norm() - 450.0).abs() < 1e-6,
            "cap-circle distance: {pushed:?}"
        );
        let far_behind = Vector2::new(-4600.0, 0.0);
        assert_eq!(project_out_of_corridor(far_behind, a, b, 450.0), far_behind);
    }

    #[test]
    fn keyhole_clears_the_keeper_work_disc() {
        // The keyhole = corridor + work disc around the ball. A robot crowding
        // the ball from BEHIND (the box-corner stall: teammates bunched at the
        // keepout line next to the ball, outside the forward corridor's reach)
        // must be pushed to the disc radius; corridor intruders still leave the
        // corridor AND the disc; far bystanders are untouched.
        let ball = Vector2::new(-4260.0, 940.0);
        let goal = Vector2::new(4500.0, 0.0);
        let (disc, corr) = (700.0, 450.0);
        // Behind/beside the ball, outside the corridor, inside the disc.
        let crowding = Vector2::new(-4100.0, 1100.0);
        let p = project_out_of_keyhole(crowding, ball, goal, disc, corr);
        assert!(
            ((p - ball).norm() - disc).abs() < 1e-6,
            "pushed to the disc rim: {p:?}"
        );
        // On the strike ray near the ball: must clear BOTH corridor and disc.
        let on_ray = ball + (goal - ball).normalize() * 300.0;
        let p = project_out_of_keyhole(on_ray, ball, goal, disc, corr);
        let seg_d = {
            let ab = goal - ball;
            let t = ((p - ball).dot(&ab) / ab.norm_squared()).clamp(0.0, 1.0);
            (p - (ball + ab * t)).norm()
        };
        assert!((p - ball).norm() >= disc - 1e-6, "out of the disc: {p:?}");
        assert!(seg_d >= corr - 1e-6, "out of the corridor: {p:?}");
        // Far bystander: untouched.
        let far = Vector2::new(-2000.0, -1500.0);
        assert_eq!(project_out_of_keyhole(far, ball, goal, disc, corr), far);
    }

    #[test]
    fn lane_openness_drops_when_blocked() {
        let from = Vector2::new(0.0, 0.0);
        let to = Vector2::new(2000.0, 0.0);
        let blocker = PlayerState::new(
            PlayerId::new(9),
            Vector2::new(1000.0, 0.0),
            Vector2::new(0.0, 0.0),
            Angle::from_radians(0.0),
        );
        let blocked = lane_openness(from, to, std::slice::from_ref(&blocker), 500.0);
        let clear = lane_openness(from, to, &[], 500.0);
        assert!(blocked < clear);
        assert_eq!(clear, 1.0);
    }

    #[test]
    fn lane_openness_ignores_on_ball_contester() {
        // Regression for the "pass into the void" bug: a defender pressed on the
        // carrier sits at the shared origin of every forward lane. It must not tank
        // the openness of a downfield lane that is otherwise wide open, or the
        // planner never commits the pass and hoofs the ball into the corner.
        let from = Vector2::new(0.0, 0.0);
        let to = Vector2::new(3000.0, 0.0); // an open outlet straight ahead
        let contester = PlayerState::new(
            PlayerId::new(9),
            Vector2::new(200.0, 5.0), // on the ball, 5mm off the line, ~200mm away
            Vector2::new(0.0, 0.0),
            Angle::from_radians(0.0),
        );
        let open = lane_openness(from, to, std::slice::from_ref(&contester), 400.0);
        assert_eq!(open, 1.0, "an on-ball contester must not block the lane");

        // A blocker the same 5mm off the line but genuinely downfield still blocks.
        let downfield = PlayerState::new(
            PlayerId::new(9),
            Vector2::new(1500.0, 5.0),
            Vector2::new(0.0, 0.0),
            Angle::from_radians(0.0),
        );
        let blocked = lane_openness(from, to, std::slice::from_ref(&downfield), 400.0);
        assert!(
            blocked < 0.1,
            "a downfield blocker on the line still blocks"
        );
    }

    // Shared hoof_ray_target invocation with the config-default weights.
    fn hoof(ball: Vector2, opps: &[PlayerState], mates: &[Vector2]) -> Option<Vector2> {
        hoof_ray_target(
            ball, opps, mates, 4500.0, 1000.0, None, 110.0, 280.0, 21.5, 3000.0, 3000.0, 2000.0,
            1200.0, 1500.0, 600.0, 0.3, 150.0,
        )
    }

    /// A wall of four opponents whose projected shadows (from the origin) close
    /// the whole goal mouth — no open window survives.
    fn mouth_wall() -> Vec<PlayerState> {
        [-360.0, -120.0, 120.0, 360.0]
            .iter()
            .enumerate()
            .map(|(i, &y)| opp(20 + i as u32, 4000.0, y))
            .collect()
    }

    #[test]
    fn hoof_ray_from_kickoff_spot_exists_and_is_goal_bound() {
        // Regression: the old landing-point scorer had NO legal direction from the
        // exact center spot (in-x needed ≥33° off-axis, in-y needed ≤33°).
        let t = hoof(Vector2::new(0.0, 0.0), &[], &[]).expect("kickoff hoof must exist");
        assert!(
            (t.x - 4500.0).abs() < 1e-6,
            "exit must be on their goal line"
        );
        assert!(
            t.y.abs() <= 350.0 + 1e-6,
            "exit must be inside the (inset) goal mouth, got {t:?}"
        );
    }

    #[test]
    fn hoof_ray_targets_open_mouth_window_past_a_keeper() {
        // A centred keeper leaves two slivers just inside the posts; the goal
        // term should put the exit inside one of them, not at the keeper.
        let keeper = opp(9, 4400.0, 0.0);
        let t = hoof_ray_target(
            Vector2::new(0.0, 0.0),
            std::slice::from_ref(&keeper),
            &[],
            4500.0,
            1000.0,
            Some(keeper.id),
            110.0,
            280.0,
            21.5,
            3000.0,
            3000.0,
            2000.0,
            1200.0,
            1500.0,
            600.0,
            0.3,
            150.0,
        )
        .expect("hoof must exist");
        assert!(
            t.y.abs() > 250.0 && t.y.abs() < 500.0,
            "exit should be in a post sliver, not at the keeper, got {t:?}"
        );
    }

    #[test]
    fn hoof_ray_favors_forward_teammate_when_mouth_is_walled() {
        // Rays are confined to the goal-mouth cone (the Div-B aimless-kick fix),
        // so a mate can only bias the ray *within* the mouth: a teammate on the
        // +y side of the cone should pull the exit to the +y half of the mouth.
        let opps = mouth_wall();
        let mate = Vector2::new(2000.0, 800.0);
        let t = hoof(Vector2::new(0.0, 0.0), &opps, std::slice::from_ref(&mate))
            .expect("hoof must exist");
        assert!(
            t.y > 0.0,
            "ray should bend toward the forward target man's side of the mouth, got {t:?}"
        );
    }

    #[test]
    fn hoof_ray_avoids_intercepted_lanes() {
        // A central keeper leaves a sliver inside each post; en-route blockers
        // sitting on the +y-side mouth rays: the openness penalty should push
        // the exit to the clean -y sliver. (With a fully walled mouth every
        // legal ray ends at a wall robot, so openness cannot discriminate —
        // the mouth-cone design accepts that and ties break centrally.)
        let opps = vec![
            opp(9, 4400.0, 0.0),
            opp(30, 2000.0, 250.0),
            opp(31, 3000.0, 280.0),
        ];
        let t = hoof(Vector2::new(0.0, 0.0), &opps, &[]).expect("hoof must exist");
        assert!(t.y < 0.0, "blocked +y lanes should lose, got {t:?}");
    }

    fn opp(id: u32, x: f64, y: f64) -> PlayerState {
        PlayerState::new(
            PlayerId::new(id),
            Vector2::new(x, y),
            Vector2::new(0.0, 0.0),
            Angle::from_radians(0.0),
        )
    }

    // Geometry helpers use the attacking-toward-+x convention (opp goal at +x).
    const GOAL_X: f64 = 4500.0;
    const GOAL_W: f64 = 1000.0;
    const RR: f64 = 110.0;
    const KR: f64 = 280.0;
    const BR: f64 = 21.5;
    const BIAS: f64 = 0.35;

    #[test]
    fn open_goal_no_keeper_aims_centre() {
        // Empty mouth → widest window is the whole goal, aim at centre.
        let ball = Vector2::new(3000.0, 0.0);
        let aim = best_shot(ball, GOAL_X, GOAL_W, &[], None, RR, KR, BR, BIAS)
            .expect("open mouth must yield a shot");
        assert!(
            aim.target.y.abs() < 1.0,
            "should aim centre, got {:?}",
            aim.target
        );
        assert!(
            aim.angle > 0.2,
            "full mouth should be a wide window, got {}",
            aim.angle
        );
    }

    #[test]
    fn central_keeper_forces_a_corner() {
        // Keeper square in the mouth centre → the centre is shadowed, aim must shift
        // off-centre into one of the open corners.
        let ball = Vector2::new(3000.0, 0.0);
        let keeper = opp(9, 4400.0, 0.0);
        let aim = best_shot(
            ball,
            GOAL_X,
            GOAL_W,
            &[keeper],
            Some(PlayerId::new(9)),
            RR,
            KR,
            BR,
            BIAS,
        )
        .expect("corners are open");
        assert!(
            aim.target.y.abs() > 200.0,
            "central keeper should push the aim into a corner, got {:?}",
            aim.target
        );
    }

    #[test]
    fn frame_3963_off_centre_keeper_yields_open_corner_shot() {
        // The canonical missed chance, mapped into +x convention: carrier ~1.3m out,
        // central; keeper offset to +y. Both corners are open and the keeper is
        // slightly toward +y, so the model must find a comfortably-open window and
        // prefer the corner *away* from the keeper (-y).
        let ball = Vector2::new(3167.0, 28.0); // mirror of real (-3167,-28)
        let keeper = opp(1, 4214.0, -188.0); // mirror of real (-4214,188)
        let aim = best_shot(
            ball,
            GOAL_X,
            GOAL_W,
            &[keeper],
            Some(PlayerId::new(1)),
            RR,
            KR,
            BR,
            BIAS,
        )
        .expect("a 1.3m chance with an off-centre keeper must be a shot");
        assert!(
            aim.angle >= crate::config::SHOT_MIN_ANGLE,
            "window {} should clear the shoot gate {}",
            aim.angle,
            crate::config::SHOT_MIN_ANGLE
        );
        assert!(
            aim.target.y > 0.0,
            "keeper sits at -y, so aim should bias to the +y corner, got {:?}",
            aim.target
        );
    }

    #[test]
    fn fully_crowded_mouth_has_no_shot() {
        // A wall of opponents spanning the mouth just in front of the line leaves no
        // gap → no shot (caller falls through to pass/kick-ahead).
        let ball = Vector2::new(3000.0, 0.0);
        let wall: Vec<PlayerState> = (0..6)
            .map(|i| opp(i, 4300.0, -500.0 + i as f64 * 200.0))
            .collect();
        assert!(
            best_shot(
                ball,
                GOAL_X,
                GOAL_W,
                &wall,
                Some(PlayerId::new(0)),
                RR,
                KR,
                BR,
                BIAS
            )
            .filter(|s| s.angle >= crate::config::SHOT_MIN_ANGLE)
            .is_none(),
            "a packed mouth should offer no viable shot window"
        );
    }

    #[test]
    fn finishing_pocket_slides_off_a_covered_base() {
        // Ball on +y → finisher works the -y side. A defender sits exactly on the
        // default cutback point, killing its receiver clearance; the pocket search
        // must relocate the finisher to a still-open, shootable spot.
        let ball = Vector2::new(3000.0, 500.0);
        let base = Vector2::new(3500.0, -400.0);
        let opps = vec![opp(5, 3500.0, -400.0)]; // a defender sitting on `base`
        let pos = best_finishing_pocket(
            base, ball, GOAL_X, GOAL_W, &opps, None, RR, KR, BR, BIAS, 3000.0, // half_wid
            400.0,  // y_offset
            None, 0.0, 0.0,
        );
        assert!(
            (pos - base).norm() > 1.0,
            "a marker on the base point should push the finisher off it, got {pos:?}"
        );
        assert!(
            best_shot(pos, GOAL_X, GOAL_W, &opps, None, RR, KR, BR, BIAS)
                .map(|s| s.angle)
                .unwrap_or(0.0)
                > 0.0,
            "the chosen pocket must still have an open shooting window, got {pos:?}"
        );
        assert!(
            receiver_clearance(pos, &opps) > 0.0,
            "the chosen pocket must be clear of the marker, got {pos:?}"
        );
    }

    #[test]
    fn finishing_pocket_incumbency_can_hold_a_pocket() {
        // No opponents: every pocket is clear, scored purely by its shot angle, so
        // the stateless search picks the widest-angle candidate. Feeding a *different*
        // valid pocket back as the incumbent with a strong enough multiplicative
        // bonus must flip the choice to it — proving the hysteresis can hold a pocket
        // the raw argmax wouldn't pick — while the normal bonus on the incumbent
        // simply keeps it.
        let ball = Vector2::new(3000.0, 500.0);
        let base = Vector2::new(3500.0, -400.0);
        let stateless = best_finishing_pocket(
            base,
            ball,
            GOAL_X,
            GOAL_W,
            &[],
            None,
            RR,
            KR,
            BR,
            BIAS,
            3000.0,
            400.0,
            None,
            0.0,
            0.0,
        );
        // A different but valid candidate: deeper depth (base.x - 300) on the widest
        // away-side band (y_offset * 1.7), matching the grid `best_finishing_pocket`
        // samples so it lands exactly on a candidate.
        let alt = Vector2::new(base.x - 300.0, -400.0 * 1.7);
        assert!(
            (alt - stateless).norm() > 1.0,
            "alt {alt:?} must differ from the stateless pick {stateless:?}"
        );
        // Normal bonus on the incumbent holds it.
        let held = best_finishing_pocket(
            base,
            ball,
            GOAL_X,
            GOAL_W,
            &[],
            None,
            RR,
            KR,
            BR,
            BIAS,
            3000.0,
            400.0,
            Some(stateless),
            crate::config::BOX_RUNNER_STICKINESS,
            crate::config::BOX_RUNNER_STICKY_RADIUS,
        );
        assert!(
            (held - stateless).norm() < 1.0,
            "the incumbent pocket should hold under a normal bonus, got {held:?}"
        );
        // A strong bonus on the alternative flips the choice onto it.
        let flipped = best_finishing_pocket(
            base,
            ball,
            GOAL_X,
            GOAL_W,
            &[],
            None,
            RR,
            KR,
            BR,
            BIAS,
            3000.0,
            400.0,
            Some(alt),
            5.0,
            crate::config::BOX_RUNNER_STICKY_RADIUS,
        );
        assert!(
            (flipped - alt).norm() < 1.0,
            "a strong incumbency bonus should hold the alt pocket, got {flipped:?}"
        );
    }

    #[test]
    fn finishing_pocket_degrades_to_base_with_no_window() {
        // Mouth fully walled off → no candidate has a shot window → return `base`
        // unchanged (never worse than the old static behaviour).
        let ball = Vector2::new(3000.0, 500.0);
        let base = Vector2::new(3500.0, -400.0);
        let wall: Vec<PlayerState> = (0..7)
            .map(|i| opp(i, 4300.0, -600.0 + i as f64 * 200.0))
            .collect();
        let pos = best_finishing_pocket(
            base, ball, GOAL_X, GOAL_W, &wall, None, RR, KR, BR, BIAS, 3000.0, 400.0, None, 0.0,
            0.0,
        );
        assert!(
            (pos - base).norm() < 1.0,
            "with no shooting window the finisher should stay on base, got {pos:?}"
        );
    }
}
