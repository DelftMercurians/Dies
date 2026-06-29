use dies_strategy_api::prelude::*;

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

/// If `ball` is within `margin` of a touchline or goal line, return the heading a
/// capturer should dribble it on to "rescue" it back into play: the inward normal
/// of the nearest boundary, blended with the toward-`opp_goal` direction by
/// `goal_bias`. Returns `None` when the ball is comfortably inside the field (the
/// caller then uses the normal toward-goal pickup heading).
///
/// The dominant component is always inward, so driving the ball on this heading
/// moves it *away* from the line rather than along/over it — the touchline-capture
/// fix. The nearest single boundary is chosen; corners pick whichever is closer.
pub fn boundary_rescue_heading(
    ball: Vector2,
    opp_goal: Vector2,
    half_len: f64,
    half_wid: f64,
    margin: f64,
    goal_bias: f64,
) -> Option<Angle> {
    // Distance to each boundary and that boundary's inward normal.
    let candidates = [
        (half_wid - ball.y, Vector2::new(0.0, -1.0)), // top touchline
        (half_wid + ball.y, Vector2::new(0.0, 1.0)),  // bottom touchline
        (half_len - ball.x, Vector2::new(-1.0, 0.0)), // opp goal line
        (half_len + ball.x, Vector2::new(1.0, 0.0)),  // own goal line
    ];
    let (dist, inward) = candidates
        .into_iter()
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))?;
    if dist > margin {
        return None;
    }

    let to_goal = (opp_goal - ball)
        .try_normalize(1e-6)
        .unwrap_or(Vector2::new(1.0, 0.0));
    let blended = inward + to_goal * goal_bias;
    let dir = blended.try_normalize(1e-6).unwrap_or(inward);
    Some(Angle::from_radians(dir.y.atan2(dir.x)))
}

/// Check whether the corridor from `ball_pos` to `goal_center` is free of opponents.
///
/// The corridor is a rectangle of the given `corridor_width` (perpendicular to the
/// ball→goal line). An opponent blocks the shot when its projection onto the segment
/// falls in [0, 1] **and** its perpendicular distance is less than `corridor_width / 2`.
pub fn is_clear_shot(
    ball_pos: Vector2,
    goal_center: Vector2,
    opponents: &[PlayerState],
    corridor_width: f64,
) -> bool {
    let dir = goal_center - ball_pos;
    let len_sq = dir.x * dir.x + dir.y * dir.y;

    // Degenerate case: ball is on top of the goal.
    if len_sq < 1e-6 {
        return true;
    }

    let half_w = corridor_width / 2.0;

    for opp in opponents {
        let to_opp = opp.position - ball_pos;

        // Projection parameter along the segment.
        let t = (to_opp.x * dir.x + to_opp.y * dir.y) / len_sq;

        if !(0.0..=1.0).contains(&t) {
            continue;
        }

        // Perpendicular distance from the line.
        let perp_dist = (to_opp.x * dir.y - to_opp.y * dir.x).abs() / len_sq.sqrt();

        if perp_dist < half_w {
            return false;
        }
    }

    true
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
            let score = angle * receiver_clearance(cand, opponents);
            if best.map(|(_, s)| score > s).unwrap_or(true) {
                best = Some((cand, score));
            }
        }
    }
    best.map(|(p, _)| p).unwrap_or(base)
}

/// Find the best area in the opponent half to pass the ball to.
///
/// Samples a grid of candidate positions in the attacking half (x > 0) and scores
/// each one by how far it is from the nearest opponent plus a bonus for having a
/// clear passing lane from `ball_pos`.
///
/// Returns `None` if every candidate scores below the minimum threshold (500 mm).
///
/// Reserved for the passing milestone (planner's no-clear-shot branch).
#[allow(dead_code)]
pub fn best_pass_area(
    ball_pos: Vector2,
    opponents: &[PlayerState],
    field_half_length: f64,
    field_half_width: f64,
) -> Option<Vector2> {
    let xs = [field_half_length * 0.25, field_half_length * 0.6];
    let ys = [
        -field_half_width * 0.4,
        -field_half_width * 0.15,
        0.0,
        field_half_width * 0.15,
        field_half_width * 0.4,
    ];

    const MIN_SCORE: f64 = 500.0;
    const CLEAR_LANE_BONUS: f64 = 300.0;
    const LANE_CORRIDOR: f64 = 200.0;

    let mut best: Option<(Vector2, f64)> = None;

    for &x in &xs {
        for &y in &ys {
            let candidate = Vector2::new(x, y);

            // Minimum distance to any opponent.
            let min_opp_dist = opponents
                .iter()
                .map(|o| (o.position - candidate).norm())
                .fold(f64::INFINITY, f64::min);

            // Clear-lane bonus: simplified corridor check between ball and candidate.
            let lane_bonus = if is_clear_shot(ball_pos, candidate, opponents, LANE_CORRIDOR) {
                CLEAR_LANE_BONUS
            } else {
                0.0
            };

            let score = min_opp_dist + 0.5 * lane_bonus;

            if score > MIN_SCORE && (best.is_none() || score > best.unwrap().1) {
                best = Some((candidate, score));
            }
        }
    }

    best.map(|(pos, _)| pos)
}

/// Aim a full-power advancement kick so the ball comes to rest INSIDE the field.
///
/// A struck ball rolls a roughly fixed distance (`travel` ≈ KICK_SPEED / ball
/// damping) before stopping, regardless of where the kicker "aimed" — so naively
/// hoofing toward an open-space point a short way ahead overshoots and sends the
/// ball out of bounds (a stoppage that hands the opponent a free kick). Instead of
/// choosing a *point*, this chooses a kick *direction*: it sweeps a forward cone
/// around the ball→goal line and keeps only directions whose resting point
/// (`ball + travel·dir`) stays within the field-of-play margin. Among those it
/// favours forward progress and a landing spot clear of opponents. When a straight
/// hoof would clear the field, an angled one toward a corner keeps the ball in play.
///
/// Returns the resting point to aim at (the driver kicks along `ball → point`), or
/// `None` if no forward direction keeps the ball in — the caller should then keep
/// the ball (dribble) rather than boot it out.
#[allow(clippy::too_many_arguments)]
pub fn safe_kick_target(
    ball: Vector2,
    opponents: &[PlayerState],
    opp_goal: Vector2,
    half_len: f64,
    half_wid: f64,
    travel: f64,
    margin: f64,
    min_progress: f64,
    open_weight: f64,
    open_cap: f64,
) -> Option<Vector2> {
    let to_goal = opp_goal - ball;
    let (base, goal_dir) = if to_goal.norm() > 1e-6 {
        (to_goal.y.atan2(to_goal.x), to_goal / to_goal.norm())
    } else {
        (0.0, Vector2::new(1.0, 0.0))
    };
    let x_lim = half_len - margin;
    let y_lim = half_wid - margin;

    let mut best: Option<(Vector2, f64)> = None;
    // Sweep ±80° around the goal direction in 10° steps.
    for i in -8..=8 {
        let theta = base + (i as f64) * 10.0_f64.to_radians();
        let dir = Vector2::new(theta.cos(), theta.sin());
        let landing = ball + dir * travel;
        if landing.x.abs() > x_lim || landing.y.abs() > y_lim {
            continue; // a full-power kick this way rolls out of bounds
        }
        let progress = goal_dir.dot(&(landing - ball));
        if progress < min_progress {
            continue; // too little forward gain to justify releasing the ball
        }
        let min_opp = opponents
            .iter()
            .map(|o| (o.position - landing).norm())
            .fold(f64::INFINITY, f64::min);
        let score = progress + open_weight * min_opp.min(open_cap);
        if best.map_or(true, |(_, b)| score > b) {
            best = Some((landing, score));
        }
    }
    best.map(|(p, _)| p)
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
    // `min_separation` of any of these are excluded, so two supporters can never
    // resolve to the same outlet (structural anti-stacking, not a soft tie-break).
    taken: &[Vector2],
    min_separation: f64,
) -> Vector2 {
    const PAD: f64 = 400.0;
    // Conservative grid stays modest and central; attacking mode reaches toward the
    // goal line and flanks wide (outer bands clear the box keepout at the corners).
    let (x_targets, y_mags, fwd_bonus): ([f64; 3], [f64; 3], f64) = if attacking {
        (
            [
                ball.x + 1500.0,
                ball.x + 3000.0,
                half_len - goal_line_setback,
            ],
            flank_y_fracs,
            // Stronger pull toward the goal line; still below the openness range
            // (0..1) so we never stand behind an opponent for the sake of depth.
            1e-4,
        )
    } else {
        (
            [ball.x + 1000.0, ball.x + 2000.0, ball.x + 3000.0],
            [0.18, 0.32, 0.46],
            5e-5,
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
            let score = open + clearance + fwd_bonus * cand.x + SIDE_BIAS * sign * cand.y;
            if fallback.is_none() || score > fallback.unwrap().1 {
                fallback = Some((cand, score));
            }
            if taken.iter().any(|t| (cand - t).norm() < min_separation) {
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

/// Direction to break out of a ball contest: a unit vector **perpendicular to the
/// carrier→presser squeeze axis**, on whichever side is more open.
///
/// In a pin both dribblers push along the squeeze axis, so the perpendicular is
/// the one unopposed escape direction. Of the two perpendiculars we pick the side
/// with the more open lane from the ball (via [`lane_openness`]), mildly biased
/// toward midfield so we don't shepherd the ball into a touchline. Robust to the
/// degenerate case where carrier and presser coincide.
pub fn escape_direction(
    carrier: Vector2,
    presser: Vector2,
    ball: Vector2,
    opponents: &[PlayerState],
    field_half_width: f64,
) -> Vector2 {
    const PROBE_DIST: f64 = 1000.0;
    const CORRIDOR: f64 = 400.0;
    const CENTER_BIAS: f64 = 0.25;

    // Squeeze axis (carrier → presser), with finite fallbacks if robots coincide.
    let mut axis = presser - carrier;
    if axis.norm() < 1e-6 {
        axis = ball - carrier;
    }
    if axis.norm() < 1e-6 {
        axis = Vector2::new(1.0, 0.0);
    }
    let axis = axis / axis.norm();
    let perp = Vector2::new(-axis.y, axis.x); // unit ⟂ to the axis

    let half_w = field_half_width.max(1.0);
    let score = |dir: Vector2| -> f64 {
        let cand = ball + dir * PROBE_DIST;
        let openness = lane_openness(ball, cand, opponents, CORRIDOR);
        let center = 1.0 - (cand.y.abs() / half_w).min(1.0);
        openness + CENTER_BIAS * center
    };

    if score(perp) >= score(-perp) {
        perp
    } else {
        -perp
    }
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
/// opponent approaches the line. The smooth analogue of [`is_clear_shot`].
pub fn lane_openness(from: Vector2, to: Vector2, opponents: &[PlayerState], corridor: f64) -> f64 {
    let dir = to - from;
    let len = dir.norm();
    if len < 1e-6 {
        return 1.0;
    }
    let mut min_perp = f64::INFINITY;
    for opp in opponents {
        let to_opp = opp.position - from;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_rescue_pushes_inward_near_top_touchline() {
        // Ball 200mm inside the top touchline (half_wid = 3000), opp goal at +x.
        // Rescue heading must point into the field (-y) so the dribble moves the
        // ball away from the line, while keeping a forward (+x) component.
        let ball = Vector2::new(-2735.0, 2800.0);
        let opp_goal = Vector2::new(4500.0, 0.0);
        let h = boundary_rescue_heading(ball, opp_goal, 4500.0, 3000.0, 300.0, 0.6)
            .expect("ball within margin should trigger a rescue");
        let v = h.to_vector();
        assert!(
            v.y < -0.5,
            "heading should pull strongly inward (-y), got {v:?}"
        );
        assert!(
            v.x > 0.0,
            "heading should keep a forward (+x) component, got {v:?}"
        );
    }

    #[test]
    fn no_rescue_when_ball_is_well_inside() {
        let ball = Vector2::new(0.0, 0.0);
        let opp_goal = Vector2::new(4500.0, 0.0);
        assert!(boundary_rescue_heading(ball, opp_goal, 4500.0, 3000.0, 300.0, 0.6).is_none());
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
    fn escape_direction_is_lateral_and_avoids_the_blocked_side() {
        // Presser dead ahead along +x → escape must be lateral (±y). An opponent
        // blocks the +y lane, so we must squirt out toward -y.
        let carrier = Vector2::new(0.0, 0.0);
        let presser = Vector2::new(500.0, 0.0);
        let ball = Vector2::new(250.0, 0.0);
        let blocker = PlayerState::new(
            PlayerId::new(9),
            Vector2::new(250.0, 1000.0), // sitting in the +y escape lane
            Vector2::new(0.0, 0.0),
            Angle::from_radians(0.0),
        );
        let dir = escape_direction(
            carrier,
            presser,
            ball,
            std::slice::from_ref(&blocker),
            3000.0,
        );
        assert!(
            dir.x.abs() < 1e-6,
            "escape should be perpendicular to the axis"
        );
        assert!(
            dir.y < 0.0,
            "escape should avoid the blocked +y side, got {dir:?}"
        );
        assert!(
            (dir.norm() - 1.0).abs() < 1e-6,
            "escape should be a unit vector"
        );
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

    #[test]
    fn safe_kick_target_keeps_the_ball_in_field() {
        let opp_goal = Vector2::new(4500.0, 0.0);
        let half_len = 4500.0;
        let half_wid = 3000.0;
        let margin = 300.0;
        let in_field = |p: Vector2| {
            p.x.abs() <= half_len - margin + 1.0 && p.y.abs() <= half_wid - margin + 1.0
        };

        // Deep in our own half there is room ahead: a long forward hoof stays in
        // play and advances the ball.
        let from_back = Vector2::new(-2000.0, 0.0);
        let t = safe_kick_target(
            from_back,
            &[],
            opp_goal,
            half_len,
            half_wid,
            5000.0,
            margin,
            800.0,
            0.5,
            2000.0,
        )
        .expect("a forward hoof should be available from deep");
        assert!(in_field(t), "resting point must stay in field, got {t:?}");
        assert!(t.x > from_back.x, "hoof should advance the ball, got {t:?}");

        // In the attacking half a full-power kick (~5 m roll) cannot stay in play,
        // so the helper declines (None) and the planner keeps the ball instead of
        // booting it out of bounds.
        let from_front = Vector2::new(2000.0, 0.0);
        assert!(
            safe_kick_target(
                from_front,
                &[],
                opp_goal,
                half_len,
                half_wid,
                5000.0,
                margin,
                800.0,
                0.5,
                2000.0,
            )
            .is_none(),
            "no full-power hoof should stay in field from the attacking half"
        );
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
            base,
            ball,
            GOAL_X,
            GOAL_W,
            &opps,
            None,
            RR,
            KR,
            BR,
            BIAS,
            3000.0, // half_wid
            400.0,  // y_offset
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
    fn finishing_pocket_degrades_to_base_with_no_window() {
        // Mouth fully walled off → no candidate has a shot window → return `base`
        // unchanged (never worse than the old static behaviour).
        let ball = Vector2::new(3000.0, 500.0);
        let base = Vector2::new(3500.0, -400.0);
        let wall: Vec<PlayerState> = (0..7)
            .map(|i| opp(i, 4300.0, -600.0 + i as f64 * 200.0))
            .collect();
        let pos = best_finishing_pocket(
            base, ball, GOAL_X, GOAL_W, &wall, None, RR, KR, BR, BIAS, 3000.0, 400.0,
        );
        assert!(
            (pos - base).norm() < 1.0,
            "with no shooting window the finisher should stay on base, got {pos:?}"
        );
    }
}
