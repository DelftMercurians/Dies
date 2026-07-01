//! Global path planner: grid-based any-angle search (Theta*) with a
//! line-of-sight fast path and per-robot hysteresis.
//!
//! The planner answers one question per robot per tick: *which way* should it
//! head to route around static geometry toward its target. It returns a single
//! next waypoint that becomes the MTP setpoint; ORCA then handles the moving
//! robots reactively on top.
//!
//! Cost is kept near-zero in the common case: a straight line-of-sight check
//! short-circuits to the target whenever the direct path is clear (open play),
//! and a hysteresis cache reuses the committed path while the target hasn't
//! moved and the route is still clear — so the full Theta* search only runs when
//! a robot is genuinely boxed out of a direct shot.
//!
//! All coordinates are team-relative, millimetres.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use dies_core::{AvoidanceConfig, DebugColor, PlayerId, Vector2};

use super::super::team_context::PlayerContext;
use super::obstacle::{ObstacleSet, ObstacleShape};

/// Upper bound on rendered path segments; stale keys above the live count are
/// cleared each frame so a shrinking path doesn't leave ghosts.
const MAX_DEBUG_SEGS: usize = 32;

/// How far past the start↔target bounding box the local search grid extends, to
/// give Theta* room to detour around obstacles [mm].
const GRID_MARGIN: f64 = 1500.0;
/// Safety cap on A* node expansions; a blown budget falls back to a straight
/// line (ORCA still protects the robot).
const MAX_EXPANSIONS: usize = 20_000;

pub struct GlobalPlanner {
    grid_resolution: f64,
    /// Agent-disc inflation used for planner occupancy: `robot_clearance +
    /// planner_margin`, so the planner clears robots by more than ORCA's hard
    /// radius and routes outside ORCA's braking band.
    clearance: f64,
    replan_target_tol: f64,
    cached: HashMap<PlayerId, CachedPath>,
}

#[derive(Clone)]
struct CachedPath {
    /// Remaining waypoints (string-pulled), final entry == target.
    waypoints: Vec<Vector2>,
    target: Vector2,
}

impl GlobalPlanner {
    pub fn new(cfg: &AvoidanceConfig) -> Self {
        Self {
            grid_resolution: cfg.grid_resolution.max(10.0),
            clearance: cfg.robot_clearance + cfg.planner_margin,
            replan_target_tol: cfg.replan_target_tol,
            cached: HashMap::new(),
        }
    }

    pub fn update_settings(&mut self, cfg: &AvoidanceConfig) {
        self.grid_resolution = cfg.grid_resolution.max(10.0);
        self.clearance = cfg.robot_clearance + cfg.planner_margin;
        self.replan_target_tol = cfg.replan_target_tol;
    }

    /// Drop cached paths for robots we no longer control.
    pub fn retain(&mut self, live: &HashSet<PlayerId>) {
        self.cached.retain(|id, _| live.contains(id));
    }

    /// Next waypoint for `id` to steer toward, and whether it is the **final
    /// target** (decelerate to a stop) or an intermediate corner (pass through
    /// at cruise). Falls back to `target` (straight line) whenever the direct
    /// path is clear or planning fails. Draws the planned route (player-scoped
    /// debug under `plan.*`).
    /// The remaining path the robot should follow: a polyline from the next
    /// waypoint to the final target (`[target]` for a clear line-of-sight or a
    /// planning failure). The path-follower consumes the whole thing. Draws the
    /// route under the player's `plan.*` debug keys.
    pub fn plan(
        &mut self,
        id: PlayerId,
        start: Vector2,
        target: Vector2,
        ego_radius: f64,
        obstacles: &ObstacleSet,
        ctx: &PlayerContext,
    ) -> Vec<Vector2> {
        let shapes: Vec<ObstacleShape> = obstacles.as_planner_shapes(self.clearance).collect();
        let step = self.grid_resolution * 0.5;

        // 1. Line-of-sight fast path: direct shot is clear → straight to target.
        if segment_clear(start, target, &shapes, ego_radius, step) {
            self.cached.remove(&id);
            ctx.debug_string("plan.status", "direct");
            self.draw_route(ctx, start, &[target], DebugColor::Green);
            return vec![target];
        }

        // 2. Hysteresis: reuse the committed path while the target is stable and
        //    the remaining route is still collision-free.
        if let Some(route) = self.try_cached(id, start, target, &shapes, ego_radius, step) {
            ctx.debug_string("plan.status", "cached");
            self.draw_route(ctx, start, &route, DebugColor::Orange);
            return route;
        }

        // 3. Full Theta* search.
        match self.search(start, target, &shapes, ego_radius, step) {
            Some(path) => {
                ctx.debug_string("plan.status", "planned");
                self.draw_route(ctx, start, &path, DebugColor::Orange);
                self.cached.insert(
                    id,
                    CachedPath {
                        waypoints: path.clone(),
                        target,
                    },
                );
                path
            }
            None => {
                self.cached.remove(&id);
                // Planner couldn't find a route → straight-line fallback. Drawn
                // red to flag that the robot is heading straight at obstacles.
                ctx.debug_string("plan.status", "no_path");
                self.draw_route(ctx, start, &[target], DebugColor::Red);
                vec![target]
            }
        }
    }

    /// Returns the remaining route (waypoints, final == target) if the cached
    /// path is still valid; advances past reached waypoints as a side effect.
    fn try_cached(
        &mut self,
        id: PlayerId,
        start: Vector2,
        target: Vector2,
        shapes: &[ObstacleShape],
        ego_radius: f64,
        step: f64,
    ) -> Option<Vec<Vector2>> {
        let cached = self.cached.get_mut(&id)?;
        if (cached.target - target).norm() > self.replan_target_tol {
            return None;
        }
        // Prune intermediate waypoints we've passed (cache hygiene only — the
        // follower projects onto the whole polyline anyway). The final target
        // (last entry) is never dropped here.
        let arrive = self.grid_resolution;
        while cached.waypoints.len() > 1 && (cached.waypoints[0] - start).norm() < arrive {
            cached.waypoints.remove(0);
        }
        let wps = cached.waypoints.clone();
        if wps.is_empty() {
            return None;
        }
        // The remaining route (start → wp0 → wp1 → … → target) must still be clear.
        if !segment_clear(start, wps[0], shapes, ego_radius, step) {
            return None;
        }
        for seg in wps.windows(2) {
            if !segment_clear(seg[0], seg[1], shapes, ego_radius, step) {
                return None;
            }
        }
        Some(wps)
    }

    /// Render the planned route as connected segments plus a marker on the next
    /// waypoint, clearing stale segment keys from a previous longer path.
    fn draw_route(
        &self,
        ctx: &PlayerContext,
        start: Vector2,
        route: &[Vector2],
        color: DebugColor,
    ) {
        let mut prev = start;
        for (i, wp) in route.iter().enumerate().take(MAX_DEBUG_SEGS) {
            ctx.debug_line_colored(format!("plan.seg{:02}", i), prev, *wp, color);
            prev = *wp;
        }
        for i in route.len()..MAX_DEBUG_SEGS {
            ctx.debug_remove(format!("plan.seg{:02}", i));
        }
        if let Some(next) = route.first() {
            ctx.debug_cross_colored("plan.waypoint", *next, DebugColor::Purple);
        }
    }

    /// Theta* over a local grid. Returns the string-pulled waypoint list
    /// (excluding `start`, final entry == `target`), or `None` if unreachable.
    fn search(
        &self,
        start: Vector2,
        target: Vector2,
        shapes: &[ObstacleShape],
        ego_radius: f64,
        step: f64,
    ) -> Option<Vec<Vector2>> {
        let res = self.grid_resolution;
        let lo = Vector2::new(
            start.x.min(target.x) - GRID_MARGIN,
            start.y.min(target.y) - GRID_MARGIN,
        );
        let hi = Vector2::new(
            start.x.max(target.x) + GRID_MARGIN,
            start.y.max(target.y) + GRID_MARGIN,
        );
        let nx = (((hi.x - lo.x) / res).ceil() as usize).max(1) + 1;
        let ny = (((hi.y - lo.y) / res).ceil() as usize).max(1) + 1;
        let cell_pos =
            |i: usize, j: usize| Vector2::new(lo.x + i as f64 * res, lo.y + j as f64 * res);
        let to_cell = |p: Vector2| {
            let i = (((p.x - lo.x) / res).round() as i64).clamp(0, nx as i64 - 1) as usize;
            let j = (((p.y - lo.y) / res).round() as i64).clamp(0, ny as i64 - 1) as usize;
            (i, j)
        };
        let idx = |i: usize, j: usize| j * nx + i;
        let blocked = |i: usize, j: usize| min_distance(cell_pos(i, j), shapes) < ego_radius;

        let (si, sj) = to_cell(start);
        let (gi, gj) = to_cell(target);

        // If the goal cell is solid the target is unreachable on the grid; bail
        // to the straight-line fallback.
        if blocked(gi, gj) {
            return None;
        }

        let n = nx * ny;
        let mut g = vec![f64::INFINITY; n];
        let mut parent = vec![usize::MAX; n];
        let mut closed = vec![false; n];

        let start_i = idx(si, sj);
        let goal_i = idx(gi, gj);
        g[start_i] = 0.0;
        parent[start_i] = start_i;

        let h = |i: usize, j: usize| (cell_pos(i, j) - cell_pos(gi, gj)).norm();
        let mut open = BinaryHeap::new();
        open.push(Node {
            f: h(si, sj),
            cell: start_i,
        });

        let mut expansions = 0usize;
        while let Some(Node { cell, .. }) = open.pop() {
            if cell == goal_i {
                return Some(self.reconstruct(
                    cell, &parent, start, target, &cell_pos, nx, shapes, ego_radius, step,
                ));
            }
            if closed[cell] {
                continue;
            }
            closed[cell] = true;
            expansions += 1;
            if expansions > MAX_EXPANSIONS {
                return None;
            }

            let (ci, cj) = (cell % nx, cell / nx);
            for (di, dj) in NEIGHBORS {
                let ni = ci as i64 + di;
                let nj = cj as i64 + dj;
                if ni < 0 || nj < 0 || ni >= nx as i64 || nj >= ny as i64 {
                    continue;
                }
                let (ni, nj) = (ni as usize, nj as usize);
                if blocked(ni, nj) {
                    continue;
                }
                let nb = idx(ni, nj);
                if closed[nb] {
                    continue;
                }

                // Theta*: try to inherit the parent of `cell` if it can see the
                // neighbour, yielding any-angle (non-grid) paths.
                let par = parent[cell];
                let (cand_parent, cand_g) = {
                    let (pi, pj) = (par % nx, par / nx);
                    if segment_clear(cell_pos(pi, pj), cell_pos(ni, nj), shapes, ego_radius, step) {
                        (par, g[par] + (cell_pos(pi, pj) - cell_pos(ni, nj)).norm())
                    } else {
                        (cell, g[cell] + (cell_pos(ci, cj) - cell_pos(ni, nj)).norm())
                    }
                };

                if cand_g < g[nb] {
                    g[nb] = cand_g;
                    parent[nb] = cand_parent;
                    open.push(Node {
                        f: cand_g + h(ni, nj),
                        cell: nb,
                    });
                }
            }
        }
        None
    }

    #[allow(clippy::too_many_arguments)]
    fn reconstruct(
        &self,
        goal: usize,
        parent: &[usize],
        start: Vector2,
        target: Vector2,
        cell_pos: &impl Fn(usize, usize) -> Vector2,
        nx: usize,
        shapes: &[ObstacleShape],
        ego_radius: f64,
        step: f64,
    ) -> Vec<Vector2> {
        // Grid path from goal back to start.
        let mut rev = vec![target];
        let mut cur = goal;
        loop {
            let p = parent[cur];
            if p == cur {
                break;
            }
            rev.push(cell_pos(p % nx, p / nx));
            cur = p;
        }
        rev.reverse(); // now start-ish → target; rev[0] is the start cell
                       // Replace the start cell with the exact start, then string-pull.
        rev[0] = start;

        // String-pull: greedily keep the farthest point reachable from the last
        // committed anchor, dropping redundant intermediate waypoints.
        let mut out: Vec<Vector2> = Vec::new();
        let mut anchor = start;
        let mut i = 1;
        while i < rev.len() {
            // Extend as far as line-of-sight allows.
            let mut j = i;
            while j + 1 < rev.len() && segment_clear(anchor, rev[j + 1], shapes, ego_radius, step) {
                j += 1;
            }
            out.push(rev[j]);
            anchor = rev[j];
            i = j + 1;
        }
        if out.is_empty() {
            out.push(target);
        }
        out
    }
}

const NEIGHBORS: [(i64, i64); 8] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
];

struct Node {
    f: f64,
    cell: usize,
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f
    }
}
impl Eq for Node {}
impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap on f: reverse the float comparison.
        other.f.total_cmp(&self.f)
    }
}
impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Minimum signed clearance over all shapes at `p` (∞ if no obstacles).
fn min_distance(p: Vector2, shapes: &[ObstacleShape]) -> f64 {
    shapes
        .iter()
        .map(|s| s.distance(p))
        .fold(f64::INFINITY, f64::min)
}

/// Whether a robot of `ego_radius` can travel the straight segment a→b without
/// any obstacle clearance dropping below `ego_radius`. Sampled at `step`.
fn segment_clear(
    a: Vector2,
    b: Vector2,
    shapes: &[ObstacleShape],
    ego_radius: f64,
    step: f64,
) -> bool {
    let len = (b - a).norm();
    let n = (len / step.max(1.0)).ceil() as usize;
    for k in 0..=n {
        let t = if n == 0 { 0.0 } else { k as f64 / n as f64 };
        let p = a + (b - a) * t;
        if min_distance(p, shapes) < ego_radius {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::super::super::team_context::{PlayerContext, TeamContext};
    use super::super::obstacle::{ObstacleKind, ObstacleSet, ObstacleShape, StaticObstacle};
    use super::{segment_clear, GlobalPlanner};
    use dies_core::{AvoidanceConfig, PlayerId, SideAssignment, TeamColor, Vector2};

    fn ctx() -> PlayerContext {
        TeamContext::new(TeamColor::Blue, SideAssignment::YellowOnPositive)
            .player_context(PlayerId::new(0))
    }

    fn disc(center: Vector2, radius: f64) -> ObstacleSet {
        ObstacleSet {
            statics: vec![StaticObstacle {
                shape: ObstacleShape::Circle { center, radius },
                kind: ObstacleKind::Ball,
            }],
            agents: Vec::new(),
        }
    }

    #[test]
    fn clear_line_returns_target() {
        let cfg = AvoidanceConfig::default();
        let mut p = GlobalPlanner::new(&cfg);
        let target = Vector2::new(1000.0, 0.0);
        let path = p.plan(
            PlayerId::new(0),
            Vector2::new(-1000.0, 0.0),
            target,
            90.0,
            &ObstacleSet::default(),
            &ctx(),
        );
        assert_eq!(path.len(), 1, "direct shot is a single-point path");
        assert!((path[0] - target).norm() < 1e-6, "got {:?}", path);
    }

    #[test]
    fn detours_around_disc() {
        let cfg = AvoidanceConfig::default();
        let mut p = GlobalPlanner::new(&cfg);
        let start = Vector2::new(-1000.0, 0.0);
        let target = Vector2::new(1000.0, 0.0);
        // Disc squarely blocking the direct path.
        let obs = disc(Vector2::new(0.0, 0.0), 400.0);
        let path = p.plan(PlayerId::new(0), start, target, 90.0, &obs, &ctx());
        let wp = path[0];

        assert!(
            path.len() >= 2,
            "a detour has intermediate waypoints: {:?}",
            path
        );
        assert!(
            (wp - target).norm() > 1e-6,
            "should not go straight: {:?}",
            wp
        );
        // The first leg toward the chosen waypoint must itself be collision-free.
        let shapes: Vec<ObstacleShape> = obs.as_planner_shapes(cfg.robot_clearance).collect();
        assert!(
            segment_clear(start, wp, &shapes, 90.0, cfg.grid_resolution * 0.5),
            "first leg not clear: {:?}",
            wp
        );
    }
}
