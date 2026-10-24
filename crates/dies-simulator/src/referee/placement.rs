//! Restart placement constraints and ball repositioning.
use super::*;
/// How far inside the field lines the ball is placed after it leaves the field.
/// Must be large enough that a robot can fully get behind the ball to take the
/// free kick (otherwise it would stay pinned against the boundary wall).
/// Doubles as the rule 5.3.3 "0.2 m from all field lines" placement constraint.
pub(crate) const FREE_KICK_PLACEMENT_MARGIN: f64 = 200.0;
/// Minimum distance a free-kick ball must be placed from either defense area
/// (SSL rule 5.3.3). Keeping this clear is what lets the kicker get behind the
/// ball without violating the 0.2 m opponent-defense-area keep-out (rule 8.4.1).
pub(crate) const FREE_KICK_DEFENSE_AREA_DISTANCE: f64 = 1000.0;

/// If `p` lies within `margin` of the axis-aligned rectangle `[min, max]`, move
/// it to the closest point exactly `margin` outside the rectangle; otherwise
/// return it unchanged.
fn push_out_of_rect(p: Vector2, min: Vector2, max: Vector2, margin: f64) -> Vector2 {
    let inside_x = p.x > min.x && p.x < max.x;
    let inside_y = p.y > min.y && p.y < max.y;

    if inside_x && inside_y {
        // Strictly inside: exit through the nearest face.
        let to_left = p.x - min.x;
        let to_right = max.x - p.x;
        let to_bottom = p.y - min.y;
        let to_top = max.y - p.y;
        let nearest = to_left.min(to_right).min(to_bottom).min(to_top);
        let mut out = p;
        if nearest == to_left {
            out.x = min.x - margin;
        } else if nearest == to_right {
            out.x = max.x + margin;
        } else if nearest == to_bottom {
            out.y = min.y - margin;
        } else {
            out.y = max.y + margin;
        }
        return out;
    }

    let nearest = Vector2::new(p.x.clamp(min.x, max.x), p.y.clamp(min.y, max.y));
    let delta = p - nearest;
    let dist = delta.norm();
    if dist >= margin {
        return p;
    }
    if dist > 1e-6 {
        nearest + delta / dist * margin
    } else {
        // On an edge: push along +y or -y, away from the rectangle centre.
        let dir = if p.y >= (min.y + max.y) / 2.0 {
            Vector2::new(0.0, 1.0)
        } else {
            Vector2::new(0.0, -1.0)
        };
        nearest + dir * margin
    }
}

impl AutoReferee {
    /// Place the ball at rest at the given (x, y) position. Used to reset the
    /// ball onto a reachable spot after it leaves the field or after a goal, so
    /// the match keeps running in self-play.
    pub(crate) fn place_ball(&mut self, ctx: &mut RefereeContext<'_>, x: f64, y: f64) {
        ctx.actions.push(RefereeAction::PlaceBall { x, y });
    }

    /// Nudge a desired free-kick ball position to the closest spot that satisfies
    /// the SSL rule 5.3.3 placement constraints: at least 0.2 m from every field
    /// line and at least 1 m from either defense area. A position that is already
    /// valid is returned unchanged.
    pub(crate) fn valid_free_kick_position(
        &self,
        ctx: &mut RefereeContext<'_>,
        desired: Vector2,
    ) -> Vector2 {
        valid_free_kick_position(&ctx.config.field_geometry, desired)
    }
}

pub(crate) fn valid_free_kick_position(
    geom: &dies_core::FieldGeometry,
    desired: Vector2,
) -> Vector2 {
    let hl = geom.field_length / 2.0;
    let hw = geom.field_width / 2.0;
    let half_pa_w = geom.penalty_area_width / 2.0;

    let mut pos = desired;

    // Push out of the 1 m keep-out zone around each defense area (one per
    // goal line). The areas are 2× penalty_area_depth apart in x, so their
    // keep-out zones never overlap and the pushes don't fight each other.
    //
    // Extend each rectangle *behind* its goal line (well off the field) so the
    // goal-line face is never the nearest exit. Otherwise a ball that went out
    // over the goal line — closer to the goal line than to the front edge or
    // the sides — would be pushed back through the goal (off the field), and
    // the field-line clamp below would drag it straight back inside the box,
    // leaving the free kick at an illegal in-area position. The only valid
    // exits are the front (field-facing) edge and the two sides.
    for &goal_x in &[-hl, hl] {
        let inner_x = goal_x - geom.penalty_area_depth * goal_x.signum();
        let outer_x = goal_x + 10_000.0 * goal_x.signum();
        let min = Vector2::new(inner_x.min(outer_x), -half_pa_w);
        let max = Vector2::new(inner_x.max(outer_x), half_pa_w);
        pos = push_out_of_rect(pos, min, max, FREE_KICK_DEFENSE_AREA_DISTANCE);
    }

    // Keep the ball at least 0.2 m inside every field line.
    pos.x = pos.x.clamp(
        -(hl - FREE_KICK_PLACEMENT_MARGIN),
        hl - FREE_KICK_PLACEMENT_MARGIN,
    );
    pos.y = pos.y.clamp(
        -(hw - FREE_KICK_PLACEMENT_MARGIN),
        hw - FREE_KICK_PLACEMENT_MARGIN,
    );
    pos
}
