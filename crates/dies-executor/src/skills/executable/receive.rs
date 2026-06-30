use dies_core::{Angle, Vector2};
use dies_strategy_protocol::{SkillCommand, SkillStatus};
use dies_tunables_macro::tunables;

use crate::control::skill_executor::{ExecutableSkill, SkillContext, SkillProgress};
use crate::control::{PlayerControlInput, Velocity};

tunables! {
    section "Receive";

    /// Minimum closing speed along the pass line before the receiver commits to an
    /// interception. Below this the ball isn't meaningfully on its way, so the
    /// receiver holds the planned intercept point instead of chasing it.
    #[tunable(unit = "mm/s", min = 0.0, max = 1000.0, step = 25.0)]
    MIN_CLOSING_SPEED: f64 = 250.0;
    /// Maximum look-ahead for the interception prediction. A real pass arrives well
    /// within this; a longer predicted crossing time means the ball is barely
    /// converging (or diverging), so the receiver holds rather than chases.
    #[tunable(unit = "s", min = 0.5, max = 5.0, step = 0.1)]
    MAX_PREDICT_TIME: f64 = 2.5;

    /// Gain on the lateral (pass-line normal) correction. `1.0` slides exactly onto
    /// the predicted crossing point; `<1` damps the slide (steadier, risks missing a
    /// fast off-line ball), `>1` over-commits. The result is still clamped to the
    /// capture limit. This is the primary knob for tuning lateral aggressiveness.
    #[tunable(min = 0.0, max = 2.0, step = 0.05)]
    LATERAL_GAIN: f64 = 1.0;

    /// Closing speed below which an on-track pass counts as "slow": rather than wait
    /// at the intercept point, the receiver advances up the pass line to fetch the
    /// ball (so a soft/dying pass doesn't stall short of it).
    #[tunable(unit = "mm/s", min = 0.0, max = 3000.0, step = 50.0)]
    FETCH_SPEED: f64 = 900.0;
    /// Maximum distance the receiver advances toward the passer to fetch a slow ball.
    /// Capped each frame by the ball's remaining distance, so the receiver never
    /// drives past the live ball.
    #[tunable(unit = "mm", min = 0.0, max = 2000.0, step = 50.0)]
    FETCH_REACH: f64 = 700.0;

    /// Closing speed above which an incoming pass counts as "too fast": as it arrives
    /// the receiver backs off along the ball's travel direction to cushion the catch.
    #[tunable(unit = "mm/s", min = 500.0, max = 6000.0, step = 100.0)]
    CUSHION_SPEED: f64 = 2500.0;
    /// Cushion retreat speed per mm/s of closing speed above `CUSHION_SPEED`. `0.4`
    /// gives back 40% of the excess as backward velocity, softening relative impact.
    #[tunable(min = 0.0, max = 1.0, step = 0.05)]
    CUSHION_GAIN: f64 = 0.4;
    /// Depth window before the intercept point within which cushioning engages. The
    /// retreat only fires when the ball is this close to arrival ("on impact"), so it
    /// doesn't pull the receiver back early in flight.
    #[tunable(unit = "mm", min = 100.0, max = 1500.0, step = 50.0)]
    CUSHION_RANGE: f64 = 700.0;
}

fn dkey(ctx: &SkillContext<'_>, tag: &str) -> String {
    format!("p{}.recv.{}", ctx.player.id.as_u32(), tag)
}

/// The pass-line interception solve, shared by [`ReceiveSkill`] and the reflex
/// one-timer receiver ([`super::reflex_receive::ReflexReceiveSkill`]).
pub(super) struct Intercept {
    /// Where to stand this frame: depth anchored at the intercept point, slid along
    /// the pass-line normal onto the predicted crossing (clamped to the capture
    /// limit) and advanced up the line to fetch a slow on-track ball.
    pub position: Vector2,
    /// Closing speed of the ball along the pass line (passer → intercept). `0` when
    /// no genuinely-incoming ball is converging (then `position == intercept_pos`).
    pub closing_speed: f64,
    /// Remaining along-line distance the ball must still travel to reach our depth.
    pub dist_to_target: f64,
    /// Unit pass-line direction (passer → intercept); zero when not converging.
    pub line_dir: Vector2,
}

/// Solve for where to stand to intercept a pass. The receiver anchors its depth at
/// `intercept_pos`, slides along the pass-line normal to where a genuinely-incoming
/// ball will cross our depth (clamped to `capture_limit`, scaled by [`LATERAL_GAIN`]),
/// and advances up the line to fetch a slow on-track ball. Returns a hold at
/// `intercept_pos` (with `closing_speed == 0`) when no ball is meaningfully on its
/// way. `ball` is `(position, velocity)` in mm / mm·s⁻¹.
pub(super) fn solve_intercept(
    from_pos: Vector2,
    intercept_pos: Vector2,
    capture_limit: f64,
    ball: Option<(Vector2, Vector2)>,
) -> Intercept {
    let hold = Intercept {
        position: intercept_pos,
        closing_speed: 0.0,
        dist_to_target: 0.0,
        line_dir: Vector2::zeros(),
    };
    let Some((ball_pos, ball_vel)) = ball else {
        return hold;
    };
    let line_vec = intercept_pos - from_pos;
    if line_vec.norm() < 1.0 {
        // Degenerate geometry (passer ~on the intercept point): hold.
        return hold;
    }
    let line_dir = line_vec.normalize();
    let normal = Vector2::new(-line_vec.y, line_vec.x).normalize();

    // Closing speed of the ball along the pass line (passer -> intercept).
    let closing_speed = ball_vel.dot(&line_dir);
    if closing_speed <= MIN_CLOSING_SPEED() {
        // Ball moving away or not converging: hold.
        return hold;
    }
    // Time for the ball to reach our depth (the normal line through the intercept).
    let to_ball = ball_pos - intercept_pos;
    let t = -to_ball.dot(&line_dir) / closing_speed;
    if !(t > 0.0 && t < MAX_PREDICT_TIME()) {
        // Crossing is in the past or too far out in time: hold.
        return hold;
    }
    // Slide laterally to where the ball will cross our depth.
    let future_ball_pos = ball_pos + ball_vel * t;
    let lateral = ((future_ball_pos - intercept_pos).dot(&normal) * LATERAL_GAIN())
        .clamp(-capture_limit, capture_limit);
    let mut position = intercept_pos + normal * lateral;

    // Remaining distance the ball still has to travel to reach our depth.
    let dist_to_target = (intercept_pos - ball_pos).dot(&line_dir);
    if closing_speed < FETCH_SPEED() {
        // Slow, on-track ball: advance toward the passer to meet it, never
        // overshooting the live ball.
        let frac = ((FETCH_SPEED() - closing_speed) / FETCH_SPEED()).clamp(0.0, 1.0);
        let reach = (frac * FETCH_REACH()).min(dist_to_target.max(0.0));
        position -= line_dir * reach;
    }

    Intercept {
        position,
        closing_speed,
        dist_to_target,
        line_dir,
    }
}

#[derive(Clone)]
pub struct ReceiveSkill {
    from_pos: Vector2,
    target_pos: Vector2,
    capture_limit: f64,
    cushion: bool,
    status: SkillStatus,
}

impl ReceiveSkill {
    pub fn new(from_pos: Vector2, target_pos: Vector2, capture_limit: f64, cushion: bool) -> Self {
        Self {
            from_pos,
            target_pos,
            capture_limit,
            cushion,
            status: SkillStatus::Running,
        }
    }

    /// The pass-line origin this receiver is intercepting from.
    pub fn from_pos(&self) -> Vector2 {
        self.from_pos
    }

    /// Update the geometry in place (used when composed inside the pass
    /// coordinator, which feeds fresh passer/intercept positions each frame).
    pub fn reconfigure(
        &mut self,
        from_pos: Vector2,
        target_pos: Vector2,
        capture_limit: f64,
        cushion: bool,
    ) {
        self.from_pos = from_pos;
        self.target_pos = target_pos;
        self.capture_limit = capture_limit;
        self.cushion = cushion;
    }
}

impl ExecutableSkill for ReceiveSkill {
    fn matches_command(&self, command: &SkillCommand) -> bool {
        matches!(command, SkillCommand::Receive { .. })
    }

    fn update_params(&mut self, command: &SkillCommand) {
        if let SkillCommand::Receive {
            from_pos,
            target_pos,
            capture_limit,
            cushion,
        } = command
        {
            self.from_pos = *from_pos;
            self.target_pos = *target_pos;
            self.capture_limit = *capture_limit;
            self.cushion = *cushion;
        }
    }

    fn tick(&mut self, ctx: SkillContext<'_>) -> SkillProgress {
        if ctx.player.has_ball {
            return SkillProgress::success();
        }

        let mut input = PlayerControlInput::new();
        let current_pos = ctx.player.position.xy();

        // Calculate angle to look at from_pos
        let look_direction = self.from_pos - current_pos;
        let target_heading = Angle::from_radians(look_direction.y.atan2(look_direction.x));

        // Move towards target
        //input.velocity = Velocity::Global(diff.normalize() * 100.0);
        input.with_dribbling(0.6);
        input.with_yaw(target_heading);

        // Position to intercept the pass. The receiver anchors its depth at the
        // planned intercept point, slides laterally (along the pass-line normal) to
        // meet a ball that is genuinely on its way, and additionally adapts its depth
        // to the in-flight ball speed:
        //   - slow but on-track ball -> advance up the line to fetch it
        //   - too-fast ball, near arrival -> retreat along the travel direction to
        //     cushion the impact
        // If the ball is not closing toward us, hold the intercept point.
        let tc = ctx.team_context;
        let ball = ctx
            .world
            .ball
            .as_ref()
            .map(|b| (b.position.xy(), b.velocity.xy()));
        let ix = solve_intercept(self.from_pos, self.target_pos, self.capture_limit, ball);
        tc.debug_value(dkey(&ctx, "closing_speed"), ix.closing_speed);

        // Too-fast ball about to arrive: back off along the travel direction so the
        // relative impact speed is reduced. (Disjoint from the fetch advance — that
        // only fires below FETCH_SPEED, this only above CUSHION_SPEED.)
        let mut cushion_vel = Vector2::zeros();
        if self.cushion && ix.closing_speed > CUSHION_SPEED() && ix.dist_to_target < CUSHION_RANGE()
        {
            let excess = ix.closing_speed - CUSHION_SPEED();
            cushion_vel = ix.line_dir * (excess * CUSHION_GAIN());
            tc.debug_value(dkey(&ctx, "cushion_speed"), cushion_vel.norm());
        }

        input.with_position(ix.position);
        if cushion_vel.norm() > 1.0 {
            // Added to the position-controller output: keeps the receiver drifting
            // backward through the catch.
            input.velocity = Velocity::Global(cushion_vel);
        }

        SkillProgress::Continue(input)
    }

    fn status(&self) -> SkillStatus {
        self.status
    }

    fn skill_type(&self) -> &'static str {
        "Receive"
    }

    fn description(&self) -> String {
        format!(
            "intercepting from ({:.0}, {:.0})",
            self.from_pos.x, self.from_pos.y
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FROM: Vector2 = Vector2::new(0.0, 0.0);
    const INTERCEPT: Vector2 = Vector2::new(2000.0, 0.0);

    #[test]
    fn holds_with_no_ball() {
        let ix = solve_intercept(FROM, INTERCEPT, 500.0, None);
        assert_eq!(ix.position, INTERCEPT);
        assert_eq!(ix.closing_speed, 0.0);
    }

    #[test]
    fn holds_when_ball_not_closing() {
        // Ball moving *away* along the line (−x) is not on its way → hold.
        let ball = Some((Vector2::new(1000.0, 300.0), Vector2::new(-1000.0, 0.0)));
        let ix = solve_intercept(FROM, INTERCEPT, 500.0, ball);
        assert_eq!(ix.position, INTERCEPT);
        assert_eq!(ix.closing_speed, 0.0);
    }

    #[test]
    fn slides_laterally_onto_the_crossing() {
        // Ball at (1000,300) closing along +x at 1000 mm/s crosses our depth in 1 s
        // at y=300 → slide to (2000, 300). 1000 > FETCH_SPEED so no fetch advance.
        let ball = Some((Vector2::new(1000.0, 300.0), Vector2::new(1000.0, 0.0)));
        let ix = solve_intercept(FROM, INTERCEPT, 500.0, ball);
        assert!((ix.position - Vector2::new(2000.0, 300.0)).norm() < 1e-6);
        assert!((ix.closing_speed - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn lateral_slide_is_clamped_to_capture_limit() {
        let ball = Some((Vector2::new(1000.0, 300.0), Vector2::new(1000.0, 0.0)));
        let ix = solve_intercept(FROM, INTERCEPT, 100.0, ball);
        assert!((ix.position - Vector2::new(2000.0, 100.0)).norm() < 1e-6);
    }
}
