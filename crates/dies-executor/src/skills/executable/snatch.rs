use std::time::Duration;

use dies_core::{Angle, PlayerData, PlayerId, Vector2};
use dies_strategy_protocol::{SkillCommand, SkillStatus};
use dies_tunables_macro::tunables;

use crate::control::skill_executor::{ExecutableSkill, SkillContext, SkillProgress};
use crate::control::PlayerControlInput;

tunables! {
    section "Snatch";

    /// Dribbler speed while contesting. Must be > 0 so the simulator counts this
    /// robot as a dribble-claimant (a contester with the dribbler off does nothing).
    #[tunable(min = 0.0, max = 1.0, step = 0.05)]
    DRIBBLER_SPEED: f64 = 0.8;
    /// Max distance from an opponent's centre at which the ball still counts as
    /// "held" by that opponent (centre-to-ball; the held ball sits ~150mm in front).
    #[tunable(unit = "mm", min = 150.0, max = 400.0, step = 10.0)]
    HOLD_RANGE: f64 = 260.0;
    /// Half-angle of the cone in front of the opponent within which the ball counts
    /// as held. Generous — we only need to identify the holder.
    #[tunable(unit = "rad", min = 0.3, max = 1.6, step = 0.05)]
    HOLD_CONE: f64 = 0.9;
    /// Our centre-to-ball distance at the contact pose. Small enough that the ball is
    /// inside our own dribbler cone (so we register as a contester), large enough that
    /// our chassis keeps clear of the opponent's (standoff = this + ~150mm hold offset).
    #[tunable(unit = "mm", min = 120.0, max = 300.0, step = 5.0)]
    CONTACT_DIST: f64 = 190.0;
    /// How close to the contact pose before we commit to pressing + rotating.
    #[tunable(unit = "mm", min = 20.0, max = 200.0, step = 5.0)]
    ENGAGE_TOL: f64 = 80.0;
    /// Heading tolerance to be considered aligned for engagement.
    #[tunable(unit = "rad", min = 0.1, max = 1.2, step = 0.05)]
    ENGAGE_ANGLE: f64 = 0.5;
    /// Slow in-place rotation rate once engaged — the peel itself.
    #[tunable(unit = "rad/s", min = 0.3, max = 3.0, step = 0.1)]
    ROT_RATE: f64 = 1.3;
    /// Speed cap while closing in, so contact stays gentle (no crash foul).
    /// Applies only inside `CAP_ZONE` — the rest of the approach runs at the
    /// controller's normal limits.
    #[tunable(unit = "mm/s", min = 200.0, max = 1500.0, step = 50.0)]
    APPROACH_SPEED_LIMIT: f64 = 700.0;
    /// Ball distance inside which the approach speed cap applies. Outside it,
    /// capping only makes far launches slow (a 3m launch at the cap used to
    /// outlast the whole skill timeout).
    #[tunable(unit = "mm", min = 100.0, max = 1500.0, step = 50.0)]
    CAP_ZONE: f64 = 400.0;
    /// Wall margin scale while contesting (`avoid_robots = false`, walls-only
    /// ORCA). Low so we can strip a ball pinned against the boundary.
    #[tunable(min = 0.0, max = 1.0, step = 0.1)]
    COMMIT_WALL_CARE: f64 = 0.0;
    /// Bail if our centre is within this of a defense area (a non-keeper in our
    /// own box, or touching the ball in either box, is a foul).
    #[tunable(unit = "mm", min = 0.0, max = 400.0, step = 10.0)]
    DEFENSE_BAIL_MARGIN: f64 = 120.0;
    /// Bail if the contested ball is within this of a defense area.
    #[tunable(unit = "mm", min = 0.0, max = 300.0, step = 10.0)]
    BALL_IN_BOX_MARGIN: f64 = 30.0;
}

/// Give up after peeling this long without the opponent losing the ball. The
/// clock starts at *engagement*, not at the first tick — the approach phase is
/// bounded by the planner (a possession flip or capturer switch replans), not
/// by us. Also bounds the never-found-a-holder bail (from the first tick).
const TIMEOUT: Duration = Duration::from_millis(3500);

/// Where a held ball sits ahead of the holder's centre, along its yaw (robot
/// radius + ball radius — same geometry the possession tracker uses). Stands in
/// for the ball while vision has lost it (usual during a hold: the holder
/// occludes it), because the coasted ball estimate goes stale as the holder moves.
const DRIBBLER_OFFSET: f64 = 111.0;

pub struct SnatchSkill {
    /// Where we want the ball to pop loose toward (team-relative). `None` → midfield.
    release_hint: Option<Vector2>,
    status: SkillStatus,
    /// The opponent we locked onto as the ball holder. Locked on first contact so
    /// we don't flip targets if two opponents are near the ball.
    target: Option<PlayerId>,
    /// The possession tracker has confirmed the locked target as the owner
    /// (`has_ball`) at least once. Gates the tracker-driven success below, so a
    /// pipeline race at skill start (tracker still debouncing the acquisition)
    /// can't read as "the holder already lost it".
    target_confirmed: bool,
    /// Whether we've reached the contact pose and started pressing + rotating.
    engaged: bool,
    /// World time (`t_received`, seconds) of the first tick; only bounds the
    /// never-found-a-holder bail. Sim-clock based so the timeout is deterministic
    /// and works under faster-than-realtime sim — `Instant`/wall-clock would fire
    /// at the wrong sim-time. See CLAUDE.md.
    started: Option<f64>,
    /// World time we first reached the contact pose — starts the peel budget.
    engaged_at: Option<f64>,
}

impl SnatchSkill {
    pub fn new(release_hint: Option<Vector2>) -> Self {
        Self {
            release_hint,
            status: SkillStatus::Running,
            target: None,
            target_confirmed: false,
            engaged: false,
            started: None,
            engaged_at: None,
        }
    }

    pub fn set_release_hint(&mut self, release_hint: Option<Vector2>) {
        self.release_hint = release_hint;
    }

    /// Whether `opp` is holding the ball (ball within range and in front of it).
    fn opp_holds_ball(opp: &PlayerData, ball: Vector2) -> bool {
        let rel = ball - opp.position;
        let dist = rel.norm();
        if dist > HOLD_RANGE() || dist < 1e-3 {
            return false;
        }
        let angle = (opp.yaw.to_vector().angle(&rel)).abs();
        angle < HOLD_CONE()
    }
}

impl ExecutableSkill for SnatchSkill {
    fn matches_command(&self, command: &SkillCommand) -> bool {
        matches!(command, SkillCommand::Snatch { .. })
    }

    fn update_params(&mut self, command: &SkillCommand) {
        if let SkillCommand::Snatch { release_hint } = command {
            self.release_hint = *release_hint;
        }
    }

    fn tick(&mut self, ctx: SkillContext<'_>) -> SkillProgress {
        let now = ctx.world.t_received;
        let started = *self.started.get_or_insert(now);

        let Some(ball) = ctx.world.ball.as_ref() else {
            return SkillProgress::Continue(PlayerControlInput::default());
        };
        let ball_est = ball.position.xy();
        let player_pos = ctx.player.position;

        // We came away with the ball ourselves — also a successful strip.
        if ctx.player.has_ball {
            self.status = SkillStatus::Succeeded;
            return SkillProgress::success();
        }

        // Identify / re-locate the holder. Prefer the possession tracker's owner
        // (`has_ball` is stamped from it for opponents too) — it matches the
        // planner's `Steal { from }`; the local cone test only covers the
        // couple-frame IPC skew between the strategy's decision and our first
        // tick. Afterwards track that specific robot.
        let holder = match self.target {
            Some(id) => ctx.world.opp_players.iter().find(|p| p.id == id),
            None => ctx
                .world
                .opp_players
                .iter()
                .find(|p| p.has_ball)
                .or_else(|| {
                    ctx.world
                        .opp_players
                        .iter()
                        .filter(|p| Self::opp_holds_ball(p, ball_est))
                        .min_by(|a, b| {
                            let da = (a.position - ball_est).norm();
                            let db = (b.position - ball_est).norm();
                            da.total_cmp(&db)
                        })
                }),
        };

        let Some(holder) = holder else {
            // No holder. If we'd locked one and it's gone from vision, we treat
            // the strip as done. If we never found one, there's nothing to snatch.
            if self.target.is_some() {
                self.status = SkillStatus::Succeeded;
                return SkillProgress::success();
            }
            if now - started > TIMEOUT.as_secs_f64() {
                self.status = SkillStatus::Failed;
                return SkillProgress::failure();
            }
            return SkillProgress::Continue(PlayerControlInput::default());
        };
        self.target = Some(holder.id);
        if holder.has_ball {
            self.target_confirmed = true;
        }

        // The possession tracker says the target lost the ball → strip succeeded.
        // The tracker is the single source of truth here: it is debounced
        // (`release_frames`), coasts through occlusion, and fast-exits on a kick —
        // an instantaneous geometric test fires spuriously on one noisy frame.
        if self.target_confirmed && !holder.has_ball {
            self.status = SkillStatus::Succeeded;
            return SkillProgress::success();
        }

        // Peel budget. Counted from engagement — the approach phase is the
        // planner's to bound (possession flips and capturer switches replan).
        if let Some(engaged_at) = self.engaged_at {
            if now - engaged_at > TIMEOUT.as_secs_f64() {
                self.status = SkillStatus::Failed;
                return SkillProgress::failure();
            }
        }

        // Where the ball actually is. While occluded (usual during a hold — the
        // holder hides it from vision) the ball estimate coasts stale as the
        // holder moves, so use the holder's dribbler point instead. A *detected*
        // ball wins: it is real contact information (e.g. pushed off-mouth).
        let ball_pos = if ball.detected {
            ball_est
        } else {
            holder.position + holder.yaw.to_vector() * DRIBBLER_OFFSET
        };

        // `a`: unit vector from the holder through the ball (≈ the holder's facing).
        // We stand `CONTACT_DIST` beyond the ball along `a`, on the opposite side
        // from the holder, facing back toward it — so our front meets the ball and
        // our body stays a standoff clear of the opponent chassis.
        let a = {
            let v = ball_pos - holder.position;
            if v.norm() > 1e-3 {
                v.normalize()
            } else {
                (ball_pos - player_pos).normalize()
            }
        };
        let contact_pose = ball_pos + a * CONTACT_DIST();
        let face = Angle::from_vector(-a);

        // Contesting drops robot + defense-box ORCA, so bail rather than illegally
        // enter a defense area (keeper's box, or touching the ball inside one).
        if let Some(field) = ctx.world.field_geom.as_ref() {
            if field.in_defense_area(player_pos, DEFENSE_BAIL_MARGIN())
                || field.in_defense_area(ball_pos, BALL_IN_BOX_MARGIN())
            {
                self.status = SkillStatus::Failed;
                return SkillProgress::failure();
            }
        }

        let mut input = PlayerControlInput::new();
        input.avoid_robots = false; // we deliberately close on the opponent
        input.avoid_ball = false; // we want to touch the ball
        input.wall_care = COMMIT_WALL_CARE(); // …but stay on the field (tight margin)
        input.with_position(contact_pose);
        input.with_dribbling(DRIBBLER_SPEED());

        let to_ball = (ball_pos - player_pos).norm();
        // Gentle contact only needs the cap near the ball; capping the whole
        // approach made any far launch slower than the skill could afford.
        if to_ball < CAP_ZONE() {
            input.with_speed_limit(APPROACH_SPEED_LIMIT());
        }

        let heading_err = (face - ctx.player.yaw).radians().abs();
        let in_contact = to_ball < CONTACT_DIST() + ENGAGE_TOL();
        if !self.engaged {
            self.engaged = in_contact && heading_err < ENGAGE_ANGLE();
            if self.engaged {
                self.engaged_at = Some(now);
            }
        }

        if self.engaged {
            // Pure in-place rotation: command angular velocity (no yaw setpoint) so
            // the robot keeps spinning. Direction chosen so the ball sweeps toward
            // the release hint. The simulator's peel model reads our rotation rate.
            let release = self
                .release_hint
                .unwrap_or_else(|| Vector2::new(0.0, ball_pos.y));
            let g = release - ball_pos;
            // Tangential direction at the ball for a +ω (CCW) sweep around the holder.
            let t_ccw = Vector2::new(-a.y, a.x);
            let sign = if g.dot(&t_ccw) >= 0.0 { 1.0 } else { -1.0 };
            input.angular_velocity = Some(ROT_RATE() * sign);
        } else {
            input.with_yaw(face);
        }

        ctx.team_context.debug_string(
            format!("p{}.snatch", ctx.player.id.as_u32()),
            if self.engaged {
                "peeling"
            } else {
                "approaching"
            },
        );

        self.status = SkillStatus::Running;
        SkillProgress::Continue(input)
    }

    fn status(&self) -> SkillStatus {
        self.status
    }

    fn skill_type(&self) -> &'static str {
        "Snatch"
    }

    fn description(&self) -> String {
        if self.engaged {
            "peeling ball off opponent".to_string()
        } else {
            "approaching held ball".to_string()
        }
    }
}
