//! Rolling friction, dribbler ownership, contested balls, and snatch peeling.
use crate::{
    config::SimulationConfig, physics::PhysicsWorld, realism::gaussian, robot::RobotFleet,
};
use dies_core::{PlayerId, TeamColor, Vector2, Vector3};
use rand::{rngs::StdRng, Rng};
use rapier3d_f64::prelude::*;
// Simulation constants - these are in mm
pub(crate) const BALL_RADIUS: f64 = 21.45;
/// Speed (mm/s) at which a contested ball is squirted out from between two
/// robots that both try to dribble it, so it escapes instead of staying pinned.
const CONTESTED_BALL_POP_SPEED: f64 = 1200.0;
/// Snatch peel: how fast a contesting robot's rotation drags a held ball around
/// the holder's dribbler, as a fraction of the contester's own yaw rate. `1.0` =
/// the ball sweeps as fast as the contester spins; lower makes the strip slower.
const SNATCH_PEEL_GAIN: f64 = 0.6;
/// Speed (mm/s) the ball is flung along the sweep direction when it finally pops
/// out of the holder's dribbler cone. Gentle — it's a strip for a teammate, not a
/// shot.
const SNATCH_RELEASE_SPEED: f64 = 500.0;
/// Max centre-to-ball distance (mm) at which a rotating robot counts as pressing
/// (contesting) a held ball. Distance-gated, not cone-gated, so a robot spinning
/// in place still registers as pressing while its facing sweeps.
const SNATCH_PRESS_RANGE: f64 = 250.0;
pub(crate) struct BallContext<'a> {
    pub(crate) config: &'a SimulationConfig,
    pub(crate) physics: &'a mut PhysicsWorld,
    pub(crate) robots: &'a RobotFleet,
    pub(crate) rng: &'a mut StdRng,
}

pub(crate) struct BallInteraction {
    pub(crate) ball_being_dribbled_by: Option<(PlayerId, TeamColor)>,
    /// Accumulated peel angle (rad) of a held ball being stripped: how far a
    /// contesting robot's rotation has dragged the ball around the holder's
    /// dribbler. Reset to 0 whenever the ball isn't being actively peeled; once it
    /// exceeds the holder's dribbler cone the ball pops loose.
    pub(crate) ball_peel_angle: f64,
}

impl BallInteraction {
    pub(crate) fn new() -> Self {
        Self {
            ball_being_dribbled_by: None,
            ball_peel_angle: 0.0,
        }
    }

    pub(crate) fn resolve(
        &mut self,
        ctx: &mut BallContext<'_>,
        dt: f64,
        dribble_claimants: &[(PlayerId, TeamColor)],
        ball_kicked_this_frame: bool,
    ) {
        // Resolve ball ownership from this frame's claimants. A ball is only
        // cleanly captured when exactly one robot is actively dribbling it.
        //
        // Before the normal resolution we check for an active *snatch*: an
        // established holder (last frame's owner) still gripping the ball, plus a
        // distinct robot pressing its dribbler against the ball and rotating. The
        // contester's rotation drags the ball around the holder's dribbler
        // (`ball_peel_angle` accumulates from the contester's yaw rate) until it
        // leaves the holder's cone and pops loose (handled in the pin block).
        //
        // The contester is detected by *contact distance*, not the dribbler cone:
        // a robot rotating in place sweeps its cone off the ball, but it's still
        // pressing — so cone-gating would drop it mid-spin. Touching without
        // rotating does nothing — only ω peels.
        //
        // With no established holder, several claimants are a genuine loose-ball
        // 50/50: the contact pops it loose (`pop_contested_ball`) as before, which
        // is what stops ownership flipping every frame and deadlocking self-play.
        let prior_holder = self.ball_being_dribbled_by;
        let ball_xy = ctx
            .physics
            .ball
            .as_ref()
            .map(|b| {
                let v = ctx
                    .physics
                    .rigid_body_set
                    .get(b._rigid_body_handle)
                    .unwrap()
                    .position()
                    .translation
                    .vector;
                Vector2::new(v.x, v.y)
            })
            .unwrap_or_else(Vector2::zeros);

        // Fastest-rotating presser contesting the holder, if any.
        let snatch_omega = prior_holder
            .filter(|h| !ball_kicked_this_frame && dribble_claimants.contains(h))
            .map(|holder| {
                let mut best = 0.0_f64;
                for p in ctx.robots.players.iter() {
                    if (p.id, p.team_color) == holder || p.current_dribble_speed <= 0.0 {
                        continue;
                    }
                    let rb = ctx.physics.rigid_body_set.get(p.rigid_body_handle).unwrap();
                    let ppos = rb.position().translation.vector;
                    if (ball_xy - Vector2::new(ppos.x, ppos.y)).norm() > SNATCH_PRESS_RANGE {
                        continue;
                    }
                    // The contester's rotation is commanded as angular velocity
                    // (a spin, no heading setpoint), already applied to the body
                    // via `set_angvel` this frame — read it directly. (The body's
                    // orientation itself only integrates in the physics step that
                    // runs after this resolution, so a yaw delta would read zero.)
                    let omega = rb.angvel().z;
                    if omega.abs() > best.abs() {
                        best = omega;
                    }
                }
                (holder, best)
            });

        if ball_kicked_this_frame {
            self.ball_being_dribbled_by = None;
            self.ball_peel_angle = 0.0;
        } else if let Some((holder, omega)) = snatch_omega {
            // Pure-ω: no centerline bias — the skill picks the spin direction to aim
            // where the ball pops loose.
            self.ball_peel_angle += SNATCH_PEEL_GAIN * omega * dt;
            self.ball_being_dribbled_by = Some(holder);
        } else if dribble_claimants.len() == 1 {
            let claimant = dribble_claimants[0];
            // Realism: a *fresh* capture (a reception or pickup) can bump off the
            // dribbler instead of sticking — likelier the faster the ball arrives.
            // A ball already held last frame is never spuriously dropped here.
            let is_fresh = prior_holder != Some(claimant);
            let bump_p = if is_fresh {
                let speed_mps = ctx
                    .physics
                    .ball
                    .as_ref()
                    .map(|b| {
                        ctx.physics
                            .rigid_body_set
                            .get(b._rigid_body_handle)
                            .unwrap()
                            .linvel()
                            .norm()
                    })
                    .unwrap_or(0.0)
                    / 1000.0;
                ctx.config.realism.capture_fail_prob(claimant.1, speed_mps)
            } else {
                0.0
            };
            if bump_p > 0.0 && ctx.rng.gen::<f64>() < bump_p {
                // Bump: deflect the ball outward from the robot (with lateral
                // spread) so it clearly leaves the dribbler and isn't re-grabbed
                // next frame. No capture this frame.
                let robot_xy = ctx
                    .robots
                    .players
                    .iter()
                    .find(|p| (p.id, p.team_color) == claimant)
                    .map(|p| {
                        let v = ctx
                            .physics
                            .rigid_body_set
                            .get(p.rigid_body_handle)
                            .unwrap()
                            .position()
                            .translation
                            .vector;
                        Vector2::new(v.x, v.y)
                    })
                    .unwrap_or(ball_xy);
                let mut out = ball_xy - robot_xy;
                if out.norm() < 1e-6 {
                    out = Vector2::new(1.0, 0.0);
                }
                let base = out.y.atan2(out.x);
                let spread = gaussian(&mut ctx.rng, 0.5);
                let ang = base + spread;
                if let Some(ball) = ctx.physics.ball.as_ref() {
                    let bb = ctx
                        .physics
                        .rigid_body_set
                        .get_mut(ball._rigid_body_handle)
                        .unwrap();
                    bb.set_linvel(
                        Vector::new(
                            ang.cos() * ctx.config.realism.bump_speed,
                            ang.sin() * ctx.config.realism.bump_speed,
                            0.0,
                        ),
                        true,
                    );
                }
                self.ball_being_dribbled_by = None;
                self.ball_peel_angle = 0.0;
            } else {
                self.ball_being_dribbled_by = Some(claimant);
                self.ball_peel_angle = 0.0;
            }
        } else {
            if dribble_claimants.len() > 1 {
                self.pop_contested_ball(ctx, &dribble_claimants);
            }
            self.ball_being_dribbled_by = None;
            self.ball_peel_angle = 0.0;
        }

        if let Some((player_id, team_color)) = self.ball_being_dribbled_by {
            // Read the holder's pose into locals so the borrow ends before we mutate
            // the ball body and the peel state.
            let (player_position, yaw_angle) = {
                let player = ctx
                    .robots
                    .players
                    .iter()
                    .find(|p| p.id == player_id && p.team_color == team_color)
                    .unwrap();
                let rb = ctx
                    .physics
                    .rigid_body_set
                    .get(player.rigid_body_handle)
                    .unwrap();
                (
                    rb.position().translation.vector,
                    rb.rotation().euler_angles().2,
                )
            };

            // Pin the ball to the dribbler, rotated by the accumulated peel angle.
            // At peel 0 this is the holder's dribbler centerline (the normal hold).
            let base_offset = ctx.config.player_radius + ctx.config.dribbler_radius - 80.0;
            let hold_angle = yaw_angle + self.ball_peel_angle;
            let dir = Vector::new(hold_angle.cos(), hold_angle.sin(), 0.0);
            let dribbler_position = player_position + dir * base_offset;
            let released = self.ball_peel_angle.abs() > ctx.config.dribbler_angle;

            let ball_body = ctx
                .physics
                .rigid_body_set
                .get_mut(ctx.physics.ball.as_ref().unwrap()._rigid_body_handle)
                .unwrap();
            ball_body.set_position(
                Isometry::translation(
                    dribbler_position.x,
                    dribbler_position.y,
                    dribbler_position.z,
                ),
                true,
            );
            if released {
                // Ball swept past the holder's dribbler cone: it pops loose, flung
                // along the sweep (the direction the contester aimed it) plus a
                // radial-outward component so it clearly leaves the holder's grip
                // and isn't re-claimed on the next frame.
                let tangential = Vector::new(-dir.y, dir.x, 0.0) * self.ball_peel_angle.signum();
                let release_dir = (tangential + dir).normalize();
                ball_body.set_linvel(release_dir * SNATCH_RELEASE_SPEED, true);
            } else {
                ball_body.set_linvel(Vector3::zeros(), true);
            }

            if released {
                self.ball_being_dribbled_by = None;
                self.ball_peel_angle = 0.0;
            }
        }
    }

    /// Resolve a contested ball: when two or more robots try to dribble the same
    /// ball, none can hold it. Squirt the ball out perpendicular to the line
    /// joining the first two contenders so it escapes the pile instead of
    /// staying pinned between them (which would deadlock a 50/50).
    fn pop_contested_ball(
        &mut self,
        ctx: &mut BallContext<'_>,
        claimants: &[(PlayerId, TeamColor)],
    ) {
        let ball_handle = match ctx.physics.ball.as_ref() {
            Some(ball) => ball._rigid_body_handle,
            None => return,
        };

        // Look up the first two contenders' positions.
        let mut positions: Vec<Vector3> = Vec::with_capacity(2);
        for &(id, team) in claimants.iter().take(2) {
            if let Some(player) = ctx
                .robots
                .players
                .iter()
                .find(|p| p.id == id && p.team_color == team)
            {
                if let Some(body) = ctx.physics.rigid_body_set.get(player.rigid_body_handle) {
                    positions.push(body.position().translation.vector);
                }
            }
        }
        if positions.len() < 2 {
            return;
        }
        let (p0, p1) = (positions[0], positions[1]);

        let ball_position = match ctx.physics.rigid_body_set.get(ball_handle) {
            Some(body) => body.position().translation.vector,
            None => return,
        };

        // Escape perpendicular to the squeeze axis between the two robots.
        let axis = p1 - p0;
        let mut escape = Vector3::new(-axis.y, axis.x, 0.0);
        if escape.norm() < 1e-6 {
            escape = Vector3::new(1.0, 0.0, 0.0);
        }
        let mut escape = escape.normalize();
        // Send it toward whichever side the ball already leans, so the pop is
        // stable frame-to-frame instead of fighting itself.
        let midpoint = (p0 + p1) * 0.5;
        if escape.dot(&(ball_position - midpoint)) < 0.0 {
            escape = -escape;
        }

        let ball_body = ctx.physics.rigid_body_set.get_mut(ball_handle).unwrap();
        ball_body.set_linvel(escape * CONTESTED_BALL_POP_SPEED, true);
    }
}

pub(crate) fn apply_rolling_friction(physics: &mut PhysicsWorld, config: &SimulationConfig) {
    // Reset ball forces + apply rolling friction
    if let Some(ball) = physics.ball.as_ref() {
        let ball_body = physics
            .rigid_body_set
            .get_mut(ball._rigid_body_handle)
            .unwrap();
        ball_body.reset_forces(true);
        ball_body.set_linear_damping(config.ball_damping);
        // Constant-decel rolling friction on the planar velocity (see the
        // `ball_roll_decel` doc — proportional damping has the wrong decay
        // shape). Applied regardless of the micro-bounce z-state; airborne
        // balls are rare and flat kicks dominate.
        let v = *ball_body.linvel();
        let speed = (v.x * v.x + v.y * v.y).sqrt();
        if speed > 1e-6 {
            let dec = config.ball_roll_decel * physics.integration_parameters.dt;
            let scale = (speed - dec).max(0.0) / speed;
            ball_body.set_linvel(Vector::new(v.x * scale, v.y * scale, v.z), true);
        }
    }
}
