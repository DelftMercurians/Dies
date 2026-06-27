//! `BtRuntime` — the per-frame driver that turns a v0 strategy function into IPC
//! skill commands.
//!
//! Each frame it: ages the ball-side timers, lets the strategy declare its roles,
//! solves the assignment, (re)builds a behavior tree per robot whose role changed,
//! ticks every assigned robot's tree, and applies the resulting `SkillCommand` to
//! that robot's [`PlayerHandle`]. Trees persist across frames so node state
//! (committing guards, fetch phases, sequence cursors) survives.

use std::collections::HashMap;
use std::sync::Arc;

use dies_strategy_api::{PlayerHandle, TeamContext, World};
use dies_strategy_protocol::{PlayerId, SkillCommand, SkillStatus, WorldSnapshot};

use super::game_context::GameContext;
use super::nodes::BehaviorNode;
use super::role_assignment::{Role, RoleAssignmentSolver};
use super::situation::{BtContext, RobotSituation};
use super::Strategy;

enum BallSide {
    Our,
    Opp,
}

pub struct BtRuntime {
    solver: RoleAssignmentSolver,
    bt_context: BtContext,
    /// Per-robot live tree, tagged with the role it was built for.
    trees: HashMap<PlayerId, (String, BehaviorNode)>,
    prev_assignments: HashMap<PlayerId, String>,
    /// Side the ball is currently on and the timestamp it arrived there.
    ball_side_since: Option<(BallSide, f64)>,
}

impl Default for BtRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl BtRuntime {
    pub fn new() -> Self {
        Self {
            solver: RoleAssignmentSolver::new(),
            bt_context: BtContext::new(),
            trees: HashMap::new(),
            prev_assignments: HashMap::new(),
            ball_side_since: None,
        }
    }

    /// Run one frame of the strategy and write skill commands to `ctx`.
    pub fn update(&mut self, strategy: Strategy, world: &World, ctx: &mut TeamContext) {
        let now = world.timestamp();
        let snapshot: Arc<WorldSnapshot> = Arc::new(world.raw_snapshot().clone());

        let (on_our_side, on_opp_side) = self.update_ball_timers(world, now);

        // ── Let the strategy declare its roles ──────────────────────────
        let mut game = GameContext::new(
            world.game_state(),
            world.us_operating(),
            world.our_keeper_id(),
            on_our_side,
            on_opp_side,
        );
        strategy(&mut game);
        let problem = game.into_role_assignment_problem();

        let role_by_name: HashMap<String, Role> = problem
            .roles
            .iter()
            .map(|r| (r.name.clone(), r.clone()))
            .collect();

        // ── Solve the assignment ────────────────────────────────────────
        let active_robots = world.own_player_ids();
        let (assignments, _) = self.solver.solve(
            &problem,
            &active_robots,
            snapshot.clone(),
            Some(&self.prev_assignments),
        );
        let assignments_arc: Arc<HashMap<PlayerId, String>> = Arc::new(assignments.clone());

        // Fresh semaphore claims each frame.
        self.bt_context.begin_frame();

        // Drop trees for robots that lost their assignment.
        self.trees.retain(|id, _| assignments.contains_key(id));

        // ── Tick each assigned robot ────────────────────────────────────
        let mut outputs: Vec<(PlayerId, Option<SkillCommand>)> = Vec::new();
        for (&id, role_name) in assignments.iter() {
            let Some(role) = role_by_name.get(role_name) else {
                continue;
            };

            // (Re)build the tree when the robot is newly assigned this role.
            let needs_build = self
                .trees
                .get(&id)
                .map(|(name, _)| name != role_name)
                .unwrap_or(true);
            if needs_build {
                let build_situation = RobotSituation::new(
                    id,
                    snapshot.clone(),
                    self.bt_context.clone(),
                    assignments_arc.clone(),
                    SkillStatus::Idle,
                );
                let tree = (role.tree_builder)(&build_situation);
                self.trees.insert(id, (role_name.clone(), tree));
            }

            let skill_status = ctx
                .player_ref(id)
                .map(|p| p.skill_status())
                .unwrap_or(SkillStatus::Idle);

            let mut situation = RobotSituation::new(
                id,
                snapshot.clone(),
                self.bt_context.clone(),
                assignments_arc.clone(),
                skill_status,
            );

            let (_, command) = self.trees.get_mut(&id).unwrap().1.tick(&mut situation);
            outputs.push((id, command));
        }

        // ── Apply commands + roles ──────────────────────────────────────
        for (id, command) in outputs {
            if let Some(role) = assignments.get(&id) {
                if let Some(player) = ctx.player(id) {
                    player.set_role(role);
                    if let Some(cmd) = command {
                        apply_command(player, cmd);
                    }
                }
            }
        }

        // Stop robots with no role this frame.
        for &id in &active_robots {
            if !assignments.contains_key(&id) {
                if let Some(player) = ctx.player(id) {
                    player.stop();
                }
            }
        }

        self.prev_assignments = assignments;
    }

    /// Age the ball-side dwell timers; returns `(seconds_on_our_side,
    /// seconds_on_opp_side)`, at most one of which is `Some`.
    fn update_ball_timers(&mut self, world: &World, now: f64) -> (Option<f64>, Option<f64>) {
        let Some(ball) = world.ball_position() else {
            self.ball_side_since = None;
            return (None, None);
        };

        let side = if ball.x < 0.0 {
            BallSide::Our
        } else {
            BallSide::Opp
        };
        let start = match &self.ball_side_since {
            Some((prev, start))
                if matches!(
                    (prev, &side),
                    (BallSide::Our, BallSide::Our) | (BallSide::Opp, BallSide::Opp)
                ) =>
            {
                *start
            }
            _ => now,
        };
        self.ball_side_since = Some((side, start));

        let dwell = (now - start).max(0.0);
        match self.ball_side_since.as_ref().unwrap().0 {
            BallSide::Our => (Some(dwell), None),
            BallSide::Opp => (None, Some(dwell)),
        }
    }
}

/// Translate a `SkillCommand` into the corresponding `PlayerHandle` skill call.
/// This is the only place the BT's command vocabulary meets the IPC skill API.
fn apply_command(player: &mut PlayerHandle, cmd: SkillCommand) {
    match cmd {
        SkillCommand::GoToPos { position, heading } => {
            let b = player.go_to(position);
            if let Some(h) = heading {
                b.with_heading(h);
            }
        }
        SkillCommand::GoToBounded {
            position,
            heading,
            bounds,
        } => {
            let b = player.go_to_bounded(position, bounds);
            if let Some(h) = heading {
                b.with_heading(h);
            }
        }
        SkillCommand::Dribble {
            target_pos,
            target_heading,
        } => {
            player.dribble_to(target_pos, target_heading);
        }
        SkillCommand::PickupBall {
            target_heading,
            instant_kick,
        } => {
            if instant_kick {
                player.pickup_ball_reflex(target_heading);
            } else {
                player.pickup_ball(target_heading);
            }
        }
        SkillCommand::Shoot { target } => {
            player.reflex_shoot(target);
        }
        SkillCommand::DribbleShoot { target_heading } => {
            player.dribble_shoot(target_heading);
        }
        SkillCommand::Receive {
            from_pos,
            target_pos,
            capture_limit,
            cushion,
        } => {
            player.receive(from_pos, target_pos, capture_limit, cushion);
        }
        SkillCommand::Pass { .. } => {
            // Joint passes are coordinated via TeamContext::pass, not produced by
            // the v0 trees; ignore.
        }
        SkillCommand::Stop => player.stop(),
    }
}
