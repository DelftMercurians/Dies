//! Joint skill execution — the first-class home of multi-robot coordinated
//! skills (currently just passing).
//!
//! Unlike [`SkillExecutor`](super::skill_executor::SkillExecutor), which owns one
//! skill per robot, the [`JointSkillExecutor`] owns coordinators that each claim
//! TWO robots. A coordinator is ticked **once per frame** (not once per member),
//! which is what makes its state machine debuggable as a single entity.
//!
//! ## Claiming has ordinary skill semantics
//!
//! A pass lives in the skill slot of both robots as a [`SkillCommand::Pass`]. The
//! strategy keeps it active simply by not replacing it, and cancels it by
//! commanding anything else on either slot — exactly like any other skill. There
//! is no separate claim system.
//!
//! ## Clean release via liveness
//!
//! Every frame each coordinator verifies BOTH of its slots still reference it. If
//! one side was reassigned (a different command) or stopped, the orphaned pass is
//! terminated with a typed reason and both robots are released. This is the
//! structural guarantee that a robot never gets stuck in an abortive pass.

use std::collections::{HashMap, HashSet};

use dies_core::PlayerId;
use dies_strategy_protocol::{PassBallState, PassFailure, PassResult, PassRole, SkillCommand, SkillStatus};

use super::pass_coordinator::{PassContext, PassCoordinator};
use super::PlayerControlInput;

/// The per-frame output of the joint executor.
pub struct JointOutput {
    /// Control inputs for robots driven by a live coordinator this frame.
    pub inputs: HashMap<PlayerId, PlayerControlInput>,
    /// Robots the joint executor owns this frame — the team controller must skip
    /// these in the per-player `SkillExecutor` (their command is a `Pass`, which
    /// the per-player executor cannot handle).
    pub managed: HashSet<PlayerId>,
}

/// What a robot's command slot is asking for this frame.
enum SlotReq {
    /// Continue whatever is running.
    Continue,
    /// A pass with the given partner and role.
    Pass { partner: PlayerId, role: PassRole },
    /// Stop immediately.
    Stop,
    /// Some other (single-robot) skill — leaves any joint coordination.
    Other,
}

fn classify(cmd: Option<&Option<SkillCommand>>) -> SlotReq {
    match cmd {
        None | Some(None) => SlotReq::Continue,
        Some(Some(SkillCommand::Pass { partner, role, .. })) => SlotReq::Pass {
            partner: *partner,
            role: *role,
        },
        Some(Some(SkillCommand::Stop)) => SlotReq::Stop,
        Some(Some(_)) => SlotReq::Other,
    }
}

fn opposite(role: PassRole) -> PassRole {
    match role {
        PassRole::Passer => PassRole::Receiver,
        PassRole::Receiver => PassRole::Passer,
    }
}

/// Manages all active pass coordinators.
pub struct JointSkillExecutor {
    coordinators: Vec<PassCoordinator>,
    /// Last reported joint status per player (persists like `SkillExecutor`).
    statuses: HashMap<PlayerId, SkillStatus>,
    /// Last rich result per player (persists until the player leaves the pass).
    results: HashMap<PlayerId, PassResult>,
}

impl JointSkillExecutor {
    pub fn new() -> Self {
        Self {
            coordinators: Vec::new(),
            statuses: HashMap::new(),
            results: HashMap::new(),
        }
    }

    fn coordinator_of(&self, id: PlayerId) -> Option<usize> {
        self.coordinators
            .iter()
            .position(|c| c.passer() == id || c.receiver() == id)
    }

    /// Reconcile against the command map and tick every live coordinator once.
    pub fn tick_all(
        &mut self,
        commands: &HashMap<PlayerId, Option<SkillCommand>>,
        ctx: &PassContext<'_>,
    ) -> JointOutput {
        // --- 1. Liveness / cancellation over existing coordinators. ---
        for coord in self.coordinators.iter_mut() {
            let passer = coord.passer();
            let receiver = coord.receiver();
            let mut cancel: Option<(PassFailure, PassBallState)> = None;
            for (member, other) in [(passer, receiver), (receiver, passer)] {
                match classify(commands.get(&member)) {
                    SlotReq::Continue => {}
                    SlotReq::Pass { partner, .. } if partner == other => {}
                    SlotReq::Pass { .. } => {
                        // Re-pointed at a different partner.
                        cancel = Some((PassFailure::PartnerLeft, PassBallState::Unknown));
                    }
                    SlotReq::Other => {
                        cancel = Some((PassFailure::PartnerLeft, PassBallState::Unknown));
                    }
                    SlotReq::Stop => {
                        cancel = Some((PassFailure::Cancelled, PassBallState::Unknown));
                    }
                }
            }
            if let Some((reason, ball_state)) = cancel {
                coord.force_terminate(reason, ball_state);
            } else if let SlotReq::Pass { .. } = classify(commands.get(&passer)) {
                // Refresh the target hint from the passer's command if present.
                if let Some(Some(SkillCommand::Pass { target_hint, .. })) = commands.get(&passer) {
                    coord.update_target_hint(*target_hint);
                }
            }
        }

        // --- 2. Create new coordinators for fresh, mutually-matched requests. ---
        let mut covered: HashSet<PlayerId> = HashSet::new();
        for c in &self.coordinators {
            covered.insert(c.passer());
            covered.insert(c.receiver());
        }
        let mut managed: HashSet<PlayerId> = covered.clone();

        // Gather this frame's pass requests in a deterministic order.
        let mut requests: Vec<(PlayerId, PlayerId, PassRole)> = commands
            .iter()
            .filter_map(|(id, cmd)| match cmd {
                Some(SkillCommand::Pass { partner, role, .. }) => Some((*id, *partner, *role)),
                _ => None,
            })
            .collect();
        requests.sort_by_key(|(id, _, _)| *id);

        for (id, partner, role) in requests {
            managed.insert(id);
            if covered.contains(&id) {
                continue;
            }
            // The partner must request a matching pass back.
            let matched = matches!(
                classify(commands.get(&partner)),
                SlotReq::Pass { partner: back, role: prole }
                    if back == id && prole == opposite(role)
            );
            if matched {
                // Create exactly once, from the lower id.
                if id < partner {
                    let target_hint = match commands.get(&id) {
                        Some(Some(SkillCommand::Pass { target_hint, .. })) => *target_hint,
                        _ => None,
                    };
                    let (p, r) = match role {
                        PassRole::Passer => (id, partner),
                        PassRole::Receiver => (partner, id),
                    };
                    self.coordinators.push(PassCoordinator::new(p, r, target_hint));
                    covered.insert(id);
                    covered.insert(partner);
                    // Fresh pass — drop any stale terminal status/result.
                    self.statuses.remove(&id);
                    self.statuses.remove(&partner);
                    self.results.remove(&id);
                    self.results.remove(&partner);
                }
            } else {
                // Orphaned request (partner never joined / already left). Fail
                // cleanly so the lone robot is released, not stuck.
                self.statuses.insert(id, SkillStatus::Failed);
                self.results.insert(
                    id,
                    PassResult::Failure {
                        reason: PassFailure::PartnerLeft,
                        ball_state: PassBallState::Unknown,
                    },
                );
            }
        }

        // --- 3. Tick every coordinator once; collect inputs/status/result. ---
        let mut inputs: HashMap<PlayerId, PlayerControlInput> = HashMap::new();
        for coord in self.coordinators.iter_mut() {
            let passer = coord.passer();
            let receiver = coord.receiver();
            let out = coord.tick(ctx);
            inputs.insert(passer, out.passer_input);
            inputs.insert(receiver, out.receiver_input);
            self.statuses.insert(passer, out.status);
            self.statuses.insert(receiver, out.status);
            if let Some(result) = coord.result() {
                self.results.insert(passer, result);
                self.results.insert(receiver, result);
            }
        }

        // Retire finished coordinators (their status/result persist in the maps).
        self.coordinators.retain(|c| !c.is_done());

        // --- 4. Clear stale joint state for players that left via a non-pass
        // command (the per-player SkillExecutor owns them now). ---
        for (id, cmd) in commands.iter() {
            match cmd {
                Some(SkillCommand::Pass { .. }) | None => {}
                Some(_) => {
                    if !managed.contains(id) {
                        self.statuses.remove(id);
                        self.results.remove(id);
                    }
                }
            }
        }

        JointOutput { inputs, managed }
    }

    /// Joint status for a player, if the joint executor has an opinion.
    pub fn get_status(&self, id: PlayerId) -> Option<SkillStatus> {
        self.statuses.get(&id).copied()
    }

    /// All joint statuses (to merge over the per-player skill statuses).
    pub fn statuses(&self) -> &HashMap<PlayerId, SkillStatus> {
        &self.statuses
    }

    /// Current rich pass results, keyed by player. Retained until the player
    /// leaves the pass; sent to strategies via the world update.
    pub fn pass_results(&self) -> &HashMap<PlayerId, PassResult> {
        &self.results
    }

    /// Forget a removed player entirely.
    pub fn clear_player(&mut self, id: PlayerId) {
        self.statuses.remove(&id);
        self.results.remove(&id);
        if let Some(idx) = self.coordinator_of(id) {
            // Terminate the whole pass; the partner will be released next frame.
            self.coordinators.swap_remove(idx);
        }
    }
}

impl Default for JointSkillExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::pass_coordinator::test_support::{player, team_ctx, world};
    use dies_core::Vector2;

    fn pass(partner: u32, role: PassRole) -> Option<SkillCommand> {
        Some(SkillCommand::Pass {
            partner: PlayerId::new(partner),
            role,
            target_hint: None,
        })
    }

    #[test]
    fn lone_pass_request_fails_clean() {
        // Player 0 wants to pass to 1, but 1 is doing something else -> 0 is
        // released cleanly with PartnerLeft rather than hanging.
        let mut jse = JointSkillExecutor::new();
        let mut cmds = HashMap::new();
        cmds.insert(PlayerId::new(0), pass(1, PassRole::Passer));
        cmds.insert(PlayerId::new(1), Some(SkillCommand::Stop));

        let w = world(
            vec![
                player(0, Vector2::new(0.0, 0.0), 0.0),
                player(1, Vector2::new(2000.0, 0.0), 0.0),
            ],
            Some(Vector2::new(0.0, 0.0)),
            0.016,
        );
        let tc = team_ctx();
        jse.tick_all(
            &cmds,
            &PassContext {
                world: &w,
                team_context: &tc,
            },
        );
        assert_eq!(jse.get_status(PlayerId::new(0)), Some(SkillStatus::Failed));
        assert!(matches!(
            jse.pass_results().get(&PlayerId::new(0)),
            Some(PassResult::Failure {
                reason: PassFailure::PartnerLeft,
                ..
            })
        ));
    }

    #[test]
    fn matched_requests_create_coordinator() {
        let mut jse = JointSkillExecutor::new();
        let mut cmds = HashMap::new();
        cmds.insert(PlayerId::new(0), pass(1, PassRole::Passer));
        cmds.insert(PlayerId::new(1), pass(0, PassRole::Receiver));

        let w = world(
            vec![
                player(0, Vector2::new(0.0, 0.0), 0.0),
                player(1, Vector2::new(2000.0, 0.0), 0.0),
            ],
            Some(Vector2::new(3000.0, 0.0)),
            0.016,
        );
        let tc = team_ctx();
        let out = jse.tick_all(
            &cmds,
            &PassContext {
                world: &w,
                team_context: &tc,
            },
        );
        assert!(out.managed.contains(&PlayerId::new(0)));
        assert!(out.managed.contains(&PlayerId::new(1)));
        // Both members report the same joint status.
        assert_eq!(
            jse.get_status(PlayerId::new(0)),
            jse.get_status(PlayerId::new(1))
        );
    }

    #[test]
    fn reassigning_one_side_orphans_the_pass() {
        let mut jse = JointSkillExecutor::new();
        // Passer holds the ball so the pass survives Secure into Setup, where the
        // reassignment can orphan it.
        let mut passer = player(0, Vector2::new(0.0, 0.0), 0.0);
        passer.breakbeam_ball_detected = true;
        let w = world(
            vec![passer, player(1, Vector2::new(2000.0, 0.0), 0.0)],
            Some(Vector2::new(60.0, 0.0)),
            0.016,
        );
        let tc = team_ctx();
        let ctx = || PassContext {
            world: &w,
            team_context: &tc,
        };

        // Frame 1: start the pass.
        let mut cmds = HashMap::new();
        cmds.insert(PlayerId::new(0), pass(1, PassRole::Passer));
        cmds.insert(PlayerId::new(1), pass(0, PassRole::Receiver));
        jse.tick_all(&cmds, &ctx());

        // Frame 2: the receiver is reassigned -> the passer's half is orphaned.
        let mut cmds2 = HashMap::new();
        cmds2.insert(PlayerId::new(0), None); // continue
        cmds2.insert(
            PlayerId::new(1),
            Some(SkillCommand::GoToPos {
                position: Vector2::new(-2000.0, 0.0),
                heading: None,
            }),
        );
        jse.tick_all(&cmds2, &ctx());

        assert_eq!(jse.get_status(PlayerId::new(0)), Some(SkillStatus::Failed));
        assert!(matches!(
            jse.pass_results().get(&PlayerId::new(0)),
            Some(PassResult::Failure {
                reason: PassFailure::PartnerLeft,
                ..
            })
        ));
    }
}
