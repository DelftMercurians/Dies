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

use dies_core::{PlayerId, PlayerSkillInfo, SkillState};
use dies_strategy_protocol::{
    PassBallState, PassFailure, PassResult, PassRole, SkillCommand, SkillStatus,
};

use super::pass_coordinator::{PassContext, PassCoordinator};
use super::skill_executor::skill_state_from_status;
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
    /// A pass with the given partner, role, and per-pass sequence number
    /// (`0` = no nonce; see [`SkillCommand::Pass`]).
    Pass {
        partner: PlayerId,
        role: PassRole,
        seq: u64,
    },
    /// Stop immediately.
    Stop,
    /// Some other (single-robot) skill — leaves any joint coordination.
    Other,
}

fn classify(cmd: Option<&Option<SkillCommand>>) -> SlotReq {
    match cmd {
        None | Some(None) => SlotReq::Continue,
        Some(Some(SkillCommand::Pass {
            partner, role, seq, ..
        })) => SlotReq::Pass {
            partner: *partner,
            role: *role,
            seq: *seq,
        },
        Some(Some(SkillCommand::Stop)) => SlotReq::Stop,
        Some(Some(_)) => SlotReq::Other,
    }
}

/// Whether two per-pass sequence numbers refer to the same pass. `0` is the
/// "no nonce" sentinel and matches anything (legacy behavior).
fn seq_matches(a: u64, b: u64) -> bool {
    a == 0 || b == 0 || a == b
}

fn opposite(role: PassRole) -> PassRole {
    match role {
        PassRole::Passer => PassRole::Receiver,
        PassRole::Receiver => PassRole::Passer,
    }
}

/// A live coordinator plus the per-pass sequence number it was created from
/// (`0` = created from nonce-less commands).
struct Slot {
    seq: u64,
    coord: PassCoordinator,
}

/// Manages all active pass coordinators.
pub struct JointSkillExecutor {
    coordinators: Vec<Slot>,
    /// Last reported joint status per player (persists like `SkillExecutor`).
    statuses: HashMap<PlayerId, SkillStatus>,
    /// Last rich result per player (persists until the player leaves the pass).
    results: HashMap<PlayerId, PassResult>,
    /// Sequence number of the last FINISHED pass per (passer, receiver) pair.
    /// Commands still carrying this seq are a stale re-send (the strategy hasn't
    /// re-tasked the robots yet) — they must not resurrect the pass, and above
    /// all must not wipe its un-consumed terminal status/result.
    finished_seqs: HashMap<(PlayerId, PlayerId), u64>,
}

impl JointSkillExecutor {
    pub fn new() -> Self {
        Self {
            coordinators: Vec::new(),
            statuses: HashMap::new(),
            results: HashMap::new(),
            finished_seqs: HashMap::new(),
        }
    }

    fn coordinator_of(&self, id: PlayerId) -> Option<usize> {
        self.coordinators
            .iter()
            .position(|s| s.coord.passer() == id || s.coord.receiver() == id)
    }

    /// Reconcile against the command map and tick every live coordinator once.
    pub fn tick_all(
        &mut self,
        commands: &HashMap<PlayerId, Option<SkillCommand>>,
        ctx: &PassContext<'_>,
    ) -> JointOutput {
        // --- 1. Liveness / cancellation over existing coordinators. ---
        for slot in self.coordinators.iter_mut() {
            let coord = &mut slot.coord;
            let passer = coord.passer();
            let receiver = coord.receiver();
            let mut cancel: Option<(PassFailure, PassBallState)> = None;
            for (member, other) in [(passer, receiver), (receiver, passer)] {
                // A released passer's slot belongs to the per-player executor
                // again; its commands no longer affect the pass.
                if member == passer && coord.passer_released() {
                    continue;
                }
                let leaving = match classify(commands.get(&member)) {
                    SlotReq::Continue => None,
                    // Same partner AND same pass (seq): still ours. A different
                    // seq is a NEW pass decision between the same pair — cancel
                    // so step 2 can create the fresh coordinator.
                    SlotReq::Pass { partner, seq, .. }
                        if partner == other && seq_matches(seq, slot.seq) =>
                    {
                        None
                    }
                    SlotReq::Pass { .. } | SlotReq::Other => {
                        Some((PassFailure::PartnerLeft, PassBallState::Unknown))
                    }
                    SlotReq::Stop => Some((PassFailure::Cancelled, PassBallState::Unknown)),
                };
                if let Some((reason, ball_state)) = leaving {
                    if member == passer && coord.in_settle() {
                        // Settle: the passer's job ended at Flight — release it
                        // alone; the receiver handover continues. (A receiver
                        // command change goes through force_terminate, which
                        // retires the pass but never overwrites a published
                        // Success — the handover-on-command release.)
                        coord.release_passer();
                    } else {
                        cancel = Some((reason, ball_state));
                    }
                }
            }
            if let Some((reason, ball_state)) = cancel {
                coord.force_terminate(reason, ball_state);
            } else if let Some(Some(SkillCommand::Pass {
                target_hint,
                forward_to,
                ..
            })) = commands.get(&passer)
            {
                // Refresh the hints from the passer's command if present.
                if !coord.passer_released() {
                    coord.update_hints(*target_hint, *forward_to);
                }
            }
        }

        // --- 2. Create new coordinators for fresh, mutually-matched requests. ---
        let mut covered: HashSet<PlayerId> = HashSet::new();
        for s in &self.coordinators {
            if !s.coord.passer_released() {
                covered.insert(s.coord.passer());
            }
            covered.insert(s.coord.receiver());
        }
        let mut managed: HashSet<PlayerId> = covered.clone();

        // Gather this frame's pass requests in a deterministic order.
        let mut requests: Vec<(PlayerId, PlayerId, PassRole, u64)> = commands
            .iter()
            .filter_map(|(id, cmd)| match cmd {
                Some(SkillCommand::Pass {
                    partner, role, seq, ..
                }) => Some((*id, *partner, *role, *seq)),
                _ => None,
            })
            .collect();
        requests.sort_by_key(|(id, _, _, _)| *id);

        for (id, partner, role, seq) in requests {
            managed.insert(id);
            if covered.contains(&id) {
                continue;
            }
            // The partner must request a matching pass back (same pass: seq too).
            let matched = matches!(
                classify(commands.get(&partner)),
                SlotReq::Pass { partner: back, role: prole, seq: pseq }
                    if back == id && prole == opposite(role) && seq_matches(seq, pseq)
            );
            if matched {
                // Create exactly once, from the lower id.
                if id < partner {
                    let (p, r) = match role {
                        PassRole::Passer => (id, partner),
                        PassRole::Receiver => (partner, id),
                    };
                    if seq != 0 && self.finished_seqs.get(&(p, r)) == Some(&seq) {
                        // Stale re-send of a pass that already finished (the
                        // strategy hasn't re-tasked the robots yet): do NOT
                        // resurrect it — and critically, do NOT wipe the
                        // terminal status/result it left behind.
                        continue;
                    }
                    let (target_hint, forward_to) = match commands.get(&id) {
                        Some(Some(SkillCommand::Pass {
                            target_hint,
                            forward_to,
                            ..
                        })) => (*target_hint, *forward_to),
                        _ => (None, None),
                    };
                    self.coordinators.push(Slot {
                        seq,
                        coord: PassCoordinator::new(p, r, target_hint, forward_to),
                    });
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
        for slot in self.coordinators.iter_mut() {
            let coord = &mut slot.coord;
            let passer = coord.passer();
            let receiver = coord.receiver();
            let out = coord.tick(ctx);
            // A released passer belongs to the per-player executor again — no
            // input, and no status/result refresh that step 4 would then have
            // to fight.
            if !coord.passer_released() {
                inputs.insert(passer, out.passer_input);
                self.statuses.insert(passer, out.status);
            }
            inputs.insert(receiver, out.receiver_input);
            self.statuses.insert(receiver, out.status);
            if let Some(result) = coord.result() {
                if !coord.passer_released() {
                    self.results.insert(passer, result);
                }
                self.results.insert(receiver, result);
            }
        }

        // Retire finished coordinators (their status/result persist in the
        // maps), recording the finished seq so stale re-sends of the same pass
        // can't resurrect it in a later frame's step 2.
        let mut i = 0;
        while i < self.coordinators.len() {
            if self.coordinators[i].coord.is_done() {
                let s = self.coordinators.swap_remove(i);
                if s.seq != 0 {
                    self.finished_seqs
                        .insert((s.coord.passer(), s.coord.receiver()), s.seq);
                }
            } else {
                i += 1;
            }
        }

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

    /// Rich, UI-facing skill info for every player involved in a pass.
    ///
    /// Live coordinators report their current phase and role; players whose
    /// coordinator has ended (but whose terminal status lingers) report the
    /// retained status with a generic description.
    pub fn get_all_infos(&self) -> HashMap<PlayerId, PlayerSkillInfo> {
        let mut infos = HashMap::new();
        for slot in &self.coordinators {
            let coord = &slot.coord;
            let passer = coord.passer();
            let receiver = coord.receiver();
            let phase = coord.phase().as_str();
            for (id, role) in [(passer, "Passer"), (receiver, "Receiver")] {
                if id == passer && coord.passer_released() {
                    continue;
                }
                let state = self
                    .statuses
                    .get(&id)
                    .copied()
                    .map(skill_state_from_status)
                    .unwrap_or(SkillState::Running);
                infos.insert(
                    id,
                    PlayerSkillInfo {
                        skill_type: "Pass".to_string(),
                        state,
                        description: format!(
                            "{role} p{}→p{}: {phase}",
                            passer.as_u32(),
                            receiver.as_u32()
                        ),
                    },
                );
            }
        }
        // Players with a lingering terminal status but no live coordinator.
        for (id, status) in &self.statuses {
            infos.entry(*id).or_insert_with(|| PlayerSkillInfo {
                skill_type: "Pass".to_string(),
                state: skill_state_from_status(*status),
                description: "pass ended".to_string(),
            });
        }
        infos
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

    /// Test-only view of the live coordinator for a player.
    #[cfg(test)]
    fn coord_of(&self, id: PlayerId) -> Option<&PassCoordinator> {
        self.coordinator_of(id).map(|i| &self.coordinators[i].coord)
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
        pass_seq(partner, role, 0)
    }

    fn pass_seq(partner: u32, role: PassRole, seq: u64) -> Option<SkillCommand> {
        Some(SkillCommand::Pass {
            partner: PlayerId::new(partner),
            role,
            target_hint: None,
            forward_to: None,
            seq,
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
        // Free ball near the passer: the pass stays alive in Stage (the passer
        // never holds), so the reassignment can orphan it.
        let w = world(
            vec![
                player(0, Vector2::new(0.0, 0.0), 0.0),
                player(1, Vector2::new(2000.0, 0.0), 0.0),
            ],
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
                hold_ground: false,
                avoid_robots: true,
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

    /// World with the ball missing entirely: any fresh coordinator fails its
    /// entry guard with `BallLost`/`Unknown` on its first tick — a cheap way to
    /// drive a pass to a terminal result in one frame.
    fn no_ball_world() -> dies_core::TeamData {
        world(
            vec![
                player(0, Vector2::new(0.0, 0.0), 0.0),
                player(1, Vector2::new(2000.0, 0.0), 0.0),
            ],
            None,
            0.016,
        )
    }

    #[test]
    fn stale_seq_commands_do_not_resurrect_finished_pass() {
        // The wipe-bug regression: after a pass finishes, the strategy's stale
        // Pass commands (same seq) arrive for one more frame. They must neither
        // recreate the coordinator nor wipe the terminal status/result.
        let mut jse = JointSkillExecutor::new();
        let w = no_ball_world();
        let tc = team_ctx();
        let ctx = || PassContext {
            world: &w,
            team_context: &tc,
        };
        let mut cmds = HashMap::new();
        cmds.insert(PlayerId::new(0), pass_seq(1, PassRole::Passer, 7));
        cmds.insert(PlayerId::new(1), pass_seq(0, PassRole::Receiver, 7));

        // Frame 1: created, fails the entry guard, retires.
        jse.tick_all(&cmds, &ctx());
        assert!(jse.coord_of(PlayerId::new(0)).is_none());
        assert_eq!(jse.get_status(PlayerId::new(0)), Some(SkillStatus::Failed));

        // Frame 2: identical stale commands → suppressed, result preserved.
        jse.tick_all(&cmds, &ctx());
        assert!(
            jse.coord_of(PlayerId::new(0)).is_none(),
            "stale same-seq commands must not resurrect the pass"
        );
        assert!(
            matches!(
                jse.pass_results().get(&PlayerId::new(0)),
                Some(PassResult::Failure {
                    reason: PassFailure::BallLost,
                    ..
                })
            ),
            "the un-consumed terminal result must survive stale re-sends"
        );

        // A NEW pass decision (fresh seq) between the same pair creates again.
        let mut cmds2 = HashMap::new();
        cmds2.insert(PlayerId::new(0), pass_seq(1, PassRole::Passer, 8));
        cmds2.insert(PlayerId::new(1), pass_seq(0, PassRole::Receiver, 8));
        jse.tick_all(&cmds2, &ctx());
        // It fails its entry guard again (no ball), but it DID run: the stale
        // suppression applies only to seq 7.
        assert_eq!(
            jse.finished_seqs.get(&(PlayerId::new(0), PlayerId::new(1))),
            Some(&8)
        );
    }

    #[test]
    fn seq_zero_keeps_legacy_recreate_behavior() {
        // Nonce-less strategies (seq 0) keep the old semantics: re-sent
        // commands after a finish recreate the pass (no suppression).
        let mut jse = JointSkillExecutor::new();
        let w = no_ball_world();
        let tc = team_ctx();
        let ctx = || PassContext {
            world: &w,
            team_context: &tc,
        };
        let mut cmds = HashMap::new();
        cmds.insert(PlayerId::new(0), pass(1, PassRole::Passer));
        cmds.insert(PlayerId::new(1), pass(0, PassRole::Receiver));

        jse.tick_all(&cmds, &ctx());
        assert_eq!(jse.get_status(PlayerId::new(0)), Some(SkillStatus::Failed));

        // Frame 2 recreates (and the fresh pass fails again the same way).
        jse.tick_all(&cmds, &ctx());
        assert_eq!(jse.get_status(PlayerId::new(0)), Some(SkillStatus::Failed));
        assert!(jse.finished_seqs.is_empty(), "seq 0 must never be recorded");
    }

    #[test]
    fn different_seq_mid_pass_cancels_for_the_new_pass() {
        // A live pass whose members switch to a NEW seq (a fresh pass decision
        // between the same pair, e.g. an immediate retry) is cancelled so the
        // fresh coordinator can be created the same frame.
        let mut jse = JointSkillExecutor::new();
        // Free ball near the passer keeps the pass alive in Stage.
        let w = world(
            vec![
                player(0, Vector2::new(0.0, 0.0), 0.0),
                player(1, Vector2::new(2000.0, 0.0), 0.0),
            ],
            Some(Vector2::new(60.0, 0.0)),
            0.016,
        );
        let tc = team_ctx();
        let ctx = || PassContext {
            world: &w,
            team_context: &tc,
        };
        let mut cmds = HashMap::new();
        cmds.insert(PlayerId::new(0), pass_seq(1, PassRole::Passer, 3));
        cmds.insert(PlayerId::new(1), pass_seq(0, PassRole::Receiver, 3));
        jse.tick_all(&cmds, &ctx());
        assert!(jse.coord_of(PlayerId::new(0)).is_some());

        let mut cmds2 = HashMap::new();
        cmds2.insert(PlayerId::new(0), pass_seq(1, PassRole::Passer, 4));
        cmds2.insert(PlayerId::new(1), pass_seq(0, PassRole::Receiver, 4));
        // Frame 2 cancels the seq-3 pass (it still covers its players until
        // retirement at the end of the frame)...
        jse.tick_all(&cmds2, &ctx());
        assert!(jse.coord_of(PlayerId::new(0)).is_none());
        // ...and frame 3 creates the seq-4 pass — NOT suppressed, because only
        // seq 3 finished.
        jse.tick_all(&cmds2, &ctx());
        assert!(jse.coord_of(PlayerId::new(0)).is_some());
        assert_eq!(jse.get_status(PlayerId::new(0)), Some(SkillStatus::Running));
    }
}
