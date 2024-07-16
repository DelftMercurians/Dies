use dies_core::PlayerId;

use crate::roles::{
    kicker_role::KickerRole,
    passer::{self, Passer},
    receiver::{self, Receiver},
    Role,
};

use super::{Strategy, StrategyCtx};

struct Assignmmet {
    passer_id: PlayerId,
    receiver_id: PlayerId,
    passer: Passer,
    receiver: Receiver,
}

impl Assignmmet {
    fn new(passer_id: PlayerId, receiver_id: PlayerId) -> Self {
        Self {
            passer_id,
            receiver_id,
            passer: Passer::new(receiver_id),
            receiver: Receiver::new(),
        }
    }
}

pub struct TestStrat {
    assignment: Option<Assignmmet>,
}

impl TestStrat {
    pub fn new() -> Self {
        Self { assignment: None }
    }
}

impl Strategy for TestStrat {
    fn update(&mut self, ctx: StrategyCtx) {
        let assignment = {
            if let Some(assignment) = self.assignment.as_mut() {
                assignment
            } else {
                // Assign roles
                let available_ids = ctx
                    .world
                    .own_players
                    .iter()
                    .map(|player| player.id)
                    .collect::<Vec<_>>();
                if available_ids.len() < 2 {
                    return;
                }
                self.assignment.get_or_insert_with(|| {
                    let passer_id = available_ids[0];
                    let receiver_id = available_ids[1];
                    Assignmmet::new(passer_id, receiver_id)
                })
            }
        };

        if assignment.passer.has_passed() {
            assignment.receiver.set_passer_kicked();
        }
    }

    fn update_role(
        &mut self,
        player_id: PlayerId,
        ctx: crate::roles::RoleCtx,
    ) -> Option<crate::PlayerControlInput> {
        if let Some(assignment) = self.assignment.as_mut() {
            if player_id == assignment.passer_id {
                return Some(assignment.passer.update(ctx));
            } else if player_id == assignment.receiver_id {
                return Some(assignment.receiver.update(ctx));
            }
        }
        None
    }
}
