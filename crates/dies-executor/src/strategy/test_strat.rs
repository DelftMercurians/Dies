use dies_core::PlayerId;

use crate::roles::{passer::Passer, receiver::Receiver, Role};

use super::{Strategy, StrategyCtx};

struct Assignmmet {
    passer_id: PlayerId,
    receiver_id: PlayerId,
    passer: Passer,
    receiver: Receiver,
    // waller1: Waller,
    // waller1_id: PlayerId,
    // waller2: Waller,
    // waller2_id: PlayerId,
}

impl Assignmmet {
    fn new(
        passer_id: PlayerId,
        receiver_id: PlayerId,
        // waller1_id: PlayerId,
        // waller2_id: PlayerId,
    ) -> Self {
        Self {
            passer_id,
            receiver_id,
            passer: Passer::new(receiver_id),
            receiver: Receiver::new(passer_id),
            // waller1_id,
            // waller1: Waller::new(0.0),
            // waller2_id,
            // waller2: Waller::new(170.0),
        }
    }
}

pub struct TestStrat {
    assignment: Option<Assignmmet>,
}

impl Default for TestStrat {
    fn default() -> Self {
        Self::new()
    }
}

impl TestStrat {
    pub fn new() -> Self {
        Self { assignment: None }
    }
}

impl Strategy for TestStrat {
    fn name(&self) -> &'static str {
        "Test"
    }

    fn update(&mut self, ctx: StrategyCtx) {
        let assignment = {
            if let Some(assignment) = self.assignment.as_mut() {
                assignment
            } else if let Some(ball) = ctx.world.ball.as_ref() {
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
                    // Find available player closest to the ball
                    let passer_id = available_ids
                        .iter()
                        .min_by_key(|id| {
                            let player = ctx.world.get_player(**id).unwrap();
                            (player.position - ball.position.xy()).norm() as i64
                        })
                        .unwrap();
                    let receiver_id = available_ids.iter().find(|id| **id != *passer_id).unwrap();
                    // let waller1_id = available_ids
                    //     .iter()
                    //     .find(|id| **id != *passer_id && **id != *receiver_id)
                    //     .unwrap();
                    // let waller2_id = available_ids
                    //     .iter()
                    //     .find(|id| {
                    //         **id != *passer_id && **id != *receiver_id && **id != *waller1_id
                    //     })
                    //     .unwrap();
                    println!("Passer: {:?}", passer_id);
                    println!("Receiver: {:?}", receiver_id);
                    Assignmmet::new(*passer_id, *receiver_id)
                })
            } else {
                return;
            }
        };

        if assignment.passer.has_passed() {
            assignment.receiver.set_passer_kicked();
        }
    }

    fn get_role(&mut self, player_id: PlayerId, ctx: StrategyCtx) -> Option<&mut dyn Role> {
        if let Some(assignment) = self.assignment.as_mut() {
            if player_id == assignment.passer_id {
                return Some(&mut assignment.passer);
            } else if player_id == assignment.receiver_id {
                return Some(&mut assignment.receiver);
            }
        }
        None
    }
}
