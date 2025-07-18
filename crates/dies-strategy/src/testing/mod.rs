use dies_core::{GameState, Handicap};
use dies_executor::behavior_tree_api::GameContext;

use crate::v0::{
    freekick_kicker, kickoff_kicker,
    penalty_kicker::{self, score_penalty_kicker},
    striker::{self, score_striker},
};

mod test_fetchball;
mod test_movement;
mod test_passer;
mod test_receiver;

pub fn testing_strategy(game: &mut GameContext) {
    let game_state = game.game_state();
    match game_state.game_state {
        GameState::Kickoff | GameState::PrepareKickoff => {
            if game_state.us_operating {
                // Kickoff kicker
                game.add_role("kickoff_kicker")
                    .count(1)
                    .exclude(|s| s.player_id.as_u32() == 4)
                    .exclude(|s| s.has_handicap(Handicap::NoKicker))
                    .score(|s| 10.0)
                    .behavior(|s| kickoff_kicker::build_kickoff_kicker_tree(s));
            }
        }
        GameState::FreeKick => {
            if game_state.us_operating {
                // Free kick kicker
                game.add_role("free_kick_kicker")
                    .count(1)
                    .exclude(|s| s.has_handicap(Handicap::NoKicker))
                    .exclude(|s| s.player_id.as_u32() == 4)
                    .score(|s| if s.am_closest_to_ball() { 5.0 } else { 0.0 })
                    .behavior(|s| freekick_kicker::build_free_kick_kicker_tree(s));
            }
            // else {
            //     // Free kick interference
            //     game.add_role("free_kick_interference")
            //         .exclude(|s| s.player_id.as_u32() == 4)
            //         .max(1)
            //         .score(score_free_kick_interference)
            //         .behavior(|s| freekick_interference::build_free_kick_interference_tree(s));
            // }
        }
        GameState::Penalty | GameState::PreparePenalty | GameState::PenaltyRun => {
            if game_state.us_operating {
                // Penalty kicker
                game.add_role("penalty_kicker")
                    .count(1)
                    .exclude(|s| s.player_id.as_u32() == 4)
                    .exclude(|s| s.has_handicap(Handicap::NoKicker))
                    .score(score_penalty_kicker)
                    .behavior(|s| penalty_kicker::build_penalty_kicker_tree(s));
            }
        }
        _ => {}
    }

    game.add_role("striker_1")
        // .max(1)
        .score(score_striker)
        // .exclude(|s| s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]))
        .behavior(|s| striker::build_striker_tree(s));
}
