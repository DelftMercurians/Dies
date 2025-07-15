use dies_core::{GameState, Handicap};
use dies_executor::behavior_tree_api::GameContext;

use crate::v0::{
    freekick_interference::score_free_kick_interference,
    freekick_kicker::{score_free_kick_kicker, score_kickoff_kicker},
    harasser::score_as_harasser,
    penalty_kicker::score_penalty_kicker,
    striker::score_striker,
    waller::score_as_waller,
};

mod utils;

mod freekick_interference;
mod freekick_kicker;
mod harasser;
mod keeper;
mod kickoff_kicker;
mod penalty_kicker;
mod striker;
mod waller;

pub fn v0_strategy(game: &mut GameContext) {
    game.add_role("harasser")
        .count(1)
        .score(|_| 100000.0)
        .behavior(|s| harasser::build_harasser_tree(s));

    // Goalkeeper - always exactly one
    game.add_role("goalkeeper")
        .count(1)
        .score(|_| 100000.0)
        .behavior(|s| keeper::build_goalkeeper_tree(s));

    // Game state specific roles
    let game_state = game.game_state();
    match game_state.game_state {
        GameState::Kickoff | GameState::PrepareKickoff => {
            if game_state.us_operating {
                // Kickoff kicker
                game.add_role("kickoff_kicker")
                    .count(1)
                    .exclude(|s| s.has_handicap(Handicap::NoKicker))
                    .score(score_kickoff_kicker)
                    .behavior(|s| kickoff_kicker::build_kickoff_kicker_tree(s));
            }
        }
        GameState::FreeKick => {
            if game_state.us_operating {
                // Free kick kicker
                game.add_role("free_kick_kicker")
                    .count(1)
                    .exclude(|s| s.has_handicap(Handicap::NoKicker))
                    .score(score_free_kick_kicker)
                    .behavior(|s| freekick_kicker::build_free_kick_kicker_tree(s));
            } else {
                // Free kick interference
                game.add_role("free_kick_interference")
                    .min(1)
                    .max(2)
                    .score(score_free_kick_interference)
                    .behavior(|s| freekick_interference::build_free_kick_interference_tree(s));
            }
        }
        GameState::Penalty | GameState::PreparePenalty | GameState::PenaltyRun => {
            if game_state.us_operating {
                // Penalty kicker
                game.add_role("penalty_kicker")
                    .count(1)
                    .exclude(|s| s.has_handicap(Handicap::NoKicker))
                    .score(score_penalty_kicker)
                    .behavior(|s| penalty_kicker::build_penalty_kicker_tree(s));
            }
        }
        _ => {}
    }

    // Standard roles
    game.add_role("striker")
        .max(2)
        .score(score_striker)
        .exclude(|s| s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]))
        .behavior(|s| striker::build_striker_tree(s));

    game.add_role("waller")
        .min(0)
        .max(2)
        .score(score_as_waller)
        .behavior(|s| waller::build_waller_tree(s));

    game.add_role("harasser")
        .min(2)
        .max(2)
        .score(score_as_harasser)
        .behavior(|s| harasser::build_harasser_tree(s));
}
