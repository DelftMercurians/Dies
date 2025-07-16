use dies_core::{GameState, Handicap, PlayerId, TeamColor};
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
                    .max(1)
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
    game.add_role("waller_1")
        .max(1)
        .score(score_as_waller)
        .behavior(|s| waller::build_waller_tree(s));

    if game.ball_has_been_on_opp_side_for_at_least(15.0) {
        game.add_role("striker_2")
            .max(1)
            .score(score_striker)
            .exclude(|s| s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]))
            .behavior(|s| striker::build_striker_tree(s));
    } else {
        game.add_role("harasser_1")
            .max(1)
            .score(score_as_harasser)
            .behavior(|s| harasser::build_harasser_tree(s));
    }

    game.add_role("waller_2")
        .max(1)
        .score(score_as_waller)
        .behavior(|s| waller::build_waller_tree(s));

    game.add_role("striker_1")
        .max(1)
        .score(score_striker)
        .exclude(|s| s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]))
        .behavior(|s| striker::build_striker_tree(s));

    if game.ball_has_been_on_our_side_for_at_least(15.0) {
        game.add_role("harasser_2")
            .max(1)
            .score(score_as_harasser)
            .behavior(|s| harasser::build_harasser_tree(s));
    } else {
        game.add_role("striker_3")
            .max(1)
            .score(score_striker)
            .exclude(|s| s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]))
            .behavior(|s| striker::build_striker_tree(s));
    }
}
