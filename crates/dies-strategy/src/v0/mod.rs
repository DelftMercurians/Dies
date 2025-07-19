use dies_core::{GameState, Handicap};
use dies_executor::behavior_tree_api::{GameContext, RobotSituation};

mod utils;

pub mod freekick_interference;
pub mod freekick_kicker;
pub mod harasser;
pub mod keeper;
pub mod kickoff_kicker;
pub mod penalty_kicker;
pub mod secondary_harasser;
pub mod striker;
pub mod waller;

pub fn v0_strategy(game: &mut GameContext) {
    let game_state = game.game_state();

    // Goalkeeper - always exactly one
    game.add_role("goalkeeper")
        .count(1)
        .require(move |s| {
            game_state
                .our_keeper_id
                .map(|id| id == s.player_id())
                .unwrap_or(true)
        })
        .if_must_reassign_can_we_do_it_now(true)
        .score(|_| 1.0)
        .behavior(|s| keeper::build_goalkeeper_tree(s));

    // Game state specific roles
    match game_state.game_state {
        GameState::Kickoff | GameState::PrepareKickoff => {
            if game_state.us_operating {
                // Kickoff kicker
                game.add_role("kickoff_kicker")
                    .count(1)
                    .exclude(|s| s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]))
                    .score(|s| score_for_kicker(s))
                    .behavior(|s| kickoff_kicker::build_kickoff_kicker_tree(s));
            }
        }
        GameState::FreeKick => {
            if game_state.us_operating {
                // Free kick kicker
                game.add_role("free_kick_kicker")
                    .count(1)
                    .exclude(|s| s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]))
                    .score(|s| score_for_kicker(s) + (1.0 - (s.distance_to_ball() / 9000.0)))
                    .behavior(|s| freekick_kicker::build_free_kick_kicker_tree(s));
            } else {
                // Free kick interference
                game.add_role("free_kick_interference")
                    .max(1)
                    .score(|s| score_for_interference(s))
                    .behavior(|s| freekick_interference::build_free_kick_interference_tree(s));
            }
        }
        GameState::Penalty | GameState::PreparePenalty | GameState::PenaltyRun => {
            if game_state.us_operating {
                // Penalty kicker
                game.add_role("penalty_kicker")
                    .count(1)
                    .exclude(|s| s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]))
                    .score(|_| 100.0)
                    .behavior(|s| penalty_kicker::build_penalty_kicker_tree(s));
            }
        }
        _ => {}
    }

    v0_defence(game);
}

pub fn v0_defence(game: &mut GameContext) {
    // 5: 1 + 2w + 2h + 0a

    // harasser 1
    // if game.ball_has_been_on_opp_side_for_at_least(10.0) {
    //     game.add_role("striker_1")
    //         .max(1)
    //         .score(|s| 1.0 + favor_x_pos(s, 1.0))
    //         .exclude(|s| s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]))
    //         .behavior(|s| striker::build_striker_tree(s));
    // } else {
    game.add_role("harasser_1")
        .exclude(|s| s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]))
        .max(1)
        .score(|s| 1.0 + prefer_not_role(s, "waller"))
        .behavior(|s| harasser::build_harasser_tree(s));
    // }

    game.add_role("waller_1")
        .max(1)
        .score(|s| 1.0 + favor_x_pos(s, -1.0) + 10.0 * prefer_current_role(s, "waller_1"))
        .behavior(|s| waller::build_waller_tree(s));

    // harasser 2
    if game.ball_has_been_on_opp_side_for_at_least(15.0) {
        game.add_role("striker_2")
            .exclude(|s| s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]))
            .max(1)
            .score(|s| 1.0 + favor_x_pos(s, 1.0))
            .behavior(|s| striker::build_striker_tree(s));
    } else {
        game.add_role("tagging_harasser")
            .exclude(|s| s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]))
            .max(1)
            .score(|s| 1.0 + prefer_not_role(s, "waller"))
            .behavior(|s| secondary_harasser::build_secondary_harasser_tree(s));
    }

    game.add_role("waller_2")
        .max(1)
        .score(|s| 1.0 + favor_x_pos(s, -1.0) + 10.0 * prefer_current_role(s, "waller_2"))
        .behavior(|s| waller::build_waller_tree(s));

    game.add_role("striker_3")
        .score(|s| 1.0 + favor_x_pos(s, 1.0))
        .exclude(|s| s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]))
        .behavior(|s| striker::build_striker_tree(s));
}

fn score_for_kicker(s: &RobotSituation) -> f64 {
    let current_role = s.current_role();
    if current_role.contains("striker") {
        10.0
    } else if current_role.contains("harasser") {
        5.0
    } else if !current_role.contains("goalkeeper") {
        1.0
    } else {
        0.0
    }
}

fn score_for_interference(s: &RobotSituation) -> f64 {
    let current_role = s.current_role();
    if current_role.contains("striker") {
        if s.ball_position().x > 0.0 {
            10.0
        } else {
            5.0
        }
    } else if current_role.contains("harasser") {
        if s.ball_position().x > 0.0 {
            5.0
        } else {
            10.0
        }
    } else if !current_role.contains("goalkeeper") {
        1.0
    } else {
        0.0
    }
}

fn favor_x_pos(s: &RobotSituation, preferred_side: f64) -> f64 {
    let sign = s.position().x.signum();
    if sign == preferred_side.signum() {
        1.0
    } else {
        0.0
    }
}

fn prefer_current_role(s: &RobotSituation, role_name: &str) -> f64 {
    if s.current_role() == role_name {
        1.0
    } else {
        0.0
    }
}

fn prefer_not_role(s: &RobotSituation, role_name: &str) -> f64 {
    if s.current_role_is(role_name) {
        0.0
    } else {
        1.0
    }
}
