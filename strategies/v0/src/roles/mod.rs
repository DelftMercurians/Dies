//! The v0 strategy proper: declares the roles and their scorers/filters each
//! frame, exactly as the original did, then defers to the behavior trees built in
//! the sibling modules.

use dies_strategy_protocol::{GameState, Handicap};

use crate::bt::{GameContext, RobotSituation};

pub mod utils;

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
    let us_operating = game.us_operating();
    let keeper_id = game.our_keeper_id();

    // Goalkeeper — always exactly one.
    game.add_role("goalkeeper")
        .count(1)
        .require(move |s| keeper_id.map(|id| id == s.player_id()).unwrap_or(true))
        .if_must_reassign_can_we_do_it_now(true)
        .score(|_| 1.0)
        .behavior(keeper::build_goalkeeper_tree);

    // Game-state specific roles.
    match game_state {
        GameState::Kickoff | GameState::PrepareKickoff => {
            if us_operating {
                game.add_role("kickoff_kicker")
                    .count(1)
                    .exclude(|s| s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]))
                    .score(score_for_kicker)
                    .behavior(kickoff_kicker::build_kickoff_kicker_tree);
            }
        }
        GameState::FreeKick => {
            if us_operating {
                game.add_role("free_kick_kicker")
                    .count(1)
                    .exclude(|s| s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]))
                    .score(|s| score_for_kicker(s) + (1.0 - (s.distance_to_ball() / 9000.0)))
                    .behavior(freekick_kicker::build_free_kick_kicker_tree);
            } else {
                game.add_role("free_kick_interference")
                    .max(1)
                    .score(score_for_interference)
                    .behavior(freekick_interference::build_free_kick_interference_tree);
            }
        }
        GameState::Penalty | GameState::PreparePenalty | GameState::PenaltyRun => {
            if us_operating {
                game.add_role("penalty_kicker")
                    .count(1)
                    .exclude(|s| s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]))
                    .score(|_| 100.0)
                    .behavior(penalty_kicker::build_penalty_kicker_tree);
            }
        }
        _ => {}
    }

    v0_defence(game);
}

pub fn v0_defence(game: &mut GameContext) {
    // 5 field robots: 1 striker/harasser + 2 wallers + 2 harassers/strikers.

    if game.ball_has_been_on_opp_side_for_at_least(1.0) {
        game.add_role("striker_1")
            .max(1)
            .score(|s| 1.0 + favor_x_pos(s, 1.0))
            .exclude(|s| s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]))
            .behavior(striker::build_striker_tree);
    } else {
        game.add_role("harasser_1")
            .exclude(|s| s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]))
            .max(1)
            .score(|s| 1.0 + prefer_not_role(s, "waller"))
            .behavior(harasser::build_harasser_tree);
    }

    game.add_role("waller_1")
        .max(1)
        .score(|s| 1.0 + favor_x_pos(s, -1.0) + 10.0 * prefer_current_role(s, "waller_1"))
        .behavior(waller::build_waller_tree);

    if game.ball_has_been_on_opp_side_for_at_least(10.0) {
        game.add_role("striker_2")
            .exclude(|s| s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]))
            .max(1)
            .score(|s| 1.0 + favor_x_pos(s, 1.0))
            .behavior(striker::build_striker_tree);
    } else {
        game.add_role("secondary_harasser")
            .exclude(|s| s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]))
            .max(1)
            .score(|s| 1.0 + prefer_not_role(s, "waller"))
            .behavior(secondary_harasser::build_secondary_harasser_tree);
    }

    game.add_role("waller_2")
        .max(1)
        .score(|s| 1.0 + favor_x_pos(s, -1.0) + 10.0 * prefer_current_role(s, "waller_2"))
        .behavior(waller::build_waller_tree);

    game.add_role("striker_3")
        .score(|s| 1.0 + favor_x_pos(s, 1.0))
        .exclude(|s| s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]))
        .behavior(striker::build_striker_tree);
}

fn score_for_kicker(s: &RobotSituation) -> f64 {
    let current_role = s.current_role();
    if current_role.contains("striker") {
        10.0
    } else if current_role.contains("harasser") {
        10.0
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
