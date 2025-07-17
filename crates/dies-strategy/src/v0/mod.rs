use dies_core::{GameState, Handicap, PlayerId, TeamColor};
use dies_executor::behavior_tree_api::{GameContext, RobotSituation};

use crate::v0::{
    harasser::score_as_harasser, penalty_kicker::score_penalty_kicker, striker::score_striker,
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
    let game_state = game.game_state();
    if game.team_data().own_players.len() == 1 {
        // Goalkeeper - always exactly one
        game.add_role("goalkeeper")
            .count(1)
            // .can_be_reassigned(false)
            .require(move |s| {
                game_state
                    .our_keeper_id
                    .map(|id| id == s.player_id())
                    .unwrap_or(true)
            })
            // .exclude(|s| s.has_handicap(Handicap::NoKicker))
            .if_must_reassign_can_we_do_it_now(true)
            .score(|s| 10_000.0 * s.player_id().as_u32() as f64)
            .behavior(|s| keeper::build_goalkeeper_tree(s));
        return;
    } else {
        game.add_role("striker")
            .score(score_striker)
            .behavior(|s| striker::build_striker_tree(s));
    }

    // Game state specific roles
    match game_state.game_state {
        GameState::Kickoff | GameState::PrepareKickoff => {
            if game_state.us_operating {
                // Kickoff kicker
                game.add_role("kickoff_kicker")
                    .count(1)
                    .exclude(|s| s.has_handicap(Handicap::NoKicker))
                    .score(|s| {
                        let score = score_for_kicker(s);
                        if s.position().x < 0.0 {
                            score * 2.0
                        } else {
                            score
                        }
                    })
                    .behavior(|s| kickoff_kicker::build_kickoff_kicker_tree(s));
            }
        }
        GameState::FreeKick => {
            if game_state.us_operating {
                // Free kick kicker
                game.add_role("free_kick_kicker")
                    .count(1)
                    .exclude(|s| s.has_handicap(Handicap::NoKicker))
                    .score(|s| {
                        let score = score_for_kicker(s);
                        if s.am_closest_to_ball() {
                            score + 5.0
                        } else {
                            score
                        }
                    })
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

    // v0_offense(game);
    // v0_neutral(game);
    v0_defence(game);
}

pub fn v0_offense(game: &mut GameContext) {
    // 5: 1 + 1w + 1h + 2a

    // harasser
    if game.ball_has_been_on_opp_side_for_at_least(10.0) {
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

    game.add_role("striker_1")
        .max(1)
        .score(score_striker)
        .exclude(|s| s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]))
        .behavior(|s| striker::build_striker_tree(s));

    game.add_role("waller_1")
        .max(1)
        .score(score_as_waller)
        .behavior(|s| waller::build_waller_tree(s));

    game.add_role("striker_3")
        .max(1)
        .score(score_striker)
        .exclude(|s| s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]))
        .behavior(|s| striker::build_striker_tree(s));

    // the last one: striker
    game.add_role("striker_last")
        .max(1)
        .score(score_striker)
        .exclude(|s| s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]))
        .behavior(|s| striker::build_striker_tree(s));
}

pub fn v0_neutral(game: &mut GameContext) {
    // 5: 1 + 2w + 1h + 1a

    // harasser
    if game.ball_has_been_on_opp_side_for_at_least(10.0) {
        game.add_role("striker_1")
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

    game.add_role("waller_1")
        .max(1)
        .score(score_as_waller)
        .behavior(|s| waller::build_waller_tree(s));

    game.add_role("striker_2")
        .max(1)
        .score(score_striker)
        .exclude(|s| s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]))
        .behavior(|s| striker::build_striker_tree(s));

    game.add_role("waller_2")
        .max(1)
        .score(score_as_waller)
        .behavior(|s| waller::build_waller_tree(s));

    // the last one: striker
    game.add_role("striker_last")
        .max(1)
        .score(score_striker)
        .exclude(|s| s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]))
        .behavior(|s| striker::build_striker_tree(s));
}

pub fn v0_defence(game: &mut GameContext) {
    // 5: 1 + 2w + 2h + 0a

    // harasser 1
    if game.ball_has_been_on_opp_side_for_at_least(10.0) {
        game.add_role("striker_1")
            .max(1)
            .score(score_striker)
            // .exclude(|s| s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]))
            .behavior(|s| striker::build_striker_tree(s));
    } else {
        game.add_role("harasser_1")
            .max(1)
            .score(score_as_harasser)
            .behavior(|s| harasser::build_harasser_tree(s));
    }

    game.add_role("waller_1")
        .max(1)
        .score(score_as_waller)
        .behavior(|s| waller::build_waller_tree(s));

    // harasser 2
    if game.ball_has_been_on_opp_side_for_at_least(10.0) {
        game.add_role("striker_2")
            .max(1)
            .score(score_striker)
            // .exclude(|s| s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]))
            .behavior(|s| striker::build_striker_tree(s));
    } else {
        game.add_role("harasser_2")
            .max(1)
            .score(score_as_harasser)
            .behavior(|s| harasser::build_harasser_tree(s));
    }

    game.add_role("waller_2")
        .max(1)
        .score(score_as_waller)
        .behavior(|s| waller::build_waller_tree(s));

    // the last one: harasser 3
    // if game.ball_has_been_on_opp_side_for_at_least(10.0) {
    game.add_role("striker_3")
        // .max(1)
        .score(score_striker)
        // .exclude(|s| s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]))
        .behavior(|s| striker::build_striker_tree(s));
    // }
}

fn score_for_kicker(s: &RobotSituation) -> f64 {
    let current_role = s.current_role();
    // let ball_dist = s.distance_to_ball();
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

fn score_free_kick_interference(s: &RobotSituation) -> f64 {
    let mut score = 70.0;

    // Prefer robots that can position between ball and our goal
    if let Some(ball) = &s.world.ball {
        let ball_pos = ball.position.xy();
        let goal_pos = s.get_own_goal_position();
        let my_pos = s.player_data().position;

        // Calculate positioning score
        let ball_to_goal = goal_pos - ball_pos;
        let ball_to_me = my_pos - ball_pos;
        let projection =
            (ball_to_me.x * ball_to_goal.x + ball_to_me.y * ball_to_goal.y) / ball_to_goal.norm();
        let projection_ratio = projection / ball_to_goal.norm();

        if projection_ratio > 0.2 && projection_ratio < 0.8 {
            score += 20.0;
        }
    }

    score
}

// fn score_striker(s: &RobotSituation) -> f64 {
//     // Smoothly prefer robots on our side, with a soft transition at x=0
//     // Uses a logistic function to create a smooth step
//     let x = s.position().x;
//     let field_length = s.field().field_length;
//     let k = 0.01; // steepness of the transition, adjust as needed
//     let weight = 1.0 / (1.0 + ((k * x).exp())); // smoothly transitions from 1 to 0 as x goes from negative to positive
//     1000.0 * (field_length + x) * weight
// }
