import "keeper" as keeper;
import "striker" as striker;
import "penalty_kicker" as penalty_kicker;
import "harasser" as harasser;
import "waller" as waller;
import "waller_scorer" as waller_scorer;
import "harasser_scorer" as harasser_scorer;
import "striker_scorer" as striker_scorer;
import "kickoff_kicker" as kickoff_kicker;
import "freekick_kicker" as freekick_kicker;
import "freekick_interference" as freekick_interference;

fn main(game) {
    // Always need exactly one goalkeeper
    game.add_role("goalkeeper")
        .count(1)
        .score(|s| 100000.0)
        // .require(|s| s.player_id == 0)
        .behavior(|s| keeper::build_goalkeeper_tree());

    // Specialized roles based on game state
    switch game.game_state {
        "Kickoff" => {
            if game.us_operating {
                // Kickoff kicker - one robot to take the kickoff
                game.add_role("kickoff_kicker")
                    .count(1)
                    .score(|s| kickoff_kicker::score_kickoff_kicker(s))
                    // .exclude(|s| s.player_id == 0)
                    .behavior(|s| kickoff_kicker::build_kickoff_kicker_tree());
            }
        },
        "FreeKick" => {
            if game.us_operating {
                // Free kick kicker
                game.add_role("free_kick_kicker")
                    .count(1)
                    .score(|s| freekick_kicker::score_free_kick_kicker(s))
                    // .exclude(|s| s.player_id == 0)
                    .behavior(|s| freekick_kicker::build_free_kick_kicker_tree());
            } else {
                // Free kick interference - defend against opponent free kick
                game.add_role("free_kick_interference")
                    .min(1).max(2)
                    .score(|s| freekick_interference::score_free_kick_interference(s))
                    // .exclude(|s| s.player_id == 0)
                    .behavior(|s| freekick_interference::build_free_kick_interference_tree());
            }
        },
        "Penalty" => {
            if game.us_operating {
                // Penalty kicker
                game.add_role("penalty_kicker")
                    .count(1)
                    .score(|s| score_penalty_kicker(s))
                    // .exclude(|s| s.player_id == 0)
                    .behaviosr(|| build_penalty_kicker_tree());
            }
        }
    }

    // Standard striker role (simplified without special game mode handling)
    game.add_role("striker")
        .max(1)
        .score(|s| striker_scorer::score_striker(s))
        // .exclude(|s| s.player_id == 0)
        .behavior(|s| striker::build_striker_tree());

    // Dynamic waller role
    game.add_role("waller")
        .min(0).max(2)
        .score(|s| waller_scorer::score_as_waller(s))
        // .exclude(|s| s.player_id == 0)
        .behavior(|s| waller::build_waller_tree());

    // Dynamic harasser role
    game.add_role("harasser")
        .min(0).max(2)
        .score(|s| harasser_scorer::score_as_harasser(s))
        // .exclude(|s| s.player_id == 0)
        .behavior(|s| harasser::build_harasser_tree());
}
