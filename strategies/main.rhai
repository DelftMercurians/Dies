import "shared/utilities" as util;
import "shared/situations" as sit;
import "shared/subtrees" as trees;

fn main(game) {
    // Add base roles that are always needed
    game.add_role("goalkeeper")
        .count(1)
        .score(|s| 100.0)
        .require(|s| s.player_id == 0)
        .behavior(|| trees::build_goalkeeper_tree())
        .build();

    game.add_role("striker")
        .max(1)
        .score(|s| util::score_striker(s))
        .exclude(|s| s.player_id == 0)
        .behavior(|| trees::build_striker_tree())
        .build();

    game.add_role("defender")
        .min(1)
        .max(2)
        .score(|s| util::score_defender(s))
        .exclude(|s| s.player_id == 0)
        .behavior(|| trees::build_defender_tree())
        .build();

    game.add_role("support")
        .min(1)
        .score(|s| util::score_support(s))
        .exclude(|s| s.player_id == 0)
        .behavior(|| trees::build_support_tree())
        .build();

    // Add special roles based on game state
    switch game.game_state {
        "FreeKick" => {
            game.add_role("free_kicker")
                .max(1)
                .score(|s| util::score_free_kicker(s))
                .exclude(|s| s.player_id == 0)
                .behavior(|| trees::build_free_kicker_tree())
                .build();
        },
        "PenaltyKick" => {
            game.add_role("penalty_taker")
                .count(1)
                .score(|s| util::score_penalty_taker(s))
                .exclude(|s| s.player_id == 0)
                .behavior(|| trees::build_penalty_taker_tree())
                .build();
        }
    }
}