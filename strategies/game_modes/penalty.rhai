import "shared/utilities" as util;
import "shared/situations" as sit;
import "shared/subtrees" as trees;

// Main penalty tree builder
fn build_penalty_bt(player_id) {
    return Select([
        // Goalkeeper behavior (always player 0)
        Guard(|s| sit::is_goalkeeper(s),
            trees::build_penalty_goalkeeper_behavior(),
            "Is goalkeeper?"
        ),
        
        // Try to be penalty taker (only one allowed)
        Semaphore(
            trees::build_penalty_taker_behavior(),
            "penalty_taker",
            1,
            "Penalty Taker"
        ),
        
        // Everyone else stays back
        build_penalty_support_behavior(player_id)
    ], "Penalty Roles");
}

// Penalty support behavior (not exported as it's only used internally)
fn build_penalty_support_behavior(player_id) {
    return GoToPosition(
        util::get_penalty_support_pos(player_id),
        #{ heading: 0.0 },
        "Penalty Support Position"
    );
} 