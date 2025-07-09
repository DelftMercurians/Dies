import "shared/utilities" as util;
import "shared/subtrees" as trees;

// Main kickoff tree builder
fn build_kickoff_bt(player_id) {
    return Select([
        // Try to be the kicker (only one allowed in center circle)
        Semaphore(
            trees::build_kickoff_kicker_behavior(),
            "kickoff_kicker",
            1,
            "Kickoff Kicker"
        ),
        // Everyone else positions in own half, outside center circle
        build_kickoff_supporter_behavior(player_id)
    ], "Kickoff Roles");
}

// Kickoff supporter behavior (not exported as it's only used internally)
fn build_kickoff_supporter_behavior(player_id) {
    return GoToPosition(
        util::get_kickoff_supporter_pos(player_id),
        #{ heading: 0.0 },
        "Kickoff Support Position"
    );
} 