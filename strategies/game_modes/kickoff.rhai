import "shared/utilities" as util;
import "shared/subtrees" as trees;

// Main kickoff tree builder
fn build_kickoff_bt(player_id) {
    return Select([
        // Try to be the kicker (only one allowed in center circle)
        Guard(
            |s| util::we_are_attacking(s),
            Semaphore(
                trees::build_kickoff_kicker_behavior(),
                "kickoff_kicker",
                1,
                "Kickoff Kicker"
            ),
            "We are attacking"
        ),
        // Everyone else positions in own half, outside center circle
        build_kickoff_supporter_behavior(player_id)
    ], "Kickoff Roles");
}

fn build_kickoff_supporter_behavior(player_id) {
    return GoToPosition(
        util::get_kickoff_supporter_pos(player_id),
        #{ heading: 0.0 },
        "Kickoff Support Position"
    );
} 