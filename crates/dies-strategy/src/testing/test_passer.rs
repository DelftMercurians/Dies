use dies_core::PlayerId;
use dies_executor::behavior_tree_api::{
    fetch_ball, pass, sequence_node, wait, BehaviorNode, RobotSituation,
};

pub fn build_test_passer() -> BehaviorNode {
    sequence_node()
        .add(fetch_ball().build())
        .add(pass().target_player_id(find_receiver).build())
        .add(wait(5.0).build())
        .build()
        .into()
}

fn find_receiver(s: &RobotSituation) -> PlayerId {
    s.world
        .own_players
        .iter()
        .find(|p| p.id != s.player_id)
        .unwrap()
        .id
}
