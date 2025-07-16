use dies_executor::behavior_tree_api::*;

use crate::v0::utils::{find_best_preshoot_target, find_best_preshoot_target_target, find_best_shoot_target, find_best_preshoot_heading};

pub fn fetch_and_shoot() -> BehaviorNode {
    sequence_node()
        .add(fetch_ball().description("Get ball".to_string()).build())
        .add(shoot(find_best_shoot_target))
        .build()
        .into()
}

pub fn fetch_and_shoot_with_prep() -> BehaviorNode {
    fetch_ball_with_preshoot(
        find_best_preshoot_target,
        find_best_preshoot_target_target,
    )
    .with_heading(find_best_preshoot_heading)
    .description("Fetch and shoot".to_string())
    .build()
    .into()
}
