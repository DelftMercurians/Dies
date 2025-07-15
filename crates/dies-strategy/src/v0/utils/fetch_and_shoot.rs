use dies_executor::behavior_tree_api::*;

use crate::v0::utils::find_best_shoot_target;

pub fn fetch_and_shoot() -> BehaviorNode {
    sequence_node()
        .add(fetch_ball().description("Get ball".to_string()).build())
        .add(shoot(find_best_shoot_target))
        .build()
        .into()
}
