use dies_executor::{
    behavior_tree_api::*, find_best_preshoot_heading, find_best_preshoot_target,
    find_best_preshoot_target_target, find_best_shoot_target,
};

pub fn fetch_and_shoot() -> BehaviorNode {
    fetch_and_shoot_with_prep().into()
}

pub fn fetch_and_shoot_with_prep() -> BehaviorNode {
    fetch_ball_with_preshoot()
        .description("Fetch and shoot".to_string())
        .build()
        .into()
}
