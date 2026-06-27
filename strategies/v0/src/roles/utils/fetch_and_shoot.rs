use crate::bt::{fetch_ball_with_preshoot, BehaviorNode};

pub fn fetch_and_shoot() -> BehaviorNode {
    fetch_and_shoot_with_prep()
}

pub fn fetch_and_shoot_with_prep() -> BehaviorNode {
    fetch_ball_with_preshoot()
        .description("Fetch and shoot")
        .build()
        .into()
}
