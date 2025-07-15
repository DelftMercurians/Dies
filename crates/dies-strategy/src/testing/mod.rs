use dies_executor::behavior_tree_api::GameContext;

mod test_movement;
mod test_passer;
mod test_receiver;

pub fn testing_strategy(game: &mut GameContext) {
    game.add_role("movement")
        .count(6)
        .score(|_| 100000.0)
        .behavior(|_| test_movement::build_test_movement());
}
