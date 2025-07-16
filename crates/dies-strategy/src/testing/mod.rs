use dies_executor::behavior_tree_api::GameContext;

mod test_fetchball;
mod test_movement;
mod test_passer;
mod test_receiver;

pub fn testing_strategy(game: &mut GameContext) {
    game.add_role("fetchball")
        .count(1)
        .score(|_| 100000.0)
        .behavior(|_| test_fetchball::build_test_fetchball());
}
