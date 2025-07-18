use dies_executor::behavior_tree_api::GameContext;

mod test_fetchball;
mod test_movement;
mod test_passer;
mod test_receiver;

pub fn testing_strategy(game: &mut GameContext) {
    game.add_role("passer_test")
        .count(1)
        .score(|_| 1.0)
        .behavior(|_| test_passer::build_test_passer());

    game.add_role("receiver_test")
        .count(1)
        .score(|_| 1.0)
        .behavior(|_| test_receiver::build_test_receiver());
}
