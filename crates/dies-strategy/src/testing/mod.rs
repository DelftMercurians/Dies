use dies_executor::behavior_tree_api::GameContext;

mod test_passer;
mod test_receiver;

pub fn testing_strategy(game: &mut GameContext) {
    game.add_role("receiver")
        .count(1)
        .score(|_| 100000.0)
        .behavior(|_| test_receiver::build_test_receiver());

    game.add_role("passer")
        .count(1)
        .score(|_| 100000.0)
        .behavior(|_| test_passer::build_test_passer());
}
