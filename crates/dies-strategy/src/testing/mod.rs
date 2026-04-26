use dies_executor::behavior_tree_api::GameContext;

mod test_fetchball;
mod test_movement;
mod test_passer;
mod test_receiver;
mod test_dribble;
mod test_passerv2;
mod test_receiverv2;

pub fn testing_strategy(game: &mut GameContext) {
    game.add_role("receiver")
        .score(|_| 1.0)
        .require(|s| s.player_id.as_u32() == 0)
        .behavior(|_| test_receiverv2::build_test_receiverv2());
    game.add_role("passer")
        .score(|_| 1.0)
        .require(|s| s.player_id.as_u32() == 1)
        .behavior(|_| test_passerv2::build_test_passerv2());
}
