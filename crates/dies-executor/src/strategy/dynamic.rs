use std::collections::HashSet;

use dies_core::{GameState, PlayerId};

use crate::{roles::Role, StrategyMap};

use super::{AdHocStrategy, Strategy, StrategyCtx};

pub struct DynamicStrategy<F: Fn(Vec<PlayerId>) -> StrategyMap> {
    active_strategy: String,
    strategies: Option<StrategyMap>,
    player_ids: Vec<PlayerId>,
    adhoc_strategy: AdHocStrategy,
    last_game_state: GameState,
    strategy_fn: F,
}

impl<F> DynamicStrategy<F>
where
    F: Fn(Vec<PlayerId>) -> StrategyMap + 'static + Send,
{
    pub fn new(f: F) -> Self {
        Self {
            strategy_fn: f,
            strategies: None,
            player_ids: Vec::new(),
            adhoc_strategy: AdHocStrategy::new(),
            active_strategy: String::new(),
            last_game_state: GameState::Halt,
        }
    }
}

impl<F> Strategy for DynamicStrategy<F>
where
    F: Fn(Vec<PlayerId>) -> StrategyMap + 'static + Send,
{
    fn update(&mut self, ctx: StrategyCtx) {
        let player_ids = ctx.world.own_players.iter().map(|p| p.id).collect::<HashSet<_>>();
        if player_ids != self.player_ids.iter().cloned().collect() {
            self.player_ids = player_ids.iter().cloned().collect();
            self.strategies = Some((self.strategy_fn)(
                ctx.world.own_players.iter().map(|p| p.id).collect(),
            ));
        }

        let strategy_map = self.strategies.get_or_insert_with(|| {
            (self.strategy_fn)(ctx.world.own_players.iter().map(|p| p.id).collect())
        });

        let game_state = ctx.world.current_game_state.game_state;
        self.last_game_state = game_state;
        let strategy = strategy_map
            .get_strategy(&game_state)
            .unwrap_or_else(|| &mut self.adhoc_strategy);
        if strategy.name() != self.active_strategy {
            log::info!("Switching to strategy: {}", strategy.name());
            self.active_strategy = strategy.name().to_owned();
            strategy.on_enter(ctx.clone());
        }
        strategy.update(ctx);
    }

    fn get_role(&mut self, player_id: PlayerId, ctx: StrategyCtx) -> Option<&mut dyn Role> {
        let strategy_map = self.strategies.get_or_insert_with(|| {
            (self.strategy_fn)(self.player_ids.clone())
        });

        let strategy = strategy_map
            .get_strategy(&self.last_game_state)
            .unwrap_or_else(|| &mut self.adhoc_strategy);
        strategy.get_role(player_id)
    }

    fn name(&self) -> &'static str {
        "Dynamic Strategy"
    }
}
