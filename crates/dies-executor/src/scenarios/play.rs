use super::scenario::ScenarioSetup;

use crate::strategy::attack_strat::PlayStrategy;
use crate::strategy::dynamic::DynamicStrategy;
use crate::strategy::free_kick::FreeKickStrategy;
use crate::strategy::kickoff::KickoffStrategy;
use crate::strategy::penalty_kick::PenaltyKickStrategy;
use crate::strategy::AdHocStrategy;
use crate::StrategyMap;
use dies_core::{GameState, PlayerId, StrategyGameStateMacther, Vector2};

pub fn play() -> ScenarioSetup {
    let assign = |mut player_ids: Vec<PlayerId>| {
        log::info!(
            "Assigning roles for play scenario, players: {:?}",
            player_ids
        );

        let mut map = StrategyMap::new();
        let num_players = player_ids.len();
        if num_players == 0 {
            return map;
        }

        player_ids.sort();
        player_ids.reverse();
        let keeper = player_ids.pop().unwrap();
        log::info!("Keeper: {:?}", keeper);
        map.insert(
            StrategyGameStateMacther::any_of(vec![
                GameState::Halt,
                GameState::Timeout,
                GameState::Unknown,
            ].as_slice()),
            AdHocStrategy::new(),
        );
        map.insert(
            StrategyGameStateMacther::any_of(
                vec![GameState::Kickoff, GameState::PrepareKickoff].as_slice(),
            ),
            KickoffStrategy::new(),
        );
        map.insert(
            StrategyGameStateMacther::any_of(
                vec![
                    GameState::Penalty,
                    GameState::PreparePenalty,
                    GameState::PenaltyRun,
                ]
                .as_slice(),
            ),
            PenaltyKickStrategy::new(),
        );

        let mut play = PlayStrategy::new(keeper);
        let states = vec![
            GameState::Run,
            GameState::Stop,
            GameState::BallReplacement(Vector2::zeros()),
            GameState::FreeKick,
        ];
        if num_players == 1 {
            map.insert(StrategyGameStateMacther::any_of(states.as_slice()), play);
            return map;
        }

        let waller1 = player_ids.pop().unwrap();
        if num_players == 2 {
            play.defense.add_wallers(vec![waller1]);
            map.insert(StrategyGameStateMacther::any_of(states.as_slice()), play);
            return map;
        }

        let harasser = player_ids.pop().unwrap();
        play.defense.add_harasser(harasser);
        if num_players == 3 {
            map.insert(StrategyGameStateMacther::any_of(states.as_slice()), play);
            return map;
        }

        let waller2 = player_ids.pop().unwrap();
        play.defense.add_wallers(vec![waller1, waller2]);
        if num_players == 4 {
            map.insert(StrategyGameStateMacther::any_of(states.as_slice()), play);
            return map;
        }

        let attacker1 = player_ids.pop().unwrap();
        play.attack.add_attacker(attacker1);
        if num_players == 5 {
            map.insert(StrategyGameStateMacther::any_of(states.as_slice()), play);
            return map;
        }

        let attacker2 = player_ids.pop().unwrap();
        play.attack.add_attacker(attacker2);
        map.insert(StrategyGameStateMacther::any_of(states.as_slice()), play);

        map
    };

    let strat = DynamicStrategy::new(assign);
    let mut scenario = ScenarioSetup::new(strat, StrategyGameStateMacther::Any);

    scenario
        .add_ball()
        .add_own_player()
        .add_own_player()
        .add_own_player()
        .add_own_player()
        .add_own_player()
        .add_own_player();
    scenario
}
