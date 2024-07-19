use super::scenario::ScenarioSetup;

use crate::roles::attacker::{self, Attacker};
use crate::roles::dummy_role::DummyRole;
use crate::roles::harasser::Harasser;
use crate::roles::skills::FetchBallWithHeading;
use crate::roles::{waller, Goalkeeper};
use crate::strategy::attack_strat::PlayStrategy;
use crate::strategy::dynamic::DynamicStrategy;
use crate::strategy::free_kick::FreeKickStrategy;
use crate::strategy::kickoff::KickoffStrategy;
use crate::strategy::penalty_kick::PenaltyKickStrategy;
use crate::strategy::stop::StopStrategy;
use crate::strategy::test_strat::TestStrat;
use crate::{
    roles::{
        dribble_role::DribbleRole, fetcher_role::FetcherRole, kicker_role::KickerRole,
        test_role::TestRole, waller::Waller,
    },
    strategy::AdHocStrategy,
};
use crate::{strategy, StrategyMap};
use dies_core::{Angle, GameState, PlayerId, StrategyGameStateMacther, Vector2, Vector3};
use serde::{Deserialize, Serialize};

pub fn play() -> ScenarioSetup {
    let assign = |mut player_ids: Vec<PlayerId>| {
        let mut map = StrategyMap::new();
        if player_ids.len() == 0 {
            return map;
        }

        player_ids.sort();
        player_ids.reverse();
        let keeper = player_ids.pop().unwrap();
        map.insert(
            StrategyGameStateMacther::Specific(GameState::Halt),
            AdHocStrategy::new(),
        );
        map.insert(
            StrategyGameStateMacther::Specific(GameState::FreeKick),
            FreeKickStrategy::new(Some(keeper)),
        );

        let mut play = PlayStrategy::new(keeper);
        if player_ids.len() == 1 {
            map.insert(
                StrategyGameStateMacther::any_of(vec![GameState::Run, GameState::Stop].as_slice()),
                play,
            );
            return map;
        }

        let waller1 = player_ids.pop().unwrap();
        if player_ids.len() == 2 {
            play.defense.add_wallers(vec![waller1]);
            map.insert(
                StrategyGameStateMacther::any_of(vec![GameState::Run, GameState::Stop].as_slice()),
                play,
            );
            return map;
        }

        let waller2 = player_ids.pop().unwrap();
        play.defense.add_wallers(vec![waller1, waller2]);
        if player_ids.len() == 3 {
            map.insert(
                StrategyGameStateMacther::any_of(vec![GameState::Run, GameState::Stop].as_slice()),
                play,
            );
            return map;
        }

        let harasser = player_ids.pop().unwrap();
        play.defense.add_harasser(harasser);
        if player_ids.len() == 4 {
            map.insert(
                StrategyGameStateMacther::any_of(vec![GameState::Run, GameState::Stop].as_slice()),
                play,
            );
            return map;
        }

        let attacker1 = player_ids.pop().unwrap();
        play.attack.add_attacker(attacker1);
        if player_ids.len() == 5 {
            map.insert(
                StrategyGameStateMacther::any_of(vec![GameState::Run, GameState::Stop].as_slice()),
                play,
            );
            return map;
        }

        let attacker2 = player_ids.pop().unwrap();
        play.attack.add_attacker(attacker2);
        map.insert(
            StrategyGameStateMacther::any_of(vec![GameState::Run, GameState::Stop].as_slice()),
            play,
        );

        map
    };

    let strat = DynamicStrategy::new(assign);
    let mut scenario = ScenarioSetup::new(strat, StrategyGameStateMacther::Any);
    scenario
}
