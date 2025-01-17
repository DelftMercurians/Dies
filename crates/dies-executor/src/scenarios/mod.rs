mod play;
mod scenario;

use std::vec;

use dies_core::{Angle, GameState, PlayerId, StrategyGameStateMacther, Vector2};
use play::play;
use scenario::ScenarioSetup;
use serde::{Deserialize, Serialize};

use crate::{
    roles::{
        dummy_role::DummyRole, fetcher_role::FetcherRole, harasser::Harasser,
        skills::FetchBallWithHeading, waller::Waller, Goalkeeper,
    },
    strategy::{
        attack_strat::PlayStrategy, free_kick::FreeKickStrategy, penalty_kick::PenaltyKickStrategy,
        test_strat::TestStrat, AdHocStrategy,
    },
};

// **NOTE**: Add all new scenarios to the `scenarios!` macro at the end of this file.

fn empty_scenario() -> ScenarioSetup {
    ScenarioSetup::new(AdHocStrategy::new(), StrategyGameStateMacther::Any)
}

fn attack() -> ScenarioSetup {
    let mut strat = PlayStrategy::new(PlayerId::new(11));
    strat.attack.add_attacker(PlayerId::new(0));
    strat.attack.add_attacker(PlayerId::new(1));

    let mut setup = ScenarioSetup::new(strat, StrategyGameStateMacther::any());
    setup
        .add_ball()
        .add_own_player()
        .add_own_player()
        .add_own_player()
        .add_own_player()
        .add_own_player()
        .add_own_player()
        .add_opp_player_at(Vector2::new(1000.0, 1000.0))
        .add_opp_player_at(Vector2::new(4500.0, 0.0))
        .add_opp_player_at(Vector2::new(1000.0, -1000.0));
    setup
}

fn defense_test() -> ScenarioSetup {
    let mut strat = AdHocStrategy::new();
    strat.add_role(Box::new(Goalkeeper::new()));
    strat.add_role(Box::new(Waller::new_with_index(0)));
    strat.add_role(Box::new(Waller::new_with_index(1)));

    ScenarioSetup::new(strat, StrategyGameStateMacther::Any)
}

fn goalie_test() -> ScenarioSetup {
    let mut strategy = AdHocStrategy::new();
    strategy.add_role(Box::new(Goalkeeper::new()));
    let mut scenario = ScenarioSetup::new(strategy, StrategyGameStateMacther::Any);
    scenario.add_ball().add_own_player().add_own_player();
    scenario
}

fn penalty_kick() -> ScenarioSetup {
    let strategy = PenaltyKickStrategy::new();
    let mut scenario = ScenarioSetup::new(
        strategy,
        StrategyGameStateMacther::any_of(
            vec![
                GameState::PreparePenalty,
                GameState::Penalty,
                GameState::PenaltyRun,
            ]
            .as_slice(),
        ),
    );
    scenario
        .add_ball_at(Vector2::new(0.0, 300.0))
        .add_own_player_at(Vector2::new(-1000.0, 1000.0))
        .add_own_player_at(Vector2::new(-1000.0, -1000.0))
        .add_opp_player_at(Vector2::new(3500.0, 0.0));
    scenario
}

fn kick_pass() -> ScenarioSetup {
    let mut scenario = ScenarioSetup::new(TestStrat::new(), StrategyGameStateMacther::Any);

    scenario
        .add_ball_at(Vector2::new(-1000.0, 1500.0))
        .add_own_player()
        .add_own_player()
        .add_own_player()
        .add_own_player();
    scenario
}

fn free_kick() -> ScenarioSetup {
    let strategy = FreeKickStrategy::new(None);
    let mut scenario = ScenarioSetup::new(
        strategy,
        StrategyGameStateMacther::Specific(GameState::FreeKick),
    );

    // besides the ball and the 4 players,
    // add another 4 random players
    scenario
        .add_ball_at(Vector2::new(0.0, 2600.0))
        .add_own_player_at(Vector2::new(1000.0, 1000.0))
        .add_own_player_at(Vector2::new(-100.0, 0.0))
        .add_opp_player_at(Vector2::new(200.0, 200.0))
        .add_own_player_at(Vector2::new(-200.0, 2600.0))
        .add_own_player_at(Vector2::new(-500.0, 600.0))
        .add_opp_player_at(Vector2::new(-800.0, -300.0))
        .add_own_player_at(Vector2::new(0.0, 2000.0))
        .add_opp_player_at(Vector2::new(300.0, 2300.0))
        .add_own_player_at(Vector2::new(100.0, 2700.0))
        .add_opp_player_at(Vector2::new(-400.0, 2900.0))
        .add_own_player_at(Vector2::new(0.0, 2900.0));
    scenario
}

// fn kickoff() -> ScenarioSetup {
//     let strategy = KickoffStrategy::new(None);
//     let mut scenario = ScenarioSetup::new(
//         strategy,
//         StrategyGameStateMacther::Specific(GameState::PrepareKickoff),
//     );
//     scenario.add_strategy(
//         StrategyGameStateMacther::Specific(GameState::Kickoff),
//         KickoffStrategy::new(None),
//     );
//     scenario
//         .add_ball_at(Vector2::new(0.0, 0.0))
//         .add_own_player_at(Vector2::new(-1000.0, 1000.0))
//         .add_own_player_at(Vector2::new(-1000.0, -1000.0));
//     scenario
// }

fn one_harasser_one_player_one_ball() -> ScenarioSetup {
    let mut strategy = AdHocStrategy::new();
    strategy.add_role(Box::new(Harasser::new(500.0)));
    let mut scenario = ScenarioSetup::new(strategy, StrategyGameStateMacther::Any);
    scenario
        // .add_ball_at(Vector3::new(895.0, 2623.0, 0.0))
        .add_ball()
        .add_own_player_at(Vector2::new(2264.0, 336.0))
        .add_own_player_at(Vector2::new(0.0, 0.0));
    scenario
}

fn need_to_cross_the_goal_area() -> ScenarioSetup {
    let mut strategy = AdHocStrategy::new();
    strategy.add_role(Box::new(Waller::new(-240.0)));
    strategy.add_role(Box::new(Waller::new(-80.0)));
    strategy.add_role(Box::new(Waller::new(80.0)));
    strategy.add_role(Box::new(Waller::new(240.0)));
    let mut scenario = ScenarioSetup::new(strategy, StrategyGameStateMacther::Any);
    scenario
        .add_ball_at(Vector2::new(-4300.0, 3000.0))
        .add_own_player_at(Vector2::new(-4000.0, -2000.0))
        .add_own_player_at(Vector2::new(4000.0, -2000.0))
        .add_own_player_at(Vector2::new(3000.0, -2000.0))
        .add_own_player_at(Vector2::new(2000.0, -2000.0));

    scenario
}

fn need_to_cross_the_goal_area_alt() -> ScenarioSetup {
    let mut strategy = AdHocStrategy::new();
    strategy.add_role(Box::new(Waller::new(0.0)));
    let mut scenario = ScenarioSetup::new(strategy, StrategyGameStateMacther::Any);
    scenario
        .add_ball_at(Vector2::new(-0.0, -0.0))
        .add_own_player_at(Vector2::new(-3800.0, 500.0));

    scenario
}

fn fetch_ball_test_sim() -> ScenarioSetup {
    let mut strategy = AdHocStrategy::new();
    strategy.add_role(Box::new(FetcherRole::new()));
    let mut scenario = ScenarioSetup::new(strategy, StrategyGameStateMacther::Any);
    scenario
        .add_own_player()
        .add_ball_at(Vector2::new(-1000.0, -1000.0));
    scenario
}

fn fetch_ball_test_live() -> ScenarioSetup {
    let mut strategy = AdHocStrategy::new();
    strategy.add_role(Box::new(FetcherRole::new()));

    ScenarioSetup::new(strategy, StrategyGameStateMacther::Any)
}

fn fetch_ball_with_heading() -> ScenarioSetup {
    let mut strategy = AdHocStrategy::new();
    let skill = FetchBallWithHeading::new(Angle::from_degrees(90.0));
    strategy.add_role(Box::new(DummyRole::new(Box::new(skill))));

    // scenario.add_own_player();
    // .add_ball_at(Vector3::new(0.0, 0.0, 0.0))
    // .add_own_player_at(Vector2::new(1000.0, 1000.0))
    // .add_own_player_at(Vector2::new(-1000.0, -1000.0));
    ScenarioSetup::new(strategy, StrategyGameStateMacther::Any)
}

/// Creates a lookup table for scenarios as a global constant.
macro_rules! scenarios {
    ($($scen:ident),+) => {
        use paste::paste;

        paste! {
            /// Represents a scenario type.
            #[derive(Debug, Clone)]
            pub enum ScenarioType {
                $([< $scen:camel >]),+
            }
        }

        impl ScenarioType {
            /// Returns the name of the scenario.
            pub fn name(&self) -> &'static str {
                paste! {
                    match self {
                        $(ScenarioType::[< $scen:camel >] => stringify!($scen)),+
                    }
                }
            }

            /// Returns a list of scenario names.
            pub fn get_names() -> Vec<&'static str> {
                vec![$(stringify!($scen)),+]
            }

            /// Returns a scenario type by name.
            pub fn get_by_name(name: &str) -> Option<ScenarioType> {
                paste! {
                    match name {
                        $(stringify!($scen) => Some(ScenarioType::[< $scen:camel >]),)+
                        _ => None
                    }
                }
            }

            /// Returns a scenario setup by name.
            pub fn get_setup_by_name(name: &str) -> Option<ScenarioSetup> {
                Self::get_by_name(name).map(|s| s.into_setup())
            }

            /// Converts the scenario type into a scenario setup.
            pub fn into_setup(&self) -> ScenarioSetup {
                paste! {
                    match self {
                        $(ScenarioType::[< $scen:camel >] => $scen()),+
                    }
                }
            }
        }
    };
}

impl<'de> Deserialize<'de> for ScenarioType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let name = String::deserialize(deserializer)?;
        ScenarioType::get_by_name(&name)
            .ok_or_else(|| serde::de::Error::custom(format!("Scenario '{}' not found", name)))
    }
}

impl Serialize for ScenarioType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.name())
    }
}
// **NOTE**: Add new scenarios here.
scenarios! {
    attack,
    play,
    empty_scenario,
    goalie_test,
    defense_test,
    kick_pass,
    need_to_cross_the_goal_area,
    need_to_cross_the_goal_area_alt,
    fetch_ball_test_sim,
    fetch_ball_test_live,
    fetch_ball_with_heading,
    free_kick,
    penalty_kick,
    one_harasser_one_player_one_ball
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scenarios() {
        let names = ScenarioType::get_names();
        assert_eq!(names[0], "empty_scenario");

        ScenarioType::get_setup_by_name("empty_scenario").expect("empty_scenario not found");
    }
}
