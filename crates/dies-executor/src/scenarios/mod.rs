mod scenario;

use crate::roles::attacker::Attacker;
use crate::roles::dummy_role::DummyRole;
use crate::roles::harasser::Harasser;
use crate::roles::skills::FetchBallWithHeading;
use crate::roles::Goalkeeper;
use crate::strategy;
use crate::strategy::attack_strat::PlayStrategy;
use crate::strategy::free_kick::FreeKickStrategy;
use crate::strategy::kickoff::KickoffStrategy;
use crate::strategy::penalty_kick::PenaltyKickStrategy;
use crate::strategy::test_strat::TestStrat;
use crate::{
    roles::{
        dribble_role::DribbleRole, fetcher_role::FetcherRole, kicker_role::KickerRole,
        test_role::TestRole, waller::Waller,
    },
    strategy::AdHocStrategy,
};
use dies_core::{Angle, GameState, StrategyGameStateMacther, Vector2, Vector3};
use scenario::ScenarioSetup;
use serde::{Deserialize, Serialize};

// **NOTE**: Add all new scenarios to the `scenarios!` macro at the end of this file.

fn empty_scenario() -> ScenarioSetup {
    ScenarioSetup::new(AdHocStrategy::new(), StrategyGameStateMacther::Any)
}

fn play() -> ScenarioSetup {
    let mut setup = ScenarioSetup::new(PlayStrategy::new(), StrategyGameStateMacther::Any);
    setup
        .add_ball()
        .add_own_player()
        .add_own_player()
        .add_own_player()
        .add_opp_player_at(Vector2::new(1000.0, 1000.0))
        .add_opp_player_at(Vector2::new(4500.0, 0.0))
        .add_opp_player_at(Vector2::new(1000.0, -1000.0));
    setup
}

fn goalie_test() -> ScenarioSetup {
    let mut strategy = AdHocStrategy::new();
    strategy.add_role(Box::new(Goalkeeper::new()));
    let mut scenario = ScenarioSetup::new(strategy, StrategyGameStateMacther::Any);
    scenario.add_ball().add_own_player();
    scenario
}

fn penalty_kick() -> ScenarioSetup {
    let strategy = PenaltyKickStrategy::new(None);
    let mut scenario = ScenarioSetup::new(
        strategy,
        StrategyGameStateMacther::Specific(GameState::PreparePenalty),
    );
    scenario.add_strategy(
        StrategyGameStateMacther::Specific(GameState::Penalty),
        PenaltyKickStrategy::new(None),
    );
    scenario.add_strategy(
        StrategyGameStateMacther::Specific(GameState::PenaltyRun),
        PenaltyKickStrategy::new(None),
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
    scenario
        .add_ball_at(Vector2::new(0.0, 0.0))
        .add_own_player_at(Vector2::new(1000.0, 1000.0))
        .add_own_player_at(Vector2::new(-1000.0, -1000.0));
    scenario
}

fn kickoff() -> ScenarioSetup {
    let strategy = KickoffStrategy::new(None);
    let mut scenario = ScenarioSetup::new(
        strategy,
        StrategyGameStateMacther::Specific(GameState::PrepareKickoff),
    );
    scenario.add_strategy(
        StrategyGameStateMacther::Specific(GameState::Kickoff),
        KickoffStrategy::new(None),
    );
    scenario
        .add_ball_at(Vector2::new(0.0, 0.0))
        .add_own_player_at(Vector2::new(-1000.0, 1000.0))
        .add_own_player_at(Vector2::new(-1000.0, -1000.0));
    scenario
}

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
    strategy.add_role(Box::new(Waller::new(0.0)));
    let mut scenario = ScenarioSetup::new(strategy, StrategyGameStateMacther::Any);
    scenario
        .add_ball_at(Vector2::new(-4300.0, 3000.0))
        .add_own_player_at(Vector2::new(-4000.0, -2000.0));

    scenario
}

fn need_to_cross_the_goal_area_alt() -> ScenarioSetup {
    let mut strategy = AdHocStrategy::new();
    strategy.add_role(Box::new(Waller::new(0.0)));
    let mut scenario = ScenarioSetup::new(strategy, StrategyGameStateMacther::Any);
    scenario
        .add_ball_at(Vector2::new(-4500.0, -2950.0))
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
    let scenario = ScenarioSetup::new(strategy, StrategyGameStateMacther::Any);
    scenario
}

fn fetch_ball_with_heading() -> ScenarioSetup {
    let mut strategy = AdHocStrategy::new();
    let skill = FetchBallWithHeading::new(Angle::from_degrees(90.0));
    strategy.add_role(Box::new(DummyRole::new(Box::new(skill))));
    let scenario = ScenarioSetup::new(strategy, StrategyGameStateMacther::Any);
    // scenario.add_own_player();
    // .add_ball_at(Vector3::new(0.0, 0.0, 0.0))
    // .add_own_player_at(Vector2::new(1000.0, 1000.0))
    // .add_own_player_at(Vector2::new(-1000.0, -1000.0));
    scenario
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
    empty_scenario,
    play,
    goalie_test,
    kick_pass,
    need_to_cross_the_goal_area,
    need_to_cross_the_goal_area_alt,
    fetch_ball_test_sim,
    fetch_ball_test_live,
    fetch_ball_with_heading,
    free_kick,
    penalty_kick,
    kickoff,
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
