#[deny(dead_code)]
mod scenario;

use crate::roles::attacker::Attacker;
use crate::roles::harasser::Harasser;
use crate::strategy::free_kick::FreeKickStrategy;
use crate::strategy::kickoff::KickoffStrategy;
use crate::strategy::penalty_kick::PenaltyKickStrategy;
use crate::{
    roles::{
        dribble_role::DribbleRole,
        test_role::TestRole,
        waller::Waller,
        fetcher_role::FetcherRole,
        kicker_role::KickerRole
    },
    strategy::AdHocStrategy,
};
use dies_core::{GameState, Vector2, Vector3};
use scenario::ScenarioSetup;
use serde::{Deserialize, Serialize};

// **NOTE**: Add all new scenarios to the `scenarios!` macro at the end of this file.

fn empty_scenario() -> ScenarioSetup {
    ScenarioSetup::new(AdHocStrategy::new(), None)
}

fn penalty_kick() -> ScenarioSetup {
    let strategy = PenaltyKickStrategy::new(None);
    let mut scenario = ScenarioSetup::new(strategy, Some(GameState::PreparePenalty));
    scenario.add_strategy(GameState::Penalty, PenaltyKickStrategy::new(None));
    scenario.add_strategy(GameState::PenaltyRun, PenaltyKickStrategy::new(None));
    scenario
        .add_ball_at(Vector3::new(0.0, 300.0, 0.0))
        .add_own_player_at(Vector2::new(-1000.0, 1000.0))
        .add_own_player_at(Vector2::new(-1000.0, -1000.0))
        .add_opp_player_at(Vector2::new(3500.0, 0.0));
    scenario
}

fn free_kick() -> ScenarioSetup {
    let strategy = FreeKickStrategy::new(None);
    let mut scenario = ScenarioSetup::new(strategy, Some(GameState::FreeKick));
    scenario
        .add_ball_at(Vector3::new(0.0, 0.0, 0.0))
        .add_own_player_at(Vector2::new(1000.0, 1000.0))
        .add_own_player_at(Vector2::new(-1000.0, -1000.0));
    scenario
}

fn kickoff() -> ScenarioSetup {
    let strategy = KickoffStrategy::new(None);
    let mut scenario = ScenarioSetup::new(strategy, Some(GameState::PrepareKickoff));
    scenario.add_strategy(GameState::Kickoff, KickoffStrategy::new(None));
    scenario
        .add_ball_at(Vector3::new(0.0, 0.0, 0.0))
        .add_own_player_at(Vector2::new(-1000.0, 1000.0))
        .add_own_player_at(Vector2::new(-1000.0, -1000.0));
    scenario
}

fn one_harasser_one_player_one_ball() -> ScenarioSetup {
    let mut strategy = AdHocStrategy::new();
    strategy.add_role(Box::new(Harasser::new(500.0)));
    let mut scenario = ScenarioSetup::new(strategy, None);
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
    let mut scenario = ScenarioSetup::new(strategy, None);
    scenario
        .add_ball_at(Vector3::new(-4300.0, 3000.0, 0.0))
        .add_own_player_at(Vector2::new(-4000.0, -2000.0));

    scenario
}


fn need_to_cross_the_goal_area_alt() -> ScenarioSetup {
    let mut strategy = AdHocStrategy::new();
    strategy.add_role(Box::new(Waller::new(0.0)));
    let mut scenario = ScenarioSetup::new(strategy, None);
    scenario
        .add_ball_at(Vector3::new(-4500.0, -2950.0, 0.0))
        .add_own_player_at(Vector2::new(-3800.0, 500.0));

    scenario
}


fn test_role_multiple_targets() -> ScenarioSetup {
    let mut strategy = AdHocStrategy::new();
    strategy.add_role(Box::new(Attacker::new(Vector2::new(-800.0, -1000.0))));
    let mut scenario = ScenarioSetup::new(strategy, None);
    scenario
        .add_ball()
        .add_own_player_at(Vector2::new(1000.0, 1000.0));
    scenario
}

fn three_attackers() -> ScenarioSetup {
    let mut strategy = AdHocStrategy::new();
    strategy.add_role(Box::new(TestRole::new(vec![Vector2::new(1000.0, 1000.0)])));

    let mut scenario = ScenarioSetup::new(strategy, None);
    scenario.add_own_player_at(Vector2::new(-1000.0, -1000.0));
    scenario.add_opp_player_at(Vector2::new(-500.0, 0.0));
    // scenario.add_opp_player_at(Vector2::new(0.0, 500.0));
    // scenario.add_opp_player_at(Vector2::new(500.0, -500.0));
    // scenario.add_opp_player_at(Vector2::new(-250.0, 750.0));

    scenario
}

fn fetch_ball_test_live() -> ScenarioSetup {
    let mut strategy = AdHocStrategy::new();
    strategy.add_role(Box::new(FetcherRole::new()));
    let mut scenario = ScenarioSetup::new(strategy, None);
    scenario
        .add_ball()
        .add_own_player()
        .add_own_player()
        .add_own_player();
    scenario
}

fn dribble() -> ScenarioSetup {
    let mut strategy = AdHocStrategy::new();
    strategy.add_role(Box::new(DribbleRole::new()));

    let mut scenario = ScenarioSetup::new(strategy, None);
    scenario
        .add_own_player_at(Vector2::new(-2000.0, -2000.0))
        .add_ball();

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
    need_to_cross_the_goal_area,
    need_to_cross_the_goal_area_alt,
    test_role_multiple_targets,
    dribble,
    fetch_ball_test_live,
    free_kick,
    penalty_kick,
    kickoff,
    one_harasser_one_player_one_ball,
    three_attackers
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
