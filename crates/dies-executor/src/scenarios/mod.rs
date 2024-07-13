#[deny(dead_code)]
mod scenario;
use std::f64::consts::PI;

use crate::{
    roles::{dribble_role::DribbleRole, test_role::TestRole, waller::Waller, fetcher_role::FetcherRole, kicker_role::KickerRole},
    strategy::AdHocStrategy,
};
use dies_core::{PlayerId, Vector2, Vector3, Angle};
use scenario::ScenarioSetup;
use serde::{Deserialize, Serialize};

// **NOTE**: Add all new scenarios to the `scenarios!` macro at the end of this file.

fn empty_scenario() -> ScenarioSetup {
    ScenarioSetup::new(AdHocStrategy::new())
}

fn one_random_player() -> ScenarioSetup {
    let mut scenario = ScenarioSetup::new(AdHocStrategy::new());
    scenario.add_own_player();
    scenario
}

fn test_vo() -> ScenarioSetup {
    let mut scenario = ScenarioSetup::new(AdHocStrategy::new());
    scenario.add_own_player_at(Vector2::new(-1000.0, -1000.0));
    // scenario.add_own_player_at(Vector2::new(1000.0, 0.0));
    scenario
}

fn two_players_one_ball() -> ScenarioSetup {
    let mut scenario = ScenarioSetup::new(AdHocStrategy::new());
    scenario
        .add_ball()
        .add_own_player_at(Vector2::zeros())
        .add_own_player_at(Vector2::new(-500.0, 0.0));
    scenario
}

fn one_waller_one_ball() -> ScenarioSetup {
    let mut strategy = AdHocStrategy::new();
    strategy.add_role(Box::new(Waller::new(0.0)));
    strategy.add_role(Box::new(Waller::new(500.0)));
    let mut scenario = ScenarioSetup::new(strategy);
    scenario
        // .add_ball_at(Vector3::new(895.0, 2623.0, 0.0))
        .add_ball()
        .add_own_player_at(Vector2::new(2264.0, 336.0))
        .add_own_player_at(Vector2::new(500.0, -336.0))
        .add_own_player_at(Vector2::new(0.0, 0.0));
    scenario
}

fn test_role_multiple_targets() -> ScenarioSetup {
    let mut strategy = AdHocStrategy::new();
    // square
    let targets = vec![
        Vector2::new(-1000.0, -1000.0),
        Vector2::new(1000.0, -1000.0),
        Vector2::new(1000.0, 1000.0),
        Vector2::new(-1000.0, 1000.0),
    ];
    strategy.add_role(Box::new(TestRole::new(targets)));

    let mut scenario = ScenarioSetup::new(strategy);
    scenario.add_own_player_at(Vector2::new(-1000.0, -1000.0));
    scenario
}

fn two_players_crossing() -> ScenarioSetup {
    let mut strategy = AdHocStrategy::new();
    strategy.add_role(Box::new(TestRole::new(vec![Vector2::new(
        -1000.0, -1000.0,
    )])));
    strategy.add_role(Box::new(TestRole::new(vec![Vector2::new(1000.0, 1000.0)])));
    let mut scenario = ScenarioSetup::new(strategy);
    scenario
        .add_own_player_at(Vector2::new(-1000.0, -1000.0))
        .add_own_player_at(Vector2::new(1000.0, 1000.0));
    scenario
}

fn navigate_stationary_opponents() -> ScenarioSetup {
    let mut strategy = AdHocStrategy::new();
    strategy.add_role(Box::new(TestRole::new(vec![Vector2::new(1000.0, 1000.0)])));

    let mut scenario = ScenarioSetup::new(strategy);
    scenario.add_own_player_at(Vector2::new(-1000.0, -1000.0));
    scenario.add_opp_player_at(Vector2::new(-500.0, 0.0));
    // scenario.add_opp_player_at(Vector2::new(0.0, 500.0));
    // scenario.add_opp_player_at(Vector2::new(500.0, -500.0));
    // scenario.add_opp_player_at(Vector2::new(-250.0, 750.0));

    scenario
}

fn rvo_benchmark() -> ScenarioSetup {
    // 6 players on a circle, each navigating to the opposite side
    let mut strategy = AdHocStrategy::new();

    let n = 6;
    let radius = 1000.0;
    let mut targets = Vec::new();
    let mut players = Vec::new();
    for i in 0..n {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
        let p = Vector2::new(radius * angle.cos(), radius * angle.sin());
        targets.push(-p);
        players.push(p);
    }

    for (i, target) in targets.into_iter().enumerate() {
        strategy.add_role_with_id(
            PlayerId::new(i as u32),
            Box::new(TestRole::new(vec![target])),
        );
    }

    let mut scenario = ScenarioSetup::new(strategy);
    for player in players {
        scenario.add_own_player_at(player);
    }

    scenario
}

fn fetch_ball_test () -> ScenarioSetup {
    let mut strategy = AdHocStrategy::new();
    strategy.add_role_with_id(PlayerId::new(0), Box::new(FetcherRole::new()));
    strategy.add_role_with_id(PlayerId::new(1), Box::new(KickerRole::new()));
    let mut scenario = ScenarioSetup::new(strategy);
    scenario
        .add_own_player_at(Vector2::new(-2500.0, -1000.0))
        .add_own_player_at_with_yaw(Vector2::new(100.0, 0.0), Angle::from_radians(PI as f64))
        .add_ball_at(Vector3::new(0.0,0.0, 0.0));

    scenario
}

fn fetch_ball_test_live () -> ScenarioSetup {
    let mut strategy = AdHocStrategy::new();
    strategy.add_role( Box::new(FetcherRole::new()));
    let mut scenario = ScenarioSetup::new(strategy);
    scenario
        .add_own_player();
    scenario
}

fn dribble() -> ScenarioSetup {
    let mut strategy = AdHocStrategy::new();
    strategy.add_role(Box::new(DribbleRole::new()));

    let mut scenario = ScenarioSetup::new(strategy);
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
    one_random_player,
    test_vo,
    two_players_one_ball,
    one_waller_one_ball,
    two_players_crossing,
    test_role_multiple_targets,
    navigate_stationary_opponents,
    dribble,
    rvo_benchmark,
    fetch_ball_test,
    fetch_ball_test_live
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
