mod scenario;

use std::vec;

use dies_core::Vector2;
use scenario::ScenarioSetup;
use serde::{Deserialize, Serialize};

// **NOTE**: Add all new scenarios to the `scenarios!` macro at the end of this file.

fn empty_scenario() -> ScenarioSetup {
    ScenarioSetup::new()
}

fn attack() -> ScenarioSetup {
    let mut setup = ScenarioSetup::new();
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
    empty_scenario
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
