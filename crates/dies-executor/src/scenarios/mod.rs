mod scenario;
use scenario::ScenarioSetup;

use crate::strategy::AdHocStrategy;

fn empty_simulation() -> ScenarioSetup {
    ScenarioSetup::new(AdHocStrategy::new())
}

/// Creates a lookup table for scenarios as a global constant.
macro_rules! scenarios {
    ($($scen:ident),+) => {
        use paste::paste;

        paste! {
            pub enum Scenarios {
                $([< $scen:camel >]),+
            }
        }

        impl Scenarios {
            pub fn into_setup(&self) -> ScenarioSetup {
                paste! {
                    let f = match self {
                        $(Scenarios::[< $scen:camel >] => Box::new($scen)),+
                    };
                }
                f()
            }

            pub fn get_names() -> Vec<&'static str> {
                vec![$(stringify!($scen)),+]
            }

            pub fn get_setup_by_name(name: &str) -> Option<ScenarioSetup> {
                paste! {
                    match name {
                        $(stringify!($scen) => Some(Scenarios::[< $scen:camel >].into_setup()),)+
                        _ => None
                    }
                }
            }
        }
    };
}

// **NOTE**: Add new scenarios here.
scenarios! {
    empty_simulation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scenarios() {
        let names = Scenarios::get_names();
        assert_eq!(names[0], "empty_simulation");

        Scenarios::get_setup_by_name("empty_simulation").expect("empty_simulation not found");
    }
}
