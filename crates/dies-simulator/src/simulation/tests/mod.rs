//! Original unit tests, moved without changing their assertions.
use super::*;
use crate::ball::BALL_RADIUS;
use crate::referee::{FREE_KICK_DEFENSE_AREA_DISTANCE, FREE_KICK_PLACEMENT_MARGIN};
use crate::{RealismConfig, RealismTeams, SimulationBuilder, SimulationGameState};
use dies_core::RobotCmd;
use std::f64::consts::PI;

mod reflex_kick_tests;

mod realism_tests;

mod free_kick_placement_tests;

mod ball_out_tests;

mod determinism_tests;

mod snatch_peel_tests;
