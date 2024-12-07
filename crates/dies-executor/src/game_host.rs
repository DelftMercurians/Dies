use std::collections::HashMap;

use dies_core::{PlayerCmd, PlayerId, PlayerOverrideCommand, VisionMsg, WorldInstant, WorldUpdate};
use dies_protos::ssl_gc_referee_message::Referee;
use dies_world::WorldTracker;

use crate::control::TeamController;

enum StrategyInstance {
    Process,
}

impl StrategyInstance {
    fn send_update(&mut self, _update: WorldUpdate) {
        todo!()
    }

    fn recv(&mut self) {
        todo!()
    }
}

enum TeamColor {
    Blue,
    Yellow,
}

enum GameUpdate {
    Vision(VisionMsg),
    Gc(Referee),
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct TeamId(u8);

impl TeamId {
    pub fn new_pair() -> (TeamId, TeamId) {
        (TeamId(0), TeamId(1))
    }
}

pub struct ControlledTeam {
    tracker: WorldTracker,
    controller: TeamController,
    strategy: StrategyInstance,
}

enum Team {
    Controlled(ControlledTeam),
    Uncontrolled,
}

impl Team {
    fn update(&mut self, update: GameUpdate, time: WorldInstant) {
        match self {
            Team::Controlled(team) => {
                // team.tracker.update(update, time);
                team.controller.update(team.tracker.get(), HashMap::new());
                // team.strategy.send_update(team.tracker.get());
            }
            Team::Uncontrolled => {}
        }
    }

    fn override_player(&mut self, player_id: PlayerId, cmd: PlayerOverrideCommand) {
        match self {
            Team::Controlled(controlled) => {
                // controlled.controller.override_player(player_id, cmd);
            }
            Team::Uncontrolled => {}
        }
    }

    fn commands(&mut self) -> Vec<PlayerCmd> {
        match self {
            Team::Controlled(controlled) => controlled.controller.commands(),
            Team::Uncontrolled => vec![],
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum SideAssignment {
    BluePositive,
    YellowPositive,
}

impl SideAssignment {
    pub fn flip(&mut self) {
        *self = match self {
            SideAssignment::BluePositive => SideAssignment::YellowPositive,
            SideAssignment::YellowPositive => SideAssignment::BluePositive,
        }
    }
}

pub struct TeamMap {
    blue: Team,
    yellow: Team,
    side_assignment: SideAssignment,
}

impl TeamMap {
    pub fn update(&mut self, update: GameUpdate, time: WorldInstant) {
        // self.blue.update(update, time);
        // self.yellow.update(update, time);
    }

    pub fn override_player(&mut self, player_id: PlayerId, cmd: PlayerOverrideCommand) {
        todo!()
    }

    pub fn commands(&mut self) -> Vec<PlayerCmd> {
        let mut commands = self.blue.commands();
        commands.extend(self.yellow.commands());
        commands
    }
}
