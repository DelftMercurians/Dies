use std::collections::HashMap;

use dies_core::{ExecutorSettings, PlayerCmd, PlayerId, PlayerOverrideCommand, VisionMsg};
use dies_protos::ssl_gc_referee_message::Referee;
use dies_world::WorldTracker;
use dies_world::{world::WorldInstant, WorldFrame};

use crate::{control::TeamController, strategy_instance::StrategyInstance};

pub struct ControlledTeam {
    controller: TeamController,
    strategy: StrategyInstance,
}

enum Team {
    Controlled(ControlledTeam),
    Uncontrolled,
}

impl Team {
    fn update(&mut self, update: &WorldFrame, time: WorldInstant) {
        let _ = update;
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

pub enum SideAssignment {
    BluePositive,
    YellowPositive,
}

pub enum TeamConfig {
    Controlled { strategy: StrategyInstance },
    Uncontrolled,
}

pub struct TeamMapConfig {
    blue: TeamConfig,
    yellow: TeamConfig,
    side_assignment: SideAssignment,
}

pub struct TeamMap {
    blue: Team,
    yellow: Team,
    side_assignment: SideAssignment,
}

impl TeamMap {
    pub fn new(config: TeamMapConfig, settings: &ExecutorSettings) -> Self {
        let blue = match config.blue {
            TeamConfig::Controlled { strategy } => {
                let mut team = ControlledTeam {
                    controller: TeamController::new(settings),
                    strategy,
                };
                match config.side_assignment {
                    SideAssignment::BluePositive => {
                        team.tracker.set_play_dir_x(-1.0);
                        team.controller.set_opp_goal_sign(-1.0);
                    }
                    SideAssignment::YellowPositive => {}
                }
                Team::Controlled(team)
            }
            TeamConfig::Uncontrolled => Team::Uncontrolled,
        };
        let yellow = match config.yellow {
            TeamConfig::Controlled { strategy } => {
                let mut team = ControlledTeam {
                    controller: TeamController::new(settings),
                    strategy,
                };
                match config.side_assignment {
                    SideAssignment::BluePositive => {}
                    SideAssignment::YellowPositive => {
                        team.tracker.set_play_dir_x(-1.0);
                        team.controller.set_opp_goal_sign(-1.0);
                    }
                }
                Team::Controlled(team)
            }
            TeamConfig::Uncontrolled => Team::Uncontrolled,
        };
        Self {
            blue,
            yellow,
            side_assignment: config.side_assignment,
        }
    }

    pub fn update(&mut self, update: GameUpdate, time: WorldInstant) {
        self.blue.update(&update, time);
        self.yellow.update(&update, time);
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
