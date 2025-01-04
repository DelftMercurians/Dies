use dies_core::{
    ColoredPlayerId, DiesInstant, PlayerId, RobotCmd, SideAssignment, TeamColor, WorldFrame,
};
use dies_tracker::TrackerState;

use crate::{
    control::TeamController, player_override::PlayerOverrideCommand,
    strategy_instance::StrategyInstance, team_frame::TeamFrame, utils::SideTransform,
};

pub struct ControlledTeam {
    controller: TeamController,
    strategy: StrategyInstance,
}

enum Team {
    Controlled(ControlledTeam),
    Uncontrolled,
}

impl Team {
    fn update(&mut self, update: &WorldFrame, team_color: TeamColor, time: DiesInstant) {
        let transform = SideTransform::new(team_color, update.side_assignment);
        let team_frame = TeamFrame::from_world_frame(update, team_color);
        match self {
            Team::Controlled(team) => {
                todo!()
            }
            Team::Uncontrolled => {}
        }
    }

    fn override_player(&mut self, player_id: PlayerId, cmd: PlayerOverrideCommand) {
        match self {
            Team::Controlled(controlled) => {
                todo!()
            }
            Team::Uncontrolled => {}
        }
    }

    fn commands(&mut self) -> Vec<(PlayerId, RobotCmd)> {
        match self {
            Team::Controlled(controlled) => controlled.controller.commands(),
            Team::Uncontrolled => vec![],
        }
    }
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
                        team.controller.set_opp_goal_sign(-1.0);
                    }
                }
                Team::Controlled(team)
            }
            TeamConfig::Uncontrolled => Team::Uncontrolled,
        };
        Self { blue, yellow }
    }

    pub fn update(&mut self, update: TrackerState, time: DiesInstant) {
        self.blue.update(update, TeamColor::Blue, time);
        self.yellow.update(update, TeamColor::Yellow, time);
    }

    pub fn override_player(&mut self, player_id: ColoredPlayerId, cmd: PlayerOverrideCommand) {
        match player_id.team() {
            TeamColor::Blue => self.blue.override_player(player_id.id(), cmd),
            TeamColor::Yellow => self.yellow.override_player(player_id.id(), cmd),
        }
    }

    pub fn commands(&mut self) -> Vec<(ColoredPlayerId, RobotCmd)> {
        self.blue
            .commands()
            .iter()
            .map(|(id, cmd)| (ColoredPlayerId(TeamColor::Blue, *id), *cmd))
            .chain(
                self.yellow
                    .commands()
                    .iter()
                    .map(|(id, cmd)| (ColoredPlayerId(TeamColor::Yellow, *id), *cmd)),
            )
            .collect()
    }
}
