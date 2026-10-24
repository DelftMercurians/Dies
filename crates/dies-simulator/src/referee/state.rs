//! Match state and state-specific transitions.
use super::*;
use crate::time::Timer;
/// How long the game sits in a `Stop` (robots slowed, ball being placed) before
/// resuming after the ball leaves the field, a goal, or a no-progress stoppage.
/// Tune this to make self-play matches more/less leisurely.
pub(crate) const STOP_DURATION: f64 = 2.0;
/// How far the ball must travel from its last reference point to count as
/// "progress" and reset the no-progress timer (rule 8.1). Measured against a
/// sticky reference over the whole 10 s window, not frame-to-frame, so a slow
/// dribble still counts as progress and only a genuinely stuck ball trips the
/// timer.
pub(crate) const NO_PROGRESS_DISTANCE: f64 = 200.0;

#[derive(Debug, Clone, Default)]
pub enum SimulationGameState {
    #[default]
    StartGame,
    Stop,
    Halt,
    StopAndKickOff {
        team_color: TeamColor,
        stop_timer: Timer,
    },
    StopAndForceStart {
        stop_timer: Timer,
    },
    StopAndFreeKick {
        team_color: TeamColor,
        stop_timer: Timer,
    },
    PrepareKickOff {
        team_color: TeamColor,
        prepare_timer: Timer,
    },
    Run {
        wait_timer: Timer,
        no_progress_timer: Timer,
        last_ball_position: Option<Vector2>,
    },
    FreeKick {
        kick_timer: Timer,
        ball_is_kicked: bool,
        team_color: TeamColor,
    },
    Penalty {
        team_color: TeamColor,
    },
    BallPlacement {
        team_color: TeamColor,
        position: Vector2,
    },
    Null,
}

impl SimulationGameState {
    pub(crate) fn run() -> Self {
        SimulationGameState::Run {
            wait_timer: Timer::new(0.2),
            no_progress_timer: Timer::new(10.0),
            last_ball_position: None,
        }
    }

    pub(crate) fn stop() -> Self {
        SimulationGameState::Stop
    }

    pub(crate) fn stop_and_kickoff(team_color: TeamColor) -> Self {
        SimulationGameState::StopAndKickOff {
            stop_timer: Timer::new(STOP_DURATION),
            team_color,
        }
    }

    pub(crate) fn stop_and_force_start() -> Self {
        SimulationGameState::StopAndForceStart {
            stop_timer: Timer::new(STOP_DURATION),
        }
    }

    pub(crate) fn stop_and_free_kick(team_color: TeamColor) -> Self {
        SimulationGameState::StopAndFreeKick {
            stop_timer: Timer::new(STOP_DURATION),
            team_color,
        }
    }

    pub(crate) fn prepare_kickoff(team_color: TeamColor) -> Self {
        SimulationGameState::PrepareKickOff {
            team_color,
            // Hard ceiling so a kickoff can never stall the match if positions
            // never become valid; matches the referee NORMAL_START schedule.
            prepare_timer: Timer::new(10.0),
        }
    }

    pub(crate) fn free_kick(team_color: TeamColor) -> Self {
        SimulationGameState::FreeKick {
            kick_timer: Timer::new(10.0),
            ball_is_kicked: false,
            team_color,
        }
    }

    pub(crate) fn penalty(team_color: TeamColor) -> Self {
        SimulationGameState::Penalty { team_color }
    }

    pub(crate) fn ball_placement(team_color: TeamColor, position: Vector2) -> Self {
        SimulationGameState::BallPlacement {
            team_color,
            position,
        }
    }
}

impl PartialEq for SimulationGameState {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (SimulationGameState::StartGame, SimulationGameState::StartGame) => true,
            (SimulationGameState::Stop, SimulationGameState::Stop) => true,
            (SimulationGameState::Halt, SimulationGameState::Halt) => true,
            (
                SimulationGameState::StopAndKickOff { .. },
                SimulationGameState::StopAndKickOff { .. },
            ) => true,
            (
                SimulationGameState::StopAndFreeKick { .. },
                SimulationGameState::StopAndFreeKick { .. },
            ) => true,
            (
                SimulationGameState::PrepareKickOff { .. },
                SimulationGameState::PrepareKickOff { .. },
            ) => true,
            (SimulationGameState::Run { .. }, SimulationGameState::Run { .. }) => true,
            (SimulationGameState::FreeKick { .. }, SimulationGameState::FreeKick { .. }) => true,
            (SimulationGameState::Penalty { .. }, SimulationGameState::Penalty { .. }) => true,
            (
                SimulationGameState::BallPlacement { .. },
                SimulationGameState::BallPlacement { .. },
            ) => true,
            _ => false,
        }
    }
}

impl fmt::Display for SimulationGameState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimulationGameState::StartGame => write!(f, "Start game"),
            SimulationGameState::Stop => write!(f, "Stop"),
            SimulationGameState::Halt => write!(f, "Halt"),
            SimulationGameState::StopAndKickOff { .. } => write!(f, "Stop and kick off"),
            SimulationGameState::StopAndForceStart { .. } => write!(f, "Stop and force start"),
            SimulationGameState::StopAndFreeKick { .. } => write!(f, "Stop and free kick"),
            SimulationGameState::PrepareKickOff { .. } => write!(f, "Prepare kick off"),
            SimulationGameState::Run { .. } => write!(f, "Run"),
            SimulationGameState::FreeKick { .. } => write!(f, "Free kick"),
            SimulationGameState::Penalty { .. } => write!(f, "Penalty"),
            SimulationGameState::BallPlacement { .. } => write!(f, "Ball placement"),
            SimulationGameState::Null => write!(f, "Null state"),
        }
    }
}
