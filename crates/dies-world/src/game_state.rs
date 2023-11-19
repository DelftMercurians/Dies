use dies_protos::ssl_gc_referee_message::referee::Command;

pub enum GameState {
    Halt,
    UNKNOWN,
    Timeout,
    Stop,
    PrepareKickoff,
    BallReplacement,
    PreparePenalty,
    Kickoff,
    FreeKick,
    Penalty,
    Run,
}


#[derive(Debug)]
pub struct GameStateTracker {
    game_state: GameState,
}

impl GameStateTracker {
    pub fn new() -> GameStateTracker {
        GameStateTracker {
            game_state: GameState::UNKNOWN,
        }
    }

    pub fn update(&mut self, command: &Command) {
        let mut new_game_state = GameState::Halt;


        self.game_state = new_game_state;
    }

    pub fn get_game_state(&self) -> GameState {
        self.game_state.clone()
    }
}
