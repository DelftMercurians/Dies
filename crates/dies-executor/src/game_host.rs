use dies_core::{PlayerCmd, PlayerId, PlayerOverrideCommand, VisionMsg};
use dies_protos::ssl_gc_referee_message::Referee;

enum StrategyInstance {
    Process,
    Script,
}

enum TeamColor {
    Blue,
    Yellow,
}

enum GameUpdate {
    Vision(VisionMsg),
    Gc(Referee),
}

pub struct GameHost {}

impl GameHost {
    pub fn new() -> Self {
        GameHost {}
    }

    pub fn update(&mut self, update: GameUpdate) {}

    pub fn override_player(&mut self, player_id: PlayerId, cmd: PlayerOverrideCommand) {}

    pub fn commands(&mut self) -> Vec<PlayerCmd> {
        vec![]
    }
}
