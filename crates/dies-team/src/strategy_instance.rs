use dies_core::WorldUpdate;

pub enum StrategyInstance {
    Process,
}

impl StrategyInstance {
    fn new() -> Self {
        todo!()
    }

    pub fn send_update(&mut self, _update: WorldUpdate) {
        todo!()
    }

    pub fn recv(&mut self) {
        todo!()
    }
}
