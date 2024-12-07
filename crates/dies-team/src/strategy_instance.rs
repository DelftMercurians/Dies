use dies_core::WorldUpdate;

pub enum StrategyInstance {
    Process,
}

impl StrategyInstance {
    fn new() -> Self {
        let (rx, tx) = tokio::sync::mpsc::channel(100);
        tokio::task::spawn_blocking(|| {
            loop {
                let msg = rx from process
                tx.send(msg);
            }
        });
    }

    pub fn send_update(&mut self, _update: WorldUpdate) {
        todo!()
    }

    pub fn recv(&mut self) {
        self.rx.recv().unwrap()
    }
}
