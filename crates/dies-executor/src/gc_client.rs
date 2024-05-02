use dies_core::GcRefereeMsg;

pub struct GcClient {}

impl GcClient {
    pub fn new() -> Self {
        Self {}
    }

    #[allow(dead_code)]
    pub fn handle_message(&mut self, message: GcRefereeMsg) {
        tracing::info!("Received message from GC: {:?}", message);
    }

    pub fn messages(&mut self) -> Vec<GcRefereeMsg> {
        vec![]
    }
}
