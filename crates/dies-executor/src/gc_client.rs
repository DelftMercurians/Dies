use dies_core::GcRefereeMsg;

pub struct GcClient {}

impl GcClient {
    pub fn new() -> Self {
        Self {}
    }

    pub fn messages(&mut self) -> Vec<GcRefereeMsg> {
        vec![]
    }
}
