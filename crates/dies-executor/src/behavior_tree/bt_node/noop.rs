#[derive(Clone)]
pub struct NoopNode;

impl NoopNode {
    pub fn new() -> Self {
        Self {}
    }
}
