use dies_core::FieldGeometry;
use dies_core::WorldFrame;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct TrackerFrame {
    frame: Arc<WorldFrame>,
    field: Option<Arc<FieldGeometry>>,
}

impl TrackerFrame {
    pub(crate) fn new(frame: Arc<WorldFrame>, field: Option<Arc<FieldGeometry>>) -> TrackerFrame {
        TrackerFrame { frame, field }
    }

    pub fn world_frame(&self) -> &WorldFrame {
        self.frame.as_ref()
    }

    pub fn field_geometry(&self) -> Option<&FieldGeometry> {
        self.field.as_ref().map(|f| f.as_ref())
    }
}
