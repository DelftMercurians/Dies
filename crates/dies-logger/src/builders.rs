//! Per-table Arrow column builders. Each builder accumulates rows and produces
//! a `RecordBatch` on `finish()` (which also resets it for the next batch).
//!
//! These live on the worker thread; the executor only ever sends the plain row
//! structs from `frame.rs`.

use std::sync::Arc;

use anyhow::Result;
use arrow::array::{
    ArrayRef, BinaryBuilder, BooleanBuilder, Float32Builder, Float64Builder, ListBuilder,
    StringBuilder, UInt32Builder, UInt64Builder,
};
use arrow::record_batch::RecordBatch;

use crate::frame::{
    BallRow, DebugShapeRow, DebugTreeRow, DebugValueRow, EventRecord, FrameRecord, LogLineRecord,
    MarkerRecord, PlayerRow, RawRecord,
};
use crate::schema;

#[derive(Default)]
pub struct FramesBuilder {
    len: usize,
    frame_id: UInt64Builder,
    t_received: Float64Builder,
    t_capture: Float64Builder,
    dt: Float64Builder,
    game_state: StringBuilder,
    operating_team: StringBuilder,
    side_assignment: StringBuilder,
    ball_on_blue_side: Float64Builder,
    ball_on_yellow_side: Float64Builder,
}

impl FramesBuilder {
    pub fn push(&mut self, r: &FrameRecord) {
        self.frame_id.append_value(r.frame_id);
        self.t_received.append_value(r.t_received);
        self.t_capture.append_value(r.t_capture);
        self.dt.append_value(r.dt);
        self.game_state.append_value(&r.game_state);
        self.operating_team.append_value(&r.operating_team);
        self.side_assignment.append_value(&r.side_assignment);
        self.ball_on_blue_side.append_option(r.ball_on_blue_side);
        self.ball_on_yellow_side
            .append_option(r.ball_on_yellow_side);
        self.len += 1;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn finish(&mut self) -> Result<RecordBatch> {
        let cols: Vec<ArrayRef> = vec![
            Arc::new(self.frame_id.finish()),
            Arc::new(self.t_received.finish()),
            Arc::new(self.t_capture.finish()),
            Arc::new(self.dt.finish()),
            Arc::new(self.game_state.finish()),
            Arc::new(self.operating_team.finish()),
            Arc::new(self.side_assignment.finish()),
            Arc::new(self.ball_on_blue_side.finish()),
            Arc::new(self.ball_on_yellow_side.finish()),
        ];
        self.len = 0;
        Ok(RecordBatch::try_new(schema::frames(), cols)?)
    }
}

#[derive(Default)]
pub struct BallBuilder {
    len: usize,
    frame_id: UInt64Builder,
    x: Float64Builder,
    y: Float64Builder,
    z: Float64Builder,
    vx: Float64Builder,
    vy: Float64Builder,
    vz: Float64Builder,
    detected: BooleanBuilder,
}

impl BallBuilder {
    pub fn push(&mut self, frame_id: u64, r: &BallRow) {
        self.frame_id.append_value(frame_id);
        self.x.append_value(r.x);
        self.y.append_value(r.y);
        self.z.append_value(r.z);
        self.vx.append_value(r.vx);
        self.vy.append_value(r.vy);
        self.vz.append_value(r.vz);
        self.detected.append_value(r.detected);
        self.len += 1;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn finish(&mut self) -> Result<RecordBatch> {
        let cols: Vec<ArrayRef> = vec![
            Arc::new(self.frame_id.finish()),
            Arc::new(self.x.finish()),
            Arc::new(self.y.finish()),
            Arc::new(self.z.finish()),
            Arc::new(self.vx.finish()),
            Arc::new(self.vy.finish()),
            Arc::new(self.vz.finish()),
            Arc::new(self.detected.finish()),
        ];
        self.len = 0;
        Ok(RecordBatch::try_new(schema::ball(), cols)?)
    }
}

#[derive(Default)]
pub struct PlayersBuilder {
    len: usize,
    frame_id: UInt64Builder,
    team: StringBuilder,
    player_id: UInt32Builder,
    x: Float64Builder,
    y: Float64Builder,
    vx: Float64Builder,
    vy: Float64Builder,
    yaw: Float64Builder,
    raw_yaw: Float64Builder,
    angular_speed: Float64Builder,
    position_noise: Float64Builder,
    primary_status: StringBuilder,
    kicker_cap_voltage: Float32Builder,
    kicker_temp: Float32Builder,
    pack_voltage_0: Float32Builder,
    pack_voltage_1: Float32Builder,
    breakbeam_ball_detected: BooleanBuilder,
    imu_status: StringBuilder,
    kicker_status: StringBuilder,
    handicaps: StringBuilder,
}

impl PlayersBuilder {
    pub fn push(&mut self, frame_id: u64, r: &PlayerRow) {
        self.frame_id.append_value(frame_id);
        self.team.append_value(r.team);
        self.player_id.append_value(r.player_id);
        self.x.append_value(r.x);
        self.y.append_value(r.y);
        self.vx.append_value(r.vx);
        self.vy.append_value(r.vy);
        self.yaw.append_value(r.yaw);
        self.raw_yaw.append_value(r.raw_yaw);
        self.angular_speed.append_value(r.angular_speed);
        self.position_noise.append_value(r.position_noise);
        self.primary_status
            .append_option(r.primary_status.as_deref());
        self.kicker_cap_voltage.append_option(r.kicker_cap_voltage);
        self.kicker_temp.append_option(r.kicker_temp);
        self.pack_voltage_0.append_option(r.pack_voltage_0);
        self.pack_voltage_1.append_option(r.pack_voltage_1);
        self.breakbeam_ball_detected
            .append_value(r.breakbeam_ball_detected);
        self.imu_status.append_option(r.imu_status.as_deref());
        self.kicker_status.append_option(r.kicker_status.as_deref());
        self.handicaps.append_value(&r.handicaps);
        self.len += 1;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn finish(&mut self) -> Result<RecordBatch> {
        let cols: Vec<ArrayRef> = vec![
            Arc::new(self.frame_id.finish()),
            Arc::new(self.team.finish()),
            Arc::new(self.player_id.finish()),
            Arc::new(self.x.finish()),
            Arc::new(self.y.finish()),
            Arc::new(self.vx.finish()),
            Arc::new(self.vy.finish()),
            Arc::new(self.yaw.finish()),
            Arc::new(self.raw_yaw.finish()),
            Arc::new(self.angular_speed.finish()),
            Arc::new(self.position_noise.finish()),
            Arc::new(self.primary_status.finish()),
            Arc::new(self.kicker_cap_voltage.finish()),
            Arc::new(self.kicker_temp.finish()),
            Arc::new(self.pack_voltage_0.finish()),
            Arc::new(self.pack_voltage_1.finish()),
            Arc::new(self.breakbeam_ball_detected.finish()),
            Arc::new(self.imu_status.finish()),
            Arc::new(self.kicker_status.finish()),
            Arc::new(self.handicaps.finish()),
        ];
        self.len = 0;
        Ok(RecordBatch::try_new(schema::players(), cols)?)
    }
}

#[derive(Default)]
pub struct DebugValuesBuilder {
    len: usize,
    frame_id: UInt64Builder,
    key: StringBuilder,
    value: Float64Builder,
    value_str: StringBuilder,
}

impl DebugValuesBuilder {
    pub fn push(&mut self, frame_id: u64, r: &DebugValueRow) {
        self.frame_id.append_value(frame_id);
        self.key.append_value(&r.key);
        self.value.append_option(r.value);
        self.value_str.append_option(r.value_str.as_deref());
        self.len += 1;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn finish(&mut self) -> Result<RecordBatch> {
        let cols: Vec<ArrayRef> = vec![
            Arc::new(self.frame_id.finish()),
            Arc::new(self.key.finish()),
            Arc::new(self.value.finish()),
            Arc::new(self.value_str.finish()),
        ];
        self.len = 0;
        Ok(RecordBatch::try_new(schema::debug_values(), cols)?)
    }
}

#[derive(Default)]
pub struct DebugShapesBuilder {
    len: usize,
    frame_id: UInt64Builder,
    key: StringBuilder,
    shape_type: StringBuilder,
    cx: Float64Builder,
    cy: Float64Builder,
    radius: Float64Builder,
    x1: Float64Builder,
    y1: Float64Builder,
    x2: Float64Builder,
    y2: Float64Builder,
    color: StringBuilder,
    fill: StringBuilder,
    stroke: StringBuilder,
}

impl DebugShapesBuilder {
    pub fn push(&mut self, frame_id: u64, r: &DebugShapeRow) {
        self.frame_id.append_value(frame_id);
        self.key.append_value(&r.key);
        self.shape_type.append_value(r.shape_type);
        self.cx.append_option(r.cx);
        self.cy.append_option(r.cy);
        self.radius.append_option(r.radius);
        self.x1.append_option(r.x1);
        self.y1.append_option(r.y1);
        self.x2.append_option(r.x2);
        self.y2.append_option(r.y2);
        self.color.append_option(r.color.as_deref());
        self.fill.append_option(r.fill.as_deref());
        self.stroke.append_option(r.stroke.as_deref());
        self.len += 1;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn finish(&mut self) -> Result<RecordBatch> {
        let cols: Vec<ArrayRef> = vec![
            Arc::new(self.frame_id.finish()),
            Arc::new(self.key.finish()),
            Arc::new(self.shape_type.finish()),
            Arc::new(self.cx.finish()),
            Arc::new(self.cy.finish()),
            Arc::new(self.radius.finish()),
            Arc::new(self.x1.finish()),
            Arc::new(self.y1.finish()),
            Arc::new(self.x2.finish()),
            Arc::new(self.y2.finish()),
            Arc::new(self.color.finish()),
            Arc::new(self.fill.finish()),
            Arc::new(self.stroke.finish()),
        ];
        self.len = 0;
        Ok(RecordBatch::try_new(schema::debug_shapes(), cols)?)
    }
}

pub struct DebugTreeBuilder {
    len: usize,
    frame_id: UInt64Builder,
    key: StringBuilder,
    name: StringBuilder,
    node_id: StringBuilder,
    children_ids: ListBuilder<StringBuilder>,
    is_active: BooleanBuilder,
    node_type: StringBuilder,
    internal_state: StringBuilder,
    additional_info: StringBuilder,
}

impl Default for DebugTreeBuilder {
    fn default() -> Self {
        Self {
            len: 0,
            frame_id: UInt64Builder::new(),
            key: StringBuilder::new(),
            name: StringBuilder::new(),
            node_id: StringBuilder::new(),
            children_ids: ListBuilder::new(StringBuilder::new()),
            is_active: BooleanBuilder::new(),
            node_type: StringBuilder::new(),
            internal_state: StringBuilder::new(),
            additional_info: StringBuilder::new(),
        }
    }
}

impl DebugTreeBuilder {
    pub fn push(&mut self, frame_id: u64, r: &DebugTreeRow) {
        self.frame_id.append_value(frame_id);
        self.key.append_value(&r.key);
        self.name.append_value(&r.name);
        self.node_id.append_value(&r.node_id);
        for child in &r.children_ids {
            self.children_ids.values().append_value(child);
        }
        self.children_ids.append(true);
        self.is_active.append_value(r.is_active);
        self.node_type.append_value(&r.node_type);
        self.internal_state
            .append_option(r.internal_state.as_deref());
        self.additional_info
            .append_option(r.additional_info.as_deref());
        self.len += 1;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn finish(&mut self) -> Result<RecordBatch> {
        let cols: Vec<ArrayRef> = vec![
            Arc::new(self.frame_id.finish()),
            Arc::new(self.key.finish()),
            Arc::new(self.name.finish()),
            Arc::new(self.node_id.finish()),
            Arc::new(self.children_ids.finish()),
            Arc::new(self.is_active.finish()),
            Arc::new(self.node_type.finish()),
            Arc::new(self.internal_state.finish()),
            Arc::new(self.additional_info.finish()),
        ];
        self.len = 0;
        Ok(RecordBatch::try_new(schema::debug_tree(), cols)?)
    }
}

#[derive(Default)]
pub struct SettingsChangesBuilder {
    len: usize,
    frame_id: UInt64Builder,
    t: Float64Builder,
    key: StringBuilder,
    value_num: Float64Builder,
    value_str: StringBuilder,
}

impl SettingsChangesBuilder {
    pub fn push(
        &mut self,
        frame_id: u64,
        t: f64,
        key: &str,
        value_num: Option<f64>,
        value_str: Option<&str>,
    ) {
        self.frame_id.append_value(frame_id);
        self.t.append_value(t);
        self.key.append_value(key);
        self.value_num.append_option(value_num);
        self.value_str.append_option(value_str);
        self.len += 1;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn finish(&mut self) -> Result<RecordBatch> {
        let cols: Vec<ArrayRef> = vec![
            Arc::new(self.frame_id.finish()),
            Arc::new(self.t.finish()),
            Arc::new(self.key.finish()),
            Arc::new(self.value_num.finish()),
            Arc::new(self.value_str.finish()),
        ];
        self.len = 0;
        Ok(RecordBatch::try_new(schema::settings_changes(), cols)?)
    }
}

#[derive(Default)]
pub struct EventsBuilder {
    len: usize,
    frame_id: UInt64Builder,
    t: Float64Builder,
    event_type: StringBuilder,
    payload_json: StringBuilder,
}

impl EventsBuilder {
    pub fn push(&mut self, r: &EventRecord) {
        self.frame_id.append_value(r.frame_id);
        self.t.append_value(r.t);
        self.event_type.append_value(&r.event_type);
        self.payload_json.append_value(&r.payload_json);
        self.len += 1;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn finish(&mut self) -> Result<RecordBatch> {
        let cols: Vec<ArrayRef> = vec![
            Arc::new(self.frame_id.finish()),
            Arc::new(self.t.finish()),
            Arc::new(self.event_type.finish()),
            Arc::new(self.payload_json.finish()),
        ];
        self.len = 0;
        Ok(RecordBatch::try_new(schema::events(), cols)?)
    }
}

#[derive(Default)]
pub struct MarkersBuilder {
    len: usize,
    frame_id: UInt64Builder,
    t: Float64Builder,
    label: StringBuilder,
}

impl MarkersBuilder {
    pub fn push(&mut self, r: &MarkerRecord) {
        self.frame_id.append_value(r.frame_id);
        self.t.append_value(r.t);
        self.label.append_option(r.label.as_deref());
        self.len += 1;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn finish(&mut self) -> Result<RecordBatch> {
        let cols: Vec<ArrayRef> = vec![
            Arc::new(self.frame_id.finish()),
            Arc::new(self.t.finish()),
            Arc::new(self.label.finish()),
        ];
        self.len = 0;
        Ok(RecordBatch::try_new(schema::markers(), cols)?)
    }
}

#[derive(Default)]
pub struct LogsBuilder {
    len: usize,
    t: Float64Builder,
    level: StringBuilder,
    target: StringBuilder,
    source: StringBuilder,
    message: StringBuilder,
}

impl LogsBuilder {
    pub fn push(&mut self, r: &LogLineRecord) {
        self.t.append_value(r.t);
        self.level.append_value(&r.level);
        self.target.append_value(&r.target);
        self.source.append_value(&r.source);
        self.message.append_value(&r.message);
        self.len += 1;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn finish(&mut self) -> Result<RecordBatch> {
        let cols: Vec<ArrayRef> = vec![
            Arc::new(self.t.finish()),
            Arc::new(self.level.finish()),
            Arc::new(self.target.finish()),
            Arc::new(self.source.finish()),
            Arc::new(self.message.finish()),
        ];
        self.len = 0;
        Ok(RecordBatch::try_new(schema::logs(), cols)?)
    }
}

pub struct VisionBuilder {
    len: usize,
    frame_id: UInt64Builder,
    t: Float64Builder,
    kind: StringBuilder,
    bytes: BinaryBuilder,
}

impl Default for VisionBuilder {
    fn default() -> Self {
        Self {
            len: 0,
            frame_id: UInt64Builder::new(),
            t: Float64Builder::new(),
            kind: StringBuilder::new(),
            bytes: BinaryBuilder::new(),
        }
    }
}

impl VisionBuilder {
    pub fn push(&mut self, r: &RawRecord) {
        self.frame_id.append_value(r.frame_id);
        self.t.append_value(r.t);
        self.kind.append_value(&r.kind);
        self.bytes.append_value(&r.bytes);
        self.len += 1;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn finish(&mut self) -> Result<RecordBatch> {
        let cols: Vec<ArrayRef> = vec![
            Arc::new(self.frame_id.finish()),
            Arc::new(self.t.finish()),
            Arc::new(self.kind.finish()),
            Arc::new(self.bytes.finish()),
        ];
        self.len = 0;
        Ok(RecordBatch::try_new(schema::vision(), cols)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dies_core::{mock_world_data, DebugMap};

    #[test]
    fn frame_tables_roundtrip_to_batches() {
        let world = mock_world_data();
        let debug = DebugMap::new();
        let rec = FrameRecord::from_world(1, &world, &debug);

        let mut frames = FramesBuilder::default();
        let mut ball = BallBuilder::default();
        let mut players = PlayersBuilder::default();

        frames.push(&rec);
        if let Some(b) = &rec.ball {
            ball.push(rec.frame_id, b);
        }
        for p in &rec.players {
            players.push(rec.frame_id, p);
        }

        assert_eq!(frames.len(), 1);
        assert_eq!(players.len(), 2);

        let fb = frames.finish().unwrap();
        assert_eq!(fb.num_rows(), 1);
        assert_eq!(fb.schema(), schema::frames());

        let pb = players.finish().unwrap();
        assert_eq!(pb.num_rows(), 2);

        // builders reset after finish
        assert_eq!(frames.len(), 0);
    }

    #[test]
    fn debug_tree_list_column_roundtrips() {
        let mut tree = DebugTreeBuilder::default();
        tree.push(
            42,
            &DebugTreeRow {
                key: "root".into(),
                name: "Root".into(),
                node_id: "n0".into(),
                children_ids: vec!["n1".into(), "n2".into()],
                is_active: true,
                node_type: "selector".into(),
                internal_state: None,
                additional_info: Some("info".into()),
            },
        );
        let batch = tree.finish().unwrap();
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.schema(), schema::debug_tree());
    }
}
