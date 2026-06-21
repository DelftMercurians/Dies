//! Plain owned row structs sent over the worker channel, plus the projection
//! from a live `WorldData` + `DebugMap` into columnar rows.
//!
//! The executor thread produces these (cheap, reference-reading) and hands them
//! to the worker thread, which owns the Arrow builders. This keeps the hot path
//! free of any full `WorldData` clone.

use dies_core::{
    BallData, DebugColor, DebugMap, DebugShape, DebugValue, Handicap, PlayerData, SideAssignment,
    SysStatus, TeamColor, WorldData,
};

/// One world frame, projected into the per-frame tables.
#[derive(Debug, Clone)]
pub struct FrameRecord {
    pub frame_id: u64,
    // frames (spine)
    pub t_received: f64,
    pub t_capture: f64,
    pub dt: f64,
    pub game_state: String,
    pub operating_team: String,
    pub side_assignment: String,
    pub ball_on_blue_side: Option<f64>,
    pub ball_on_yellow_side: Option<f64>,
    // ball
    pub ball: Option<BallRow>,
    // players (long)
    pub players: Vec<PlayerRow>,
    // debug, classified into the three debug tables
    pub debug_values: Vec<DebugValueRow>,
    pub debug_shapes: Vec<DebugShapeRow>,
    pub debug_tree: Vec<DebugTreeRow>,
}

#[derive(Debug, Clone)]
pub struct BallRow {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
    pub detected: bool,
}

#[derive(Debug, Clone)]
pub struct PlayerRow {
    pub team: &'static str,
    pub player_id: u32,
    pub x: f64,
    pub y: f64,
    pub vx: f64,
    pub vy: f64,
    pub yaw: f64,
    pub raw_yaw: f64,
    pub angular_speed: f64,
    pub position_noise: f64,
    pub primary_status: Option<String>,
    pub kicker_cap_voltage: Option<f32>,
    pub kicker_temp: Option<f32>,
    pub pack_voltage_0: Option<f32>,
    pub pack_voltage_1: Option<f32>,
    pub breakbeam_ball_detected: bool,
    pub imu_status: Option<String>,
    pub kicker_status: Option<String>,
    pub handicaps: String,
}

#[derive(Debug, Clone)]
pub struct DebugValueRow {
    pub key: String,
    pub value: Option<f64>,
    pub value_str: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct DebugShapeRow {
    pub key: String,
    pub shape_type: &'static str,
    pub cx: Option<f64>,
    pub cy: Option<f64>,
    pub radius: Option<f64>,
    pub x1: Option<f64>,
    pub y1: Option<f64>,
    pub x2: Option<f64>,
    pub y2: Option<f64>,
    pub color: Option<String>,
    pub fill: Option<String>,
    pub stroke: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DebugTreeRow {
    pub key: String,
    pub name: String,
    pub node_id: String,
    pub children_ids: Vec<String>,
    pub is_active: bool,
    pub node_type: String,
    pub internal_state: Option<String>,
    pub additional_info: Option<String>,
}

/// A discrete, sparse event (referee/control actions, autoref kicks, ...).
#[derive(Debug, Clone)]
pub struct EventRecord {
    pub frame_id: u64,
    pub t: f64,
    pub event_type: String,
    pub payload_json: String,
}

/// A captured log line (from the `log::Log` impl).
#[derive(Debug, Clone)]
pub struct LogLineRecord {
    pub t: f64,
    pub level: String,
    pub target: String,
    pub source: String,
    pub message: String,
}

/// A user-placed point of interest, dropped live (double-space in the UI) or
/// noted by hand from the on-screen frame counter.
#[derive(Debug, Clone)]
pub struct MarkerRecord {
    pub frame_id: u64,
    pub t: f64,
    pub label: Option<String>,
}

/// Raw received wire bytes (gated vision/GC stream).
#[derive(Debug, Clone)]
pub struct RawRecord {
    pub frame_id: u64,
    pub t: f64,
    pub kind: String,
    pub bytes: Vec<u8>,
}

impl FrameRecord {
    /// Project a world frame + debug snapshot into columnar rows. Reads
    /// references only — no `WorldData` clone.
    pub fn from_world(frame_id: u64, world: &WorldData, debug: &DebugMap) -> Self {
        let mut players = Vec::with_capacity(world.blue_team.len() + world.yellow_team.len());
        for p in &world.blue_team {
            players.push(player_row(TeamColor::Blue, p));
        }
        for p in &world.yellow_team {
            players.push(player_row(TeamColor::Yellow, p));
        }

        let (mut debug_values, mut debug_shapes, mut debug_tree) =
            (Vec::new(), Vec::new(), Vec::new());
        for (key, value) in debug.iter() {
            classify_debug(key, value, &mut debug_values, &mut debug_shapes, &mut debug_tree);
        }

        Self {
            frame_id,
            t_received: world.t_received,
            t_capture: world.t_capture,
            dt: world.dt,
            game_state: world.game_state.game_state.to_string(),
            operating_team: team_color_str(world.game_state.operating_team).to_string(),
            side_assignment: side_assignment_str(world.side_assignment).to_string(),
            ball_on_blue_side: world.ball_on_blue_side.map(|d| d.as_secs_f64()),
            ball_on_yellow_side: world.ball_on_yellow_side.map(|d| d.as_secs_f64()),
            ball: world.ball.as_ref().map(ball_row),
            players,
            debug_values,
            debug_shapes,
            debug_tree,
        }
    }
}

fn ball_row(b: &BallData) -> BallRow {
    BallRow {
        x: b.position.x,
        y: b.position.y,
        z: b.position.z,
        vx: b.velocity.x,
        vy: b.velocity.y,
        vz: b.velocity.z,
        detected: b.detected,
    }
}

fn player_row(team: TeamColor, p: &PlayerData) -> PlayerRow {
    let (pv0, pv1) = match p.pack_voltages {
        Some([a, b]) => (Some(a), Some(b)),
        None => (None, None),
    };
    let handicaps = {
        let mut hs: Vec<&str> = p.handicaps.iter().map(handicap_str).collect();
        hs.sort_unstable();
        hs.join(",")
    };
    PlayerRow {
        team: team_color_str(team),
        player_id: p.id.as_u32(),
        x: p.position.x,
        y: p.position.y,
        vx: p.velocity.x,
        vy: p.velocity.y,
        yaw: p.yaw.radians(),
        raw_yaw: p.raw_yaw.radians(),
        angular_speed: p.angular_speed,
        position_noise: p.position_noise,
        primary_status: p.primary_status.map(|s| sys_status_str(s).to_string()),
        kicker_cap_voltage: p.kicker_cap_voltage,
        kicker_temp: p.kicker_temp,
        pack_voltage_0: pv0,
        pack_voltage_1: pv1,
        breakbeam_ball_detected: p.breakbeam_ball_detected,
        imu_status: p.imu_status.map(|s| sys_status_str(s).to_string()),
        kicker_status: p.kicker_status.map(|s| sys_status_str(s).to_string()),
        handicaps,
    }
}

/// Sort a `DebugMap` entry into the appropriate debug table.
fn classify_debug(
    key: &str,
    value: &DebugValue,
    values: &mut Vec<DebugValueRow>,
    shapes: &mut Vec<DebugShapeRow>,
    tree: &mut Vec<DebugTreeRow>,
) {
    match value {
        DebugValue::Number(n) => values.push(DebugValueRow {
            key: key.to_string(),
            value: Some(*n),
            value_str: None,
        }),
        DebugValue::String(s) => values.push(DebugValueRow {
            key: key.to_string(),
            value: None,
            value_str: Some(s.clone()),
        }),
        DebugValue::Shape(shape) => match shape {
            DebugShape::Cross { center, color } => shapes.push(DebugShapeRow {
                key: key.to_string(),
                shape_type: "cross",
                cx: Some(center.x),
                cy: Some(center.y),
                color: Some(debug_color_str(*color).to_string()),
                ..Default::default()
            }),
            DebugShape::Circle {
                center,
                radius,
                fill,
                stroke,
            } => shapes.push(DebugShapeRow {
                key: key.to_string(),
                shape_type: "circle",
                cx: Some(center.x),
                cy: Some(center.y),
                radius: Some(*radius),
                fill: fill.map(|c| debug_color_str(c).to_string()),
                stroke: stroke.map(|c| debug_color_str(c).to_string()),
                ..Default::default()
            }),
            DebugShape::Line { start, end, color } => shapes.push(DebugShapeRow {
                key: key.to_string(),
                shape_type: "line",
                x1: Some(start.x),
                y1: Some(start.y),
                x2: Some(end.x),
                y2: Some(end.y),
                color: Some(debug_color_str(*color).to_string()),
                ..Default::default()
            }),
            DebugShape::TreeNode {
                name,
                id,
                children_ids,
                is_active,
                node_type,
                internal_state,
                additional_info,
            } => tree.push(DebugTreeRow {
                key: key.to_string(),
                name: name.clone(),
                node_id: id.clone(),
                children_ids: children_ids.clone(),
                is_active: *is_active,
                node_type: node_type.clone(),
                internal_state: internal_state.clone(),
                additional_info: additional_info.clone(),
            }),
        },
    }
}

// --- Stable string encodings for enums (reversed in `replay::reconstruct`). ---

pub fn team_color_str(c: TeamColor) -> &'static str {
    match c {
        TeamColor::Blue => "blue",
        TeamColor::Yellow => "yellow",
    }
}

pub fn side_assignment_str(s: SideAssignment) -> &'static str {
    match s {
        SideAssignment::BlueOnPositive => "blue_on_positive",
        SideAssignment::YellowOnPositive => "yellow_on_positive",
    }
}

pub fn debug_color_str(c: DebugColor) -> &'static str {
    match c {
        DebugColor::Red => "red",
        DebugColor::Green => "green",
        DebugColor::Orange => "orange",
        DebugColor::Purple => "purple",
        DebugColor::Blue => "blue",
        DebugColor::Gray => "gray",
    }
}

pub fn handicap_str(h: &Handicap) -> &'static str {
    match h {
        Handicap::NoKicker => "no_kicker",
        Handicap::NoDribbler => "no_dribbler",
        Handicap::NoBreakbeam => "no_breakbeam",
    }
}

pub fn sys_status_str(s: SysStatus) -> &'static str {
    match s {
        SysStatus::Emergency => "emergency",
        SysStatus::Ok => "ok",
        SysStatus::Ready => "ready",
        SysStatus::Stop => "stop",
        SysStatus::Starting => "starting",
        SysStatus::Overtemp => "overtemp",
        SysStatus::NoReply => "no_reply",
        SysStatus::Armed => "armed",
        SysStatus::Disarmed => "disarmed",
        SysStatus::Safe => "safe",
        SysStatus::NotInstalled => "not_installed",
        SysStatus::Standby => "standby",
        SysStatus::Cooldown => "cooldown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dies_core::{mock_world_data, DebugValue, Vector2};

    #[test]
    fn projects_mock_world() {
        let world = mock_world_data();
        let mut debug = DebugMap::new();
        debug.insert("team_blue.p0.cost".into(), DebugValue::Number(1.5));
        debug.insert("note".into(), DebugValue::String("hello".into()));
        debug.insert(
            "team_blue.p0.target".into(),
            DebugValue::Shape(DebugShape::Cross {
                center: Vector2::new(10.0, 20.0),
                color: DebugColor::Green,
            }),
        );

        let rec = FrameRecord::from_world(7, &world, &debug);
        assert_eq!(rec.frame_id, 7);
        // mock has one blue + one yellow player
        assert_eq!(rec.players.len(), 2);
        assert_eq!(rec.players[0].team, "blue");
        assert_eq!(rec.players[1].team, "yellow");
        assert_eq!(rec.debug_values.len(), 2); // number + string
        assert_eq!(rec.debug_shapes.len(), 1);
        assert_eq!(rec.game_state, "Run");
    }
}
