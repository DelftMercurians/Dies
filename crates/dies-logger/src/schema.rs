//! Arrow schemas for every table in the columnar log format.
//!
//! A log is a directory of per-table Arrow IPC streams (`<table>.arrow`),
//! compacted to Parquet on close. Every per-frame table is keyed on a monotonic
//! `frame_id` (the join key across tables). See the module docs in `lib.rs` for
//! the overall design.

use std::sync::OnceLock;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};

/// The set of tables in a log directory, in a stable order.
pub const TABLES: &[&str] = &[
    "frames",
    "ball",
    "players",
    "player_feedback",
    "debug_values",
    "debug_shapes",
    "debug_tree",
    "settings_changes",
    "events",
    "markers",
    "logs",
    "vision",
];

fn nullable(name: &str, dt: DataType) -> Field {
    Field::new(name, dt, true)
}

fn required(name: &str, dt: DataType) -> Field {
    Field::new(name, dt, false)
}

macro_rules! cached_schema {
    ($func:ident, $($field:expr),+ $(,)?) => {
        pub fn $func() -> SchemaRef {
            static SCHEMA: OnceLock<SchemaRef> = OnceLock::new();
            SCHEMA
                .get_or_init(|| std::sync::Arc::new(Schema::new(vec![$($field),+])))
                .clone()
        }
    };
}

cached_schema!(
    frames,
    required("frame_id", DataType::UInt64),
    required("t_received", DataType::Float64),
    required("t_capture", DataType::Float64),
    required("dt", DataType::Float64),
    required("game_state", DataType::Utf8),
    required("operating_team", DataType::Utf8),
    required("side_assignment", DataType::Utf8),
    nullable("ball_on_blue_side", DataType::Float64),
    nullable("ball_on_yellow_side", DataType::Float64),
);

cached_schema!(
    ball,
    required("frame_id", DataType::UInt64),
    required("x", DataType::Float64),
    required("y", DataType::Float64),
    required("z", DataType::Float64),
    required("vx", DataType::Float64),
    required("vy", DataType::Float64),
    required("vz", DataType::Float64),
    required("detected", DataType::Boolean),
);

cached_schema!(
    players,
    required("frame_id", DataType::UInt64),
    required("team", DataType::Utf8),
    required("player_id", DataType::UInt32),
    required("x", DataType::Float64),
    required("y", DataType::Float64),
    required("vx", DataType::Float64),
    required("vy", DataType::Float64),
    required("yaw", DataType::Float64),
    required("raw_yaw", DataType::Float64),
    required("raw_x", DataType::Float64),
    required("raw_y", DataType::Float64),
    required("angular_speed", DataType::Float64),
    required("position_noise", DataType::Float64),
    nullable("primary_status", DataType::Utf8),
    nullable("kicker_cap_voltage", DataType::Float32),
    nullable("kicker_temp", DataType::Float32),
    nullable("pack_voltage_0", DataType::Float32),
    nullable("pack_voltage_1", DataType::Float32),
    required("breakbeam_ball_detected", DataType::Boolean),
    required("has_ball", DataType::Boolean),
    nullable("reflex_kick_state", DataType::Utf8),
    nullable("imu_status", DataType::Utf8),
    nullable("kicker_status", DataType::Utf8),
    required("handicaps", DataType::Utf8),
    // "radio_lost" / "card_removed" when the robot is sidelined; null otherwise.
    nullable("sideline", DataType::Utf8),
);

// Full basestation robot feedback, one row per (frame, player). The entire
// `PlayerFeedbackMsg` is captured verbatim as JSON so every field — including
// motors, currents, loop times, reflex-kick, ToF, firmware — is preserved and
// the schema stays forward-compatible as firmware adds fields. `team`/`player_id`
// are lifted out as columns for cheap filtering/joins without parsing the JSON.
cached_schema!(
    player_feedback,
    required("frame_id", DataType::UInt64),
    required("team", DataType::Utf8),
    required("player_id", DataType::UInt32),
    required("feedback_json", DataType::Utf8),
);

// `value` and `value_str` are both nullable: a `Number` debug entry fills
// `value`, a `String` entry fills `value_str`.
cached_schema!(
    debug_values,
    required("frame_id", DataType::UInt64),
    required("key", DataType::Utf8),
    nullable("value", DataType::Float64),
    nullable("value_str", DataType::Utf8),
);

cached_schema!(
    debug_shapes,
    required("frame_id", DataType::UInt64),
    required("key", DataType::Utf8),
    required("shape_type", DataType::Utf8),
    nullable("cx", DataType::Float64),
    nullable("cy", DataType::Float64),
    nullable("radius", DataType::Float64),
    nullable("x1", DataType::Float64),
    nullable("y1", DataType::Float64),
    nullable("x2", DataType::Float64),
    nullable("y2", DataType::Float64),
    nullable("color", DataType::Utf8),
    nullable("fill", DataType::Utf8),
    nullable("stroke", DataType::Utf8),
);

cached_schema!(
    debug_tree,
    required("frame_id", DataType::UInt64),
    required("key", DataType::Utf8),
    required("name", DataType::Utf8),
    required("node_id", DataType::Utf8),
    required(
        "children_ids",
        DataType::List(std::sync::Arc::new(Field::new(
            "item",
            DataType::Utf8,
            true
        )))
    ),
    required("is_active", DataType::Boolean),
    required("node_type", DataType::Utf8),
    nullable("internal_state", DataType::Utf8),
    nullable("additional_info", DataType::Utf8),
);

cached_schema!(
    settings_changes,
    required("frame_id", DataType::UInt64),
    required("t", DataType::Float64),
    required("key", DataType::Utf8),
    nullable("value_num", DataType::Float64),
    nullable("value_str", DataType::Utf8),
);

cached_schema!(
    events,
    required("frame_id", DataType::UInt64),
    required("t", DataType::Float64),
    required("event_type", DataType::Utf8),
    required("payload_json", DataType::Utf8),
);

// User-placed points of interest. `label` is optional (double-space with no
// text still drops a marker at the current frame).
cached_schema!(
    markers,
    required("frame_id", DataType::UInt64),
    required("t", DataType::Float64),
    nullable("label", DataType::Utf8),
);

cached_schema!(
    logs,
    required("t", DataType::Float64),
    required("level", DataType::Utf8),
    required("target", DataType::Utf8),
    required("source", DataType::Utf8),
    required("message", DataType::Utf8),
);

cached_schema!(
    vision,
    required("frame_id", DataType::UInt64),
    required("t", DataType::Float64),
    required("kind", DataType::Utf8),
    required("bytes", DataType::Binary),
);

/// Look up a table's schema by name. Returns `None` for unknown tables.
pub fn schema_for(table: &str) -> Option<SchemaRef> {
    Some(match table {
        "frames" => frames(),
        "ball" => ball(),
        "players" => players(),
        "player_feedback" => player_feedback(),
        "debug_values" => debug_values(),
        "debug_shapes" => debug_shapes(),
        "debug_tree" => debug_tree(),
        "settings_changes" => settings_changes(),
        "events" => events(),
        "markers" => markers(),
        "logs" => logs(),
        "vision" => vision(),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_table_has_a_schema() {
        for &t in TABLES {
            assert!(schema_for(t).is_some(), "missing schema for {t}");
        }
    }

    #[test]
    fn frame_id_is_first_and_non_null_on_per_frame_tables() {
        for &t in &["frames", "ball", "players", "debug_values", "vision"] {
            let s = schema_for(t).unwrap();
            let f = s.field(0);
            assert_eq!(f.name(), "frame_id", "{t} first column");
            assert!(!f.is_nullable(), "{t}.frame_id should be non-null");
        }
    }
}
