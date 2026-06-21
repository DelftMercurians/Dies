//! Flatten `ExecutorSettings` into dotted-key scalar rows, and diff two
//! snapshots so only changed keys are logged to the `settings_changes` stream.
//!
//! The baseline (full flattened settings) is emitted once at frame 0; thereafter
//! each `UpdateSettings` emits only the keys whose value changed (or were
//! removed — relevant for the dynamic `handicaps` map). Python reconstructs the
//! value at any frame with a forward-fill (`merge_asof`).

use std::collections::BTreeMap;

use dies_core::ExecutorSettings;
use serde_json::Value;

/// A flattened scalar value. Maps onto the `value_num` / `value_str` columns.
#[derive(Debug, Clone, PartialEq)]
pub enum Scalar {
    Num(f64),
    Str(String),
    Bool(bool),
}

impl Scalar {
    /// Split into `(value_num, value_str)` for the columnar row. Numbers fill
    /// `value_num`; strings and bools fill `value_str`.
    pub fn to_columns(&self) -> (Option<f64>, Option<String>) {
        match self {
            Scalar::Num(n) => (Some(*n), None),
            Scalar::Str(s) => (None, Some(s.clone())),
            Scalar::Bool(b) => (None, Some(b.to_string())),
        }
    }
}

/// A flattened settings snapshot: dotted key -> scalar.
pub type FlatSettings = BTreeMap<String, Scalar>;

/// A single change to emit. `None` means the key was removed (logged as a row
/// with both value columns null).
pub type Change = (String, Option<Scalar>);

/// Flatten the entire `ExecutorSettings` into dotted-key scalars.
pub fn flatten(settings: &ExecutorSettings) -> FlatSettings {
    let mut out = FlatSettings::new();
    // `ExecutorSettings` always serializes cleanly; treat a failure as empty.
    if let Ok(value) = serde_json::to_value(settings) {
        walk("", &value, &mut out);
    }
    out
}

fn walk(prefix: &str, value: &Value, out: &mut FlatSettings) {
    match value {
        Value::Null => {}
        Value::Bool(b) => {
            out.insert(prefix.to_string(), Scalar::Bool(*b));
        }
        Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                out.insert(prefix.to_string(), Scalar::Num(f));
            }
        }
        Value::String(s) => {
            out.insert(prefix.to_string(), Scalar::Str(s.clone()));
        }
        Value::Array(arr) => {
            for (i, v) in arr.iter().enumerate() {
                walk(&join(prefix, &i.to_string()), v, out);
            }
        }
        Value::Object(map) => {
            for (k, v) in map {
                walk(&join(prefix, k), v, out);
            }
        }
    }
}

fn join(prefix: &str, key: &str) -> String {
    if prefix.is_empty() {
        key.to_string()
    } else {
        format!("{prefix}.{key}")
    }
}

/// Emit one change per key that was added, changed, or removed between `prev`
/// and `next`. Sorted by key (BTreeMap iteration order) for deterministic logs.
pub fn diff(prev: &FlatSettings, next: &FlatSettings) -> Vec<Change> {
    let mut changes = Vec::new();
    for (key, val) in next {
        if prev.get(key) != Some(val) {
            changes.push((key.clone(), Some(val.clone())));
        }
    }
    for key in prev.keys() {
        if !next.contains_key(key) {
            changes.push((key.clone(), None));
        }
    }
    changes.sort_by(|a, b| a.0.cmp(&b.0));
    changes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flatten_produces_dotted_keys() {
        let s = ExecutorSettings::default();
        let flat = flatten(&s);
        // a known nested numeric field
        assert!(matches!(
            flat.get("controller_settings.max_velocity"),
            Some(Scalar::Num(v)) if (*v - 3000.0).abs() < 1e-9
        ));
        // a known bool field
        assert!(matches!(
            flat.get("goal_area_avoidance"),
            Some(Scalar::Bool(true))
        ));
    }

    #[test]
    fn diff_reports_only_changes() {
        let mut a = ExecutorSettings::default();
        let prev = flatten(&a);
        a.controller_settings.max_velocity = 1234.0;
        let next = flatten(&a);

        let changes = diff(&prev, &next);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].0, "controller_settings.max_velocity");
        assert_eq!(changes[0].1, Some(Scalar::Num(1234.0)));
    }

    #[test]
    fn diff_reports_removals_as_none() {
        let mut prev = FlatSettings::new();
        prev.insert(
            "blue_team_settings.handicaps.3".into(),
            Scalar::Str("no_kicker".into()),
        );
        let next = FlatSettings::new();
        let changes = diff(&prev, &next);
        assert_eq!(
            changes,
            vec![("blue_team_settings.handicaps.3".to_string(), None)]
        );
    }
}
