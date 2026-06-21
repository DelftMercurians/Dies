//! Generic, UI-discoverable strategy parameters.
//!
//! A strategy declares a set of typed parameters ([`ParamSpec`]); the web UI
//! renders a control per spec and pushes value changes ([`ParamValue`]) to the
//! running strategy at runtime. The first consumer is a `defense_only` toggle.
//!
//! [`ParamValue`] uses serde's default **externally-tagged** representation so it
//! round-trips over both bincode (the strategy IPC) and JSON (the web UI).
//! Adjacently-tagged enums (`#[serde(tag, content)]`) are NOT bincode-safe, so we
//! deliberately avoid that here even though it is the codebase convention for
//! UI-only command enums.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use typeshare::typeshare;

use crate::TeamColor;

/// A typed parameter value. Externally tagged → bincode- and JSON-safe.
///
/// Deliberately NOT `#[typeshare]`: typeshare insists algebraic enums use
/// adjacent/internal tagging, which bincode cannot deserialize. We keep the
/// bincode-safe external tagging and hand-write the TypeScript type in
/// `crates/dies-webui/build.rs` (the same approach used for `Vector2`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ParamValue {
    Bool(bool),
    Float(f64),
    /// `i32` rather than `i64` because typeshare maps `i64` to an error (JS numbers
    /// can't hold the full range); `i32` is ample for strategy parameters.
    Int(i32),
    Text(String),
}

impl ParamValue {
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ParamValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match self {
            ParamValue::Float(f) => Some(*f),
            ParamValue::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i32> {
        match self {
            ParamValue::Int(i) => Some(*i),
            ParamValue::Float(f) => Some(*f as i32),
            _ => None,
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            ParamValue::Text(s) => Some(s),
            _ => None,
        }
    }
}

/// Which control the UI should render for a parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[typeshare]
pub enum ParamKind {
    Bool,
    Float,
    Int,
    Text,
}

/// A strategy's declaration of one parameter: how to label/render it, and its
/// default value (used when no override has been pushed yet).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[typeshare]
pub struct ParamSpec {
    pub key: String,
    pub label: String,
    pub kind: ParamKind,
    pub default: ParamValue,
}

impl ParamSpec {
    pub fn bool(key: &str, label: &str, default: bool) -> Self {
        Self {
            key: key.into(),
            label: label.into(),
            kind: ParamKind::Bool,
            default: ParamValue::Bool(default),
        }
    }

    pub fn float(key: &str, label: &str, default: f64) -> Self {
        Self {
            key: key.into(),
            label: label.into(),
            kind: ParamKind::Float,
            default: ParamValue::Float(default),
        }
    }

    pub fn int(key: &str, label: &str, default: i32) -> Self {
        Self {
            key: key.into(),
            label: label.into(),
            kind: ParamKind::Int,
            default: ParamValue::Int(default),
        }
    }

    pub fn text(key: &str, label: &str, default: &str) -> Self {
        Self {
            key: key.into(),
            label: label.into(),
            kind: ParamKind::Text,
            default: ParamValue::Text(default.into()),
        }
    }
}

/// Current parameter values keyed by [`ParamSpec::key`]. Rust-side alias only —
/// typeshared structs inline `HashMap<String, ParamValue>` so the alias never
/// needs to cross to TypeScript.
pub type StrategyParams = HashMap<String, ParamValue>;

/// One team's strategy parameters (declared specs + current values), surfaced to
/// the UI via `ExecutorInfo`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[typeshare]
pub struct TeamStrategyParams {
    pub team: TeamColor,
    pub specs: Vec<ParamSpec>,
    pub values: HashMap<String, ParamValue>,
}
