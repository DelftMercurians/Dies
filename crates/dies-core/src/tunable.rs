//! Runtime-tunable `f64` knobs and their UI-discoverable metadata.
//!
//! [`Tunable`] is the process-global value cell behind each knob declared with
//! the `tunables!` macro (`dies-tunables-macro`): a lock-free `f64` readable on
//! the per-frame path and writable from a settings update. [`TunableSpec`] is the
//! code-generated description the web UI renders a control from. Values live in
//! `ExecutorSettings::skill_tunables` (persisted, baseline/revert-aware); the
//! specs ride `ExecutorInfo::skill_tunable_specs`.

use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use typeshare::typeshare;

/// A single runtime-tunable `f64`, stored as its IEEE-754 bit pattern in an atomic
/// so reads on the per-frame path are lock-free and the value can be updated live
/// without breaking determinism (it only changes on an explicit settings command).
pub struct Tunable {
    cell: AtomicU64,
    default: f64,
}

impl Tunable {
    /// Construct with the compile-time default — the single source of truth a
    /// revert restores to.
    pub const fn new(default: f64) -> Self {
        Self {
            cell: AtomicU64::new(default.to_bits()),
            default,
        }
    }

    /// Current effective value.
    pub fn get(&self) -> f64 {
        f64::from_bits(self.cell.load(Ordering::Relaxed))
    }

    /// Override the value (settings update).
    pub fn set(&self, value: f64) {
        self.cell.store(value.to_bits(), Ordering::Relaxed);
    }

    /// Snap back to the compile-time default (revert / cleared override).
    pub fn reset(&self) {
        self.set(self.default);
    }

    /// The compile-time default.
    pub fn default(&self) -> f64 {
        self.default
    }
}

/// The defining module's last path segment, used to namespace knob keys so the
/// same `NAME` may appear in several skills (e.g. `DRIBBLER_SPEED`). Called by the
/// macro-generated code with `module_path!()`.
pub fn tunable_module_prefix(module_path: &str) -> String {
    module_path
        .rsplit("::")
        .next()
        .unwrap_or(module_path)
        .to_string()
}

/// Code-generated description of one tunable knob, surfaced to the web UI to
/// render a control. Values themselves are stored separately in
/// `ExecutorSettings::skill_tunables` keyed by [`TunableSpec::key`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[typeshare]
pub struct TunableSpec {
    /// Namespaced unique key, `"<module>.<NAME>"` (the settings-map key).
    pub key: String,
    /// Human label derived from the knob name (Title Case).
    pub label: String,
    /// Help text from the knob's doc comment, if any.
    pub help: Option<String>,
    /// Compile-time default (shown when no override is set; revert target).
    pub default: f64,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub step: Option<f64>,
    /// Display unit suffix (e.g. `"mm"`, `"mm/s"`, `"s"`, `"rad"`).
    pub unit: Option<String>,
    /// Grouping label for the UI; defaults to the module/skill name.
    pub section: Option<String>,
}

impl TunableSpec {
    /// Build from the namespaced parts. The remaining metadata is attached via the
    /// builder methods below (all called from macro-generated code).
    pub fn new(prefix: &str, name: &str, default: f64) -> Self {
        Self {
            key: format!("{prefix}.{name}"),
            label: title_case(name),
            help: None,
            default,
            min: None,
            max: None,
            step: None,
            unit: None,
            section: None,
        }
    }

    pub fn help(mut self, help: &str) -> Self {
        self.help = Some(help.to_string());
        self
    }

    pub fn section(mut self, section: &str) -> Self {
        self.section = Some(section.to_string());
        self
    }

    pub fn unit(mut self, unit: &str) -> Self {
        self.unit = Some(unit.to_string());
        self
    }

    pub fn min(mut self, min: f64) -> Self {
        self.min = Some(min);
        self
    }

    pub fn max(mut self, max: f64) -> Self {
        self.max = Some(max);
        self
    }

    pub fn step(mut self, step: f64) -> Self {
        self.step = Some(step);
        self
    }
}

/// `COMMIT_DISTANCE` → `Commit Distance`.
fn title_case(name: &str) -> String {
    name.split('_')
        .filter(|w| !w.is_empty())
        .map(|w| {
            let mut chars = w.chars();
            match chars.next() {
                Some(first) => {
                    first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase()
                }
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
