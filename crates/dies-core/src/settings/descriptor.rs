//! Descriptor for settings types

use serde::{Deserialize, Serialize};
use ts_rs::TS;

#[derive(Serialize, Deserialize, TS)]
#[ts(export)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TypeDesc {
    Struct(StructDesc),
    Enum(EnumDesc),
    Value(ValueDesc),
}

#[derive(Serialize, Deserialize, TS)]
pub struct StructDesc {
    /// The name of the object in Rust
    pub type_name: String,
    /// The label of the object in the UI
    pub label: String,
    /// Brief description of the object
    pub description: Option<String>,
    /// The fields of the object
    pub fields: Vec<FieldDesc>,
}

#[derive(Serialize, Deserialize, TS)]
pub struct FieldDesc {
    /// The name of the field in Rust
    pub name: String,
    /// The label of the field in the UI
    pub label: String,
    /// Brief description of the field
    pub description: Option<String>,
    /// The type of the field
    pub field_type: TypeDesc,
}

#[derive(Serialize, Deserialize, TS)]
pub struct EnumDesc {
    /// The name of the enum in Rust
    pub type_name: String,
    /// The label of the enum in the UI
    pub label: String,
    /// Brief description of the enum
    pub description: Option<String>,
    /// The variants of the enum
    pub variants: Vec<VariantDesc>,
}

#[derive(Serialize, Deserialize, TS)]
pub struct VariantDesc {
    /// The name of the variant in Rust
    pub type_name: String,
    /// The label of the variant in the UI
    pub label: String,
    /// Brief description of the variant
    pub description: Option<String>,
    /// The type of the variant
    pub variant_type: TypeDesc,
}

#[derive(Serialize, Deserialize, TS)]
pub enum ValueDesc {
    Bool,
    Int {
        /// The minimum value of the integer
        min: Option<i64>,
        /// The maximum value of the integer
        max: Option<i64>,
        /// The step size of the integer. If provided, the integer will be displayed as
        /// a slider.
        step: Option<i64>,
    },
    Float {
        /// The minimum value of the float
        min: Option<f64>,
        /// The maximum value of the float
        max: Option<f64>,
        /// The step size of the float. If provided, the float will be displayed as
        /// a slider.
        step: Option<f64>,
    },
    String {
        /// The regex pattern of the string. If provided, only strings matching the
        /// pattern will be valid for saving.
        pattern: Option<String>,
    },
}
