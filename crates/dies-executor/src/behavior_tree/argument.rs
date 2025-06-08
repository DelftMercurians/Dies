use super::bt_callback::BtCallback;
use crate::behavior_tree::RobotSituation;
use anyhow::Result;
use dies_core::Angle;
use rhai::{Engine, FnPtr, NativeCallContext, Variant};

#[derive(Clone, Debug)]
pub enum Argument<T: Clone + Variant + Send + Sync + 'static> {
    Static(T),
    Callback(BtCallback<T>),
}

impl<T: Clone + Variant + Send + Sync + 'static> Argument<T> {
    pub fn from_rhai(
        val: rhai::Dynamic,
        context: &NativeCallContext,
    ) -> Result<Self, Box<rhai::EvalAltResult>> {
        let name = val.type_name().to_string();
        if let Some(fn_ptr) = val.clone().try_cast::<FnPtr>() {
            Ok(Argument::Callback(BtCallback::new_rhai(context, fn_ptr)))
        } else if let Some(static_val) = val.try_cast::<T>() {
            Ok(Argument::Static(static_val))
        } else {
            Err(Box::new(rhai::EvalAltResult::ErrorMismatchDataType(
                format!("Expected {} or callback", std::any::type_name::<T>()),
                name,
                rhai::Position::NONE,
            )))
        }
    }

    pub fn resolve(&self, situation: &RobotSituation, engine: &Engine) -> Result<T> {
        match self {
            Argument::Static(val) => Ok(val.clone()),
            Argument::Callback(cb) => cb.call(situation, engine),
        }
    }
}

impl Argument<Option<Angle>> {
    pub fn from_rhai_angle(
        val: rhai::Dynamic,
        context: &NativeCallContext,
    ) -> Result<Self, Box<rhai::EvalAltResult>> {
        if let Some(fn_ptr) = val.clone().try_cast::<FnPtr>() {
            Ok(Argument::Callback(BtCallback::new_rhai(context, fn_ptr)))
        } else if let Some(static_val) = val.clone().try_cast::<f64>() {
            Ok(Argument::Static(Some(Angle::from_radians(static_val))))
        } else if val.is::<()>( /* check for nil */ ) {
            Ok(Argument::Static(None))
        } else {
            Err(Box::new(rhai::EvalAltResult::ErrorMismatchDataType(
                "Expected f64, nil or callback".to_string(),
                val.type_name().into(),
                rhai::Position::NONE,
            )))
        }
    }
}
