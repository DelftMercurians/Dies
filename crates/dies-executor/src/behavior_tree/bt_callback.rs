#![allow(deprecated)]

use super::bt_core::RobotSituation;
use anyhow::Result;
use rhai::{Engine, FnPtr, FuncArgs, NativeCallContext, NativeCallContextStore, Variant};

#[derive(Clone)]
pub enum BtCallback<TRet> {
    Native(fn(&RobotSituation) -> TRet),
    Rhai(RhaiFunction),
}

impl<TRet> BtCallback<TRet>
where
    TRet: Clone + Variant,
{
    pub fn new_native(f: fn(&RobotSituation) -> TRet) -> Self {
        Self::Native(f)
    }

    pub fn new_rhai(call_ctx: &NativeCallContext, fn_ptr: FnPtr) -> Self {
        Self::Rhai(RhaiFunction::new(call_ctx, fn_ptr))
    }

    pub fn call<'a>(&self, situation: &RobotSituation) -> Result<TRet> {
        match self {
            BtCallback::Native(f) => Ok(f(situation)),
            BtCallback::Rhai(f) => {
                let result = f.call::<(RobotSituation,), TRet>((situation.clone(),))?;
                Ok(result)
            }
        }
    }
}

#[derive(Clone)]
pub struct RhaiFunction {
    call_ctx: NativeCallContextStore,
    fn_ptr: FnPtr,
}

impl RhaiFunction {
    pub fn new(call_ctx: &NativeCallContext, fn_ptr: FnPtr) -> Self {
        Self {
            call_ctx: call_ctx.store_data(),
            fn_ptr,
        }
    }

    pub fn call<TArgs, TRet>(&self, args: TArgs) -> Result<TRet>
    where
        TArgs: FuncArgs,
        TRet: Clone + Variant,
    {
        let engine = Engine::new_raw();
        self.fn_ptr
            .call_within_context::<TRet>(&self.call_ctx.create_context(&engine), args)
            .map_err(anyhow::Error::new)
    }
}
