//! `Argument<T>` — a tree parameter that is either a constant or a closure
//! evaluated against the live `RobotSituation` each tick.

use std::sync::Arc;

use super::situation::{RobotSituation, ShootTarget};
use super::BtCallback;

#[derive(Clone)]
pub enum Argument<T: Clone + Send + Sync + 'static> {
    Static(T),
    Callback(Arc<dyn BtCallback<T>>),
}

impl<T: Clone + Send + Sync + 'static> Argument<T> {
    pub fn static_arg(val: T) -> Self {
        Self::Static(val)
    }

    pub fn callback(cb: impl BtCallback<T>) -> Self {
        Self::Callback(Arc::new(cb))
    }

    pub fn resolve(&self, situation: &RobotSituation) -> T {
        match self {
            Argument::Static(val) => val.clone(),
            Argument::Callback(cb) => cb(situation),
        }
    }

    pub fn map<U: Clone + Send + Sync + 'static>(
        self,
        f: impl Fn(T) -> U + Send + Sync + 'static,
    ) -> Argument<U> {
        match self {
            Argument::Static(val) => Argument::Static(f(val)),
            Argument::Callback(cb) => Argument::Callback(Arc::new(move |s| f(cb(s)))),
        }
    }
}

impl<R: Clone + Send + Sync + 'static, T: BtCallback<R>> From<T> for Argument<R> {
    fn from(cb: T) -> Self {
        Argument::Callback(Arc::new(cb))
    }
}

macro_rules! impl_into_argument_for_primitive {
    ($($ty:ty),*) => {
        $(
            impl From<$ty> for Argument<$ty> {
                fn from(val: $ty) -> Self {
                    Argument::Static(val)
                }
            }
        )*
    };
}

impl_into_argument_for_primitive!(
    bool, u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64, String
);

impl<'a> From<&'a str> for Argument<String> {
    fn from(val: &'a str) -> Self {
        Argument::Static(val.to_string())
    }
}

impl From<ShootTarget> for Argument<ShootTarget> {
    fn from(val: ShootTarget) -> Self {
        Argument::Static(val)
    }
}
