use crate::behavior_tree::{BtCallback, RobotSituation};
use std::sync::Arc;

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
            Argument::Callback(cb) => {
                Argument::Callback(Arc::new(move |situation| f(cb(situation))))
            }
        }
    }
}

impl<R: Clone + Send + Sync + 'static, T: BtCallback<R>> From<T> for Argument<R> {
    fn from(cb: T) -> Self {
        Argument::Callback(Arc::new(cb))
    }
}

// Implement Into<Argument<T>> for common primitive types so they can be used as static arguments
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
