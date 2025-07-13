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
}

impl<R: Clone + Send + Sync + 'static, T: BtCallback<R>> From<T> for Argument<R> {
    fn from(cb: T) -> Self {
        Argument::Callback(Arc::new(cb))
    }
}
