use self::lock::LockOptions;

mod bootstrap;
mod config;
mod consts;
mod lock;
mod piptools;
mod platform;
mod pyproject;
mod sources;
mod sync;
mod utils;

#[allow(dead_code)]
pub fn sync() {
    sync::sync(sync::SyncOptions { output: utils::CommandOutput::Quiet, dev: false, mode: sync::SyncMode::Regular, force: false, no_lock: false, lock_options: LockOptions::default(), pyproject: None });
}