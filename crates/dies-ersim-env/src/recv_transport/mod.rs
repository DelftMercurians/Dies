pub(self) const BUF_SIZE: usize = 2 * 1024;

#[cfg(target_os = "linux")]
mod linux;

#[cfg(target_os = "linux")]
pub use linux::*;

#[cfg(target_os = "windows")]
mod win;

#[cfg(target_os = "windows")]
pub use win::*;
