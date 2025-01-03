mod field_geometry;
mod math;
mod player_id;
mod robot;
mod vec_map;
mod world_frame;

pub use dies_debug::*;
pub use field_geometry::*;
pub use math::*;
pub use player_id::*;
pub use robot::*;
pub use vec_map::*;
pub use world_frame::*;

pub type Scalar = f64;
pub type Vector2 = nalgebra::Vector2<Scalar>;
pub type Vector3 = nalgebra::Vector3<Scalar>;
