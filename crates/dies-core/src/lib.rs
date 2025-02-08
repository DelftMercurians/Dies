mod context;
mod field_geometry;
mod frame;
mod math;
mod player_id;
mod robot;
mod settings;
mod debug;
mod vec_map;

// TODO: get rid of this
pub use dies_debug::*;

pub use context::*;
pub use field_geometry::*;
pub use frame::*;
pub use math::*;
pub use player_id::*;
pub use robot::*;
pub use settings::*;
pub use vec_map::*;

pub type Scalar = f64;
pub type Vector2 = nalgebra::Vector2<Scalar>;
pub type Vector3 = nalgebra::Vector3<Scalar>;

mod test{
    fn test() {
        let ctx = 
    }
}