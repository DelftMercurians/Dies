// use bevy::prelude::*;
// use dies_core::PlayerCmd;
// use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
// use dies_simulator::Simulation;

fn main() {
//     App::new()
//         .add_plugins(DefaultPlugins)
//         .insert_resource(Simulation::default())
//         .add_systems(Startup, setup_graphics)
//         .add_system(simulation_step)
//         .run();
}

// fn setup_graphics(mut commands: Commands, mut config: ResMut<GizmoConfig>) {
//     // to see the debug lines throug objects
//     config.depth_bias = if config.depth_bias == 0. { -1. } else { 0. };

//     // Add a camera so we can see the debug-render.
//     commands.spawn((
//         Camera3dBundle {
//             transform: Transform::from_xyz(0.0, 15.0, 0.0).looking_at(Vec3::ZERO, Vec3::Y),
//             ..Default::default()
//         },
//         PanOrbitCamera::default(),
//     ));

//     // light
//     commands.spawn(PointLightBundle {
//         point_light: PointLight {
//             intensity: 1500.0,
//             shadows_enabled: true,
//             ..default()
//         },
//         transform: Transform::from_xyz(4.0, 8.0, 4.0),
//         ..default()
//     });
// }

// fn simulation_step(time: Res<Time>, mut simulation: ResMut<Simulation>, mut query: Query<(&mut Transform,)>) {
//     simulation.step(time.delta_seconds_f64() * 10.0);

//     // Handle detection packet
//     if let Some(detection) = simulation.detection() {
//         println!("Detection packet: {:?}", detection);
//         // Update Bevy entities based on detection data
//     }

//     // Handle geometry packet
//     if let Some(geometry) = simulation.geometry() {
//         println!("Geometry packet: {:?}", geometry);
//         // Update Bevy entities based on geometry data
//     }

//     // Example of updating positions or other properties of entities
//     for (mut transform,) in query.iter_mut() {
//         // Adjust transform or other components based on simulation data
//     }

//     // Push commands for players
//     simulation.push_cmd(PlayerCmd::zero(0));
// }
