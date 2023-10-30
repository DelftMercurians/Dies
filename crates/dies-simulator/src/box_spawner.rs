use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use crate::consts::*;

pub fn spawn_box_1(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    size_h_x: f32,
    size_h_y: f32,
    size_h_z: f32,
    x: f32,
    y: f32,
    z: f32,
) {
    commands
        .spawn(PbrBundle {
            mesh: meshes.add(Mesh::from(Mesh::from(shape::Box {
                min_x: -size_h_x,
                max_x: size_h_x,
                min_y: -size_h_y,
                max_y: size_h_y,
                min_z: -size_h_z,
                max_z: size_h_z,
            }))),
            material: materials.add(TERR_COL.into()),
            ..Default::default()
        })
        .insert(RigidBody::Fixed)
        .insert(Collider::cuboid(size_h_x, size_h_y, size_h_z))
        .insert(Restitution::coefficient(REST_COEF))
        .insert(Friction {
            coefficient: FRICTION_COEF_TERR,
            combine_rule: CoefficientCombineRule::Multiply,
            ..Default::default()
        })
        .insert(TransformBundle::from(Transform::from_xyz(x, y, z)));
}
