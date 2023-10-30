use bevy::prelude::*;

// dimensions are in m

pub const ROB_START_POS: [[f32; 2]; 12] = [
    [-TERR_H_X * 0.9, 0.0],                   // 1
    [-TERR_H_X * 0.4, -TERR_H_Z * 0.4], // 2
    [-TERR_H_X * 0.4, -TERR_H_Z * 0.8], // 3
    [-TERR_H_X * 0.4, TERR_H_Z * 0.2], // 4
    [-TERR_H_X * 0.4, TERR_H_Z * 0.6], // 5
    [-TERR_H_X * 0.1, 0.0],                   // 6
    [TERR_H_X * 0.1, 0.0],                    // 7
    [TERR_H_X * 0.4, -TERR_H_Z * 0.4],  // 8
    [TERR_H_X * 0.4, -TERR_H_Z * 0.8],  // 9
    [TERR_H_X * 0.4, TERR_H_Z * 0.2],  // 10
    [TERR_H_X * 0.4, TERR_H_Z * 0.6],  // 11
    [TERR_H_X * 0.9, 0.0],                    // 12
];

pub const REST_COEF: f32 = 0.7;
pub const FRICTION_COEF_TERR: f32 = 1.0;
pub const FRICTION_COEF_BALL: f32 = 1.0;
pub const FRICTION_COEF_ROB: f32 = 0.7;

pub const TERR_H_X: f32 = 13.4 / 2.0; // 13.4 m
pub const TERR_H_Z: f32 = 10.4 / 2.0; // 10.4 m

pub const WALL_H_Y: f32 = 1.0; // 100 cm
pub const WALL_H_X: f32 = 0.1; // 10 cm

pub const BALL_RADIUS: f32 = 0.04267 / 2.0; // 42.67 mm diameter

pub const ROB_R: f32 = 0.15 / 2.0; // 15 cm diameter
pub const ROB_H: f32 = 0.18; // 18 cm height

pub const SHOOTER_H_S: f32 = 0.02; // 5 cm height
pub const SHOOTER_RANGE: f32 = 0.01; // 1 cm range
pub const SHOOTER_POS_Y: f32 = 0.01; // 1 cm range

pub const TERR_COL: Color = Color::rgb(0.3, 0.5, 0.3);
pub const BALL_COL: Color = Color::rgb(1.0, 0.5, 0.0);
pub const ROB_COL: Color = Color::rgb(0.0, 0.0, 0.0);
pub const SHOOTER_COL: Color = Color::rgb(1.0, 0.0, 0.0);

// pub const TIME_STEP: f32 = 1.0 / 60.0;
