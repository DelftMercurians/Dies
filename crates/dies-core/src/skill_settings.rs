use serde::{Deserialize, Serialize};
use typeshare::typeshare;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[typeshare]
pub struct SkillSettings {
    pub dribbler_radius_near_center: f64,
    pub dribbler_radius_far_center: f64,
    pub dribbler_radius_breakbeam_factor: f64,
    pub fetch_ball_preshoot_offset: f64,
    pub fetch_ball_preshoot_ball_avoidance: f64,
}

impl Default for SkillSettings {
    fn default() -> Self {
        Self {
            dribbler_radius_near_center: 30.0,
            dribbler_radius_far_center: 25.0,
            dribbler_radius_breakbeam_factor: 2.0,
            fetch_ball_preshoot_offset: 150.0,
            fetch_ball_preshoot_ball_avoidance: 0.0,
        }
    }
}
