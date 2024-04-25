use crate::pid::PID;
use dies_core::{FieldGeometry, GameState, PlayerCmd, PlayerData};
use nalgebra::Vector2;
use std::time::{Duration, Instant};

pub struct PlayerController {
    id: u32,
    position_pid: PID<Vector2<f32>>,
    heading_pid: PID<f32>,
    kick_timer: Option<Instant>,
    last_pos: Vector2<f32>,
    last_orientation: f32,
    frame_missings: u32,
}

impl PlayerController {
    /// Create a new player controller with the given ID.
    pub fn new(id: u32) -> Self {
        let mut heading_pid = PID::new(2.0, 0.002, 0.0);
        // Fix the heading for now
        heading_pid.set_setpoint(0.0);
        Self {
            id,
            position_pid: PID::new(0.7, 0.0, 0.0),
            heading_pid,
            kick_timer: None,
            last_pos: Vector2::new(0.0, 0.0),
            last_orientation: 0.0,
            frame_missings: 0,
        }
    }

    /// Increment the missing frame count, return whether it's missing for too long
    pub fn increment_frame_missings(&mut self) -> bool {
        let threshold: u32 = 100; // for example 1.5s
        self.frame_missings += 1;
        self.frame_missings > threshold
    }

    /// Set the target position for the player.
    pub fn set_target_pos(&mut self, setpoint: Vector2<f32>) {
        self.position_pid.set_setpoint(setpoint);
    }

    /// Keep track of current position from the frame
    pub fn update_current_pos(&mut self, state: &PlayerData) {
        self.frame_missings = 0;
        self.last_pos = state.position;
        self.last_orientation = state.orientation;
    }

    /// Update the controller with the current state of the player.
    pub fn update(
        &mut self,
        is_dribbling: bool,
        is_kicking: bool,
    ) -> PlayerCmd {
        let mut cmd: PlayerCmd = PlayerCmd {
            id: self.id,
            ..Default::default()
        };
        self.set_speed(&mut cmd);

        self.handle_dribbling(&mut cmd, is_dribbling);
        self.handle_kicking(&mut cmd, is_kicking);

        cmd
    }

    /// set speed of the robot based on pids
    pub fn set_speed(&mut self, cmd: &mut PlayerCmd) {
        let pos_u = self.position_pid.update(self.last_pos);
        let head_u = self.heading_pid.update(self.last_orientation);
        let local_u = rotate_vector(Vector2::new(pos_u.x, pos_u.y), -self.last_orientation);
        cmd.sx = local_u.x;
        cmd.sy = local_u.y;
        cmd.w = head_u;
    }

    /// handle the dribbling.
    pub fn handle_dribbling(&mut self, cmd: &mut PlayerCmd, is_dribbling: bool) {
        // for example, 60 rad/s
        if is_dribbling {
            cmd.dribble_speed = 60.0;
        }
    }

    /// handle the kicking.
    pub fn handle_kicking(&mut self, cmd: &mut PlayerCmd, is_kicking: bool) {
        if !is_kicking {
            return;
        }
        if self.kick_timer.is_none() {
            self.kick_timer = Some(Instant::now());
            cmd.arm = true;
        } else {
            let elapsed = self.kick_timer.unwrap().elapsed();
            if elapsed >= Duration::from_millis(1000) {
                cmd.kick = true;
                self.kick_timer = None;
            }
        }
    }

    /// speed limit check
    pub fn speed_limit_check(&self, cmd: &mut PlayerCmd, velocity: Vector2<f32>) {
        // for example 1500 mm/s
        let max_speed = 1500.0;
        let speed = (velocity.x.powi(2) + velocity.y.powi(2)).sqrt();
        if speed > max_speed {
            let scale = max_speed / speed;
            cmd.sx *= scale;
            cmd.sy *= scale;
        }
    }

    /// out of bound detection
    pub fn out_of_bound_detection(
        &self,
        cmd: &mut PlayerCmd,
        game_state: GameState,
        field_geometry: FieldGeometry,
    ) {
        // TODO:
        // what is the excepted behavior of this?? I think pid will drag it back when
        // it is out of bound, and since pid is the only way to control the robot, it may be unnecessary
        // to do this check.
    }
}

fn rotate_vector(v: Vector2<f32>, angle: f32) -> Vector2<f32> {
    let rot = nalgebra::Rotation2::new(angle);
    rot * v
}
