use std::sync::{atomic::AtomicBool, Arc};
use std::time::Instant;

use dies_core::PlayerId;
use dies_core::{PlayerData, WorldData};

use crate::roles::Role;
use crate::PlayerControlInput;

// pub(super) static PASSER_ID: PlayerId = PlayerId::new(0);
// pub(super) static RECEIVER_ID: PlayerId = PlayerId::new(5);

pub struct Passer {
    timestamp: Instant,
    is_armed: bool,
    has_kicked: Arc<AtomicBool>,
}

impl Role for Passer {
    // assume for now that we stand close to the ball
    // and we can kick it after a few seconds

    fn update(&mut self, _player_data: &PlayerData, _world: &WorldData) -> PlayerControlInput {
        let input = PlayerControlInput::new();
        // let target_angle = self.angle_to_receiver(_player_data, _world);
        // input.with_orientation(target_angle);

        // if (self.timestamp.elapsed().as_secs() > 3) && !self.is_armed {
        //     self.is_armed = true;
        //     self.timestamp = Instant::now();
        //     let kicker = crate::KickerControlInput::Arm;

        //     println!("[PASSER]:Armed");
        //     input.with_kicker(kicker);

        //     return input;
        // } else if self.timestamp.elapsed().as_secs() > 1
        //     && self.is_armed
        //     && !self.has_kicked.load(std::sync::atomic::Ordering::Relaxed)
        // {
        //     input.with_dribbling(0.0);

        //     self.has_kicked
        //         .store(true, std::sync::atomic::Ordering::Relaxed);
        //     self.timestamp = Instant::now();

        //     let kicker = crate::KickerControlInput::Kick;
        //     input.with_kicker(kicker);

        //     println!("[PASSER]: Kicked");

        //     return input;
        // } else if self.timestamp.elapsed().as_secs_f64() < 1.1
        //     && self.has_kicked.load(std::sync::atomic::Ordering::Relaxed)
        // {
        //     // kick for 0.1 seconds

        //     input.with_dribbling(0.0);
        //     let kicker = crate::KickerControlInput::Kick;
        //     input.with_kicker(kicker);

        //     println!("[PASSER]: Kicked2");

        //     return input;
        // }
        input
    }
}
