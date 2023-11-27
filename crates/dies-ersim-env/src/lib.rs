mod bridge;
mod docker_wrapper;
mod ersim;
mod recv_transport;

pub use ersim::{ErSimConfig, ErSimEnvReceiver, ErSimEnvSender};
pub(self) use recv_transport::RecvTransport;

#[cfg(test)]
mod tests {
    use dies_core::{EnvConfig, EnvEvent};

    use crate::ersim::*;

    #[test_log::test]
    fn test_receive_vision() {
        let (_, mut env_rx) = ErSimConfig::default().build().unwrap();
        let mut got_gc = false;
        let mut got_vis = false;
        for _ in 0..10 {
            if let Ok(ev) = env_rx.recv() {
                match ev {
                    EnvEvent::GcRefereeMsg(_) => {
                        log::info!("Got gc message");
                        got_gc = true;
                    }
                    EnvEvent::VisionMsg(_) => {
                        log::info!("Got vision message");
                        got_vis = true;
                    }
                }
                if got_gc && got_vis {
                    return;
                }
            }
        }
        // Error after 10 iterations if we haven't received a vision message
        assert!(got_vis, "Failed to receive vision message");
        assert!(got_gc, "Failed to receive gc message");
    }
}
