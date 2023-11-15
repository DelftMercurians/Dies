mod bridge;
mod docker_wrapper;
mod ersim;
mod recv_transport;

pub use ersim::{ErSimConfig, ErSimEnv};
pub(self) use recv_transport::RecvTransport;

#[cfg(test)]
mod tests {
    use dies_core::{Env, EnvEvent};

    use crate::ersim::*;

    #[test_log::test]
    fn test_receive_vision() {
        let mut env = ErSimEnv::new(ErSimConfig::default()).unwrap();
        for _ in 0..3 {
            for ev in env.recv() {
                log::info!("Received event: {:?}", ev);
                if let EnvEvent::VisionMsg(_) = ev {
                    return;
                }
            }
        }
        // Error after 3 iterations if we haven't received a vision message
        panic!("Did not receive vision event");
    }
}
