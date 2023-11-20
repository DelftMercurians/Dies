mod bridge;
mod docker_wrapper;
mod ersim;
mod recv_transport;

pub use ersim::{create_ersim_env, ErSimConfig, ErSimEnvReceiver, ErSimEnvSender};
pub(self) use recv_transport::RecvTransport;

#[cfg(test)]
mod tests {
    use dies_core::EnvEvent;

    use crate::ersim::*;

    #[test_log::test]
    fn test_receive_vision() {
        let (_, mut env_rx) = create_ersim_env(ErSimConfig::default()).unwrap();
        for _ in 0..3 {
            if let Ok(ev) = env_rx.recv() {
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
