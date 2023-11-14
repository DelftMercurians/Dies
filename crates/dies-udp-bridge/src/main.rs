use std::net::UdpSocket;
use std::thread;
use std::sync::{Arc, Mutex};

const FORWARD_ADDR: &str = "host.docker.internal:10050";

fn main() -> std::io::Result<()> {
    println!("Starting bridge");

    let fwd_socket = Arc::new(Mutex::new(UdpSocket::bind("0.0.0.0:0")?));

    // Start vision multicast listener
    let vision_socket = UdpSocket::bind("0.0.0.0:10020")?;
    vision_socket.join_multicast_v4(&"224.5.23.2".parse().unwrap(), &"0.0.0.0".parse().unwrap())?;

    let fwd_clone = fwd_socket.clone();
    let t1 = thread::spawn(move || {
        let mut buf = [0u8; 2 * 1024];
        // First byte is packet type
        buf[0] = 0;
        loop {
            match vision_socket.recv_from(&mut buf[1..]) {
                Ok((size, _src)) => {
                    let data = &buf[..(size + 1)];
                    let fwd = fwd_clone.lock().unwrap();
                    let amt = fwd.send_to(data, FORWARD_ADDR).unwrap();
                    assert!(amt == data.len(), "Failed to send entire packet!");
                },
                Err(e) => println!("Vision receive error: {}", e),
            }
        }
    });

    // Start GC multicast listener
    let gc_socket = UdpSocket::bind("0.0.0.0:11003")?;
    gc_socket.join_multicast_v4(&"224.5.23.1".parse().unwrap(), &"0.0.0.0".parse().unwrap())?;

    let fwd_clone = fwd_socket.clone();
    let t2 = thread::spawn(move || {
        let mut buf = [0u8; 2 * 1024];
        // First byte is packet type
        buf[0] = 1;
        loop {
            match gc_socket.recv_from(&mut buf[1..]) {
                Ok((size, _src)) => {
                    let data = &buf[..(size + 1)];
                    let fwd = fwd_clone.lock().unwrap();
                    let amt = fwd.send_to(data, FORWARD_ADDR).unwrap();
                    assert!(amt == data.len(), "Failed to send entire packet!");
                },
                Err(e) => println!("GC receive error: {}", e),
            }
        }
    });

    // Join the threads
    t1.join().unwrap();
    t2.join().unwrap();

    Ok(())
}
