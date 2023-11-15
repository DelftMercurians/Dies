use std::net::{IpAddr, SocketAddr, ToSocketAddrs, UdpSocket};
use std::sync::{Arc, Mutex};
use std::thread;

#[allow(dead_code)]
fn main() -> std::io::Result<()> {
    println!("Starting bridge");

    // Attempt to look up the the FORWARD_ADDR
    let fwd_addr = {
        let addr_it = "host.docker.internal:10050".to_socket_addrs();
        let addr = if let Ok(mut addr_it) = addr_it {
            addr_it.next()
        } else {
            None
        };
        addr.unwrap_or(SocketAddr::new(IpAddr::from([127, 0, 0, 1]), 10050))
    };

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
                    let amt = fwd.send_to(data, &fwd_addr).unwrap();
                    assert!(amt == data.len(), "Failed to send entire packet!");
                }
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
                    let amt = fwd.send_to(data, &fwd_addr).unwrap();
                    assert!(amt == data.len(), "Failed to send entire packet!");
                }
                Err(e) => println!("GC receive error: {}", e),
            }
        }
    });

    // Join the threads
    t1.join().unwrap();
    t2.join().unwrap();

    Ok(())
}
