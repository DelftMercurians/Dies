//! Connection-reliability trial harness for the basestation.
//!
//! Reproduces the exact production connect path (`Monitor::start()` +
//! `connect_to()`, same as `BasestationHandle::spawn`) repeatedly against the
//! real hardware and reports the failure rate. Used to investigate intermittent
//! connect-on-start failures and to validate a retry-based fix.
//!
//! Usage:
//!   cargo run --example connect_trials -- [TRIALS] [MODE] [RETRIES] [RETRY_MS] [SETTLE_MS] [PORT]
//!     TRIALS    number of independent trials              (default 30)
//!     MODE      "single" | "retry"                        (default single)
//!     RETRIES   max attempts per trial in retry mode      (default 5)
//!     RETRY_MS  delay between attempts in retry mode       (default 50)
//!     SETTLE_MS delay between trials (let device settle)   (default 300)
//!     PORT      serial port                                (default /dev/ttyACM0)

use std::time::{Duration, Instant};

use glue::Monitor;

fn main() {
    let mut args = std::env::args().skip(1);
    let trials: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(30);
    let mode = args.next().unwrap_or_else(|| "single".to_string());
    let retries: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);
    let retry_ms: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(50);
    let settle_ms: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(300);
    let port = args.next().unwrap_or_else(|| "/dev/ttyACM0".to_string());

    eprintln!(
        "trials={trials} mode={mode} retries={retries} retry_ms={retry_ms} settle_ms={settle_ms} port={port}"
    );

    let mut successes = 0usize;
    let mut first_try_successes = 0usize;
    let mut total_attempts_used = 0usize;
    // histogram of how many attempts it took to succeed (1-based)
    let mut attempts_hist: Vec<usize> = vec![0; retries.max(1) + 1];

    for trial in 0..trials {
        let monitor = Monitor::start();

        let attempts_allowed = if mode == "retry" { retries } else { 1 };
        let mut connected = false;
        let mut attempts_used = 0usize;
        let t0 = Instant::now();

        for attempt in 1..=attempts_allowed {
            attempts_used = attempt;
            match monitor.connect_to(&port) {
                Ok(()) => {
                    connected = true;
                    break;
                }
                Err(()) => {
                    if attempt < attempts_allowed {
                        std::thread::sleep(Duration::from_millis(retry_ms));
                    }
                }
            }
        }
        let elapsed = t0.elapsed();

        // Mirror production's post-connect liveness check: verify the monitor
        // reports a live link shortly after a successful connect_to().
        let mut live = false;
        if connected {
            let mut m = monitor;
            for _ in 0..50 {
                if m.is_connected() {
                    live = true;
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            total_attempts_used += attempts_used;
            if attempts_used == 1 {
                first_try_successes += 1;
            }
            if attempts_used < attempts_hist.len() {
                attempts_hist[attempts_used] += 1;
            }
            if live {
                successes += 1;
            }
            m.stop();
        } else {
            monitor.stop();
        }

        eprintln!(
            "trial {:>3}: connected={} live={} attempts={} elapsed={:?}",
            trial + 1,
            connected,
            live,
            attempts_used,
            elapsed
        );

        std::thread::sleep(Duration::from_millis(settle_ms));
    }

    eprintln!("\n==== SUMMARY ====");
    eprintln!("mode:                 {mode}");
    eprintln!("trials:               {trials}");
    eprintln!(
        "live successes:       {successes}/{trials} ({:.1}%)",
        100.0 * successes as f64 / trials as f64
    );
    eprintln!(
        "first-attempt OK:     {first_try_successes}/{trials} ({:.1}%)",
        100.0 * first_try_successes as f64 / trials as f64
    );
    if mode == "retry" && successes > 0 {
        eprintln!(
            "avg attempts (when connected): {:.2}",
            total_attempts_used as f64 / (successes.max(1)) as f64
        );
        eprintln!("attempts-to-connect histogram:");
        for (n, count) in attempts_hist.iter().enumerate().skip(1) {
            if *count > 0 {
                eprintln!("  {n} attempt(s): {count}");
            }
        }
    }

    // Exit code = number of failed trials (0 = all good), so callers can count
    // failures reliably without scraping stdout.
    std::process::exit((trials - successes).min(125) as i32);
}
