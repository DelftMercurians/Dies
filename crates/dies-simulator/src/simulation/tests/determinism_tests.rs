use super::*;

/// Build the standard seeded 6v6 layout and step it `n` frames at 60 Hz with
/// no commands (robots settle, ball rests), returning the trajectory
/// fingerprint folded over every frame.
fn run(seed: u64, n: usize) -> u64 {
    let config = SimulationConfig {
        seed,
        ..Default::default()
    };
    let mut sim = SimulationBuilder::default_seeded(config)
        .with_controlled_teams(true, true)
        .build();
    let mut hash: u64 = 0;
    for _ in 0..n {
        sim.step(1.0 / 60.0);
        hash = hash.rotate_left(7).wrapping_mul(0x0000_0100_0000_01b3) ^ sim.state_fingerprint();
    }
    hash
}

#[test]
fn same_seed_is_deterministic() {
    // The whole point of headless self-play: identical seed → identical run.
    assert_eq!(run(42, 300), run(42, 300));
}

#[test]
fn different_seeds_diverge() {
    // Pose jitter must actually vary the match, otherwise sweeping seeds for
    // A-vs-B evaluation is pointless.
    assert_ne!(run(1, 300), run(2, 300));
}
