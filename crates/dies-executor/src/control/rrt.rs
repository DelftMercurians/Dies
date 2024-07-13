use dies_core::{Vector2, WorldData};
use rand::distributions::{Distribution, Uniform};

const PLAYER_RADIUS: f64 = 80.0;

fn is_free(pos: Vector2, world: &WorldData) -> bool {
    let mut free = true;
    for player in world.own_players.iter().chain(world.opp_players.iter()) {
        if (player.position - pos).norm() < 2.0 * PLAYER_RADIUS {
            free = false;
            break;
        }
    }
    free
}

pub fn find_path(start: Vector2, goal: Vector2, world: &WorldData) -> Vec<Vector2> {
    let half_w = world
        .field_geom
        .as_ref()
        .map(|g| g.field_width)
        .unwrap_or_default()
        / 2.0;
    let half_l = world
        .field_geom
        .as_ref()
        .map(|g| g.field_length)
        .unwrap_or_default()
        / 2.0;

    let result = rrt::dual_rrt_connect(
        &start.as_slice(),
        &goal.as_slice(),
        |p: &[f64]| is_free(Vector2::new(p[0], p[1]), world),
        || {
            let mut rng = rand::thread_rng();
            let x = Uniform::new(-half_w, half_w).sample(&mut rng);
            let y = Uniform::new(-half_l, half_l).sample(&mut rng);
            vec![x, y]
        },
        100.0,
        10_000,
    )
    .unwrap();

    result
        .into_iter()
        .map(|p| Vector2::new(p[0], p[1]))
        .collect()
}
