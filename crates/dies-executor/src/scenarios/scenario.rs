use anyhow::{bail, Result};
use std::{collections::HashSet, time::Duration};

use dies_basestation_client::{BasestationClient, BasestationClientConfig};
use dies_core::{
    Angle, BallPlacement, ExecutorSettings, PlayerData, PlayerId, PlayerPlacement, ScenarioInfo,
    Vector2, Vector3, WorldData,
};
use dies_simulator::{SimulationBuilder, SimulationConfig};
use dies_ssl_client::{VisionClient, VisionClientConfig};
use dies_world::WorldTracker;

use crate::{strategy::Strategy, Executor};

const LIVE_CHECK_INTERVAL: Duration = Duration::from_millis(100);
const LIVE_CHECK_TIMEOUT: Duration = Duration::from_secs(30);
const SIMULATION_FIELD_MARGIN: f64 = 0.1;

pub struct ScenarioSetup {
    /// Initial ball position. If `None`, ther won
    ball: BallPlacement,
    /// Initial setup for own players.
    own_players: Vec<PlayerPlacement>,
    /// Initial setup for opponent players.
    opp_players: Vec<PlayerPlacement>,
    /// Position tolerance for player and ball positions in mm.
    tolerance: f64,
    /// Yaw tolerance for players in rad
    yaw_tolerance: f64,
    /// Strategy to use.
    strategy: Box<dyn Strategy>,
}

impl ScenarioSetup {
    pub fn new(strategy: impl Strategy + 'static) -> Self {
        Self {
            ball: BallPlacement::NoBall,
            own_players: Vec::new(),
            opp_players: Vec::new(),
            tolerance: 10.0,
            yaw_tolerance: 10.0f64.to_radians(),
            strategy: Box::new(strategy),
        }
    }

    /// Sets the ball to be at a specific position.
    pub fn add_ball_at(&mut self, ball: Vector3) -> &mut Self {
        self.ball = BallPlacement::Position(ball);
        self
    }

    /// Sets the ball to be at any position.
    pub fn add_ball(&mut self) -> &mut Self {
        self.ball = BallPlacement::AnyPosition;
        self
    }

    /// Add own player at a random position.
    pub fn add_own_player(&mut self) -> &mut Self {
        self.own_players.push(PlayerPlacement {
            position: None,
            yaw: None,
        });
        self
    }

    /// Add an own player at a specific position.
    pub fn add_own_player_at(&mut self, player: Vector2) -> &mut Self {
        self.own_players.push(PlayerPlacement {
            position: Some(player),
            yaw: None,
        });
        self
    }

    /// Add an opponent player at a specific position.
    pub fn add_opp_player_at(&mut self, player: Vector2) -> &mut Self {
        self.opp_players.push(PlayerPlacement {
            position: Some(player),
            yaw: None,
        });
        self
    }

    /// Get the scenario information.
    pub fn get_info(&self) -> ScenarioInfo {
        ScenarioInfo {
            own_player_placements: self.own_players.clone(),
            opponent_player_placements: self.opp_players.clone(),
            ball_placement: self.ball.clone(),
            tolerance: self.tolerance,
            yaw_tolerance: self.yaw_tolerance,
        }
    }

    /// Create an executor in simulation mode from this setup.
    pub fn into_simulation(
        self,
        settings: ExecutorSettings,
        sim_config: SimulationConfig,
    ) -> Executor {
        let field_width = sim_config.field_geometry.field_width as f64;
        let field_length = sim_config.field_geometry.field_length as f64;
        let mut builder = SimulationBuilder::new(sim_config);

        match self.ball {
            BallPlacement::Position(pos) => {
                builder = builder.add_ball(pos);
            }
            BallPlacement::AnyPosition => {
                let pos2 = random_pos(field_width, field_length);
                builder = builder.add_ball(Vector3::new(pos2.x, pos2.y, 0.0));
            }
            BallPlacement::NoBall => {}
        }

        for player in self.own_players.iter() {
            let (position, yaw) = player_into_simulation(&player, field_width, field_length);
            builder = builder.add_own_player(position, yaw);
        }

        for player in self.opp_players.iter() {
            let (position, yaw) = player_into_simulation(&player, field_width, field_length);
            builder = builder.add_opp_player(position, yaw);
        }

        let sim = builder.build();

        Executor::new_simulation(settings, self.strategy, sim)
    }

    /// Create an executor in live mode from this setup.
    pub async fn into_live(
        self,
        settings: ExecutorSettings,
        ssl_config: VisionClientConfig,
        bs_config: BasestationClientConfig,
    ) -> Result<Executor> {
        // Wait for the setup check to succeed
        let mut tracker = WorldTracker::new(&settings.tracker_settings);
        let mut ssl_client = VisionClient::new(ssl_config.clone()).await?;
        let mut check_interval = tokio::time::interval(LIVE_CHECK_INTERVAL);
        let max_iterations = LIVE_CHECK_TIMEOUT.as_millis() / LIVE_CHECK_INTERVAL.as_millis();
        let mut iterations = 0;
        loop {
            let packet = ssl_client.recv().await?;
            tracker.update_from_vision(&packet);

            if self.check_live(tracker.get()) {
                break;
            }

            iterations += 1;
            if iterations >= max_iterations {
                bail!("Timeout while waiting for scenario to be live");
            }

            check_interval.tick().await;
        }

        let bs_client = BasestationClient::new(bs_config)?;
        Ok(Executor::new_live(
            settings,
            self.strategy,
            ssl_client,
            bs_client,
        ))
    }

    /// Check whether the current world state matches the scenario setup.
    fn check_live(&self, world: WorldData) -> bool {
        // Check ball
        match (&self.ball, world.ball) {
            (BallPlacement::Position(target), Some(ball)) => {
                if (target - ball.position).norm() > self.tolerance {
                    return false;
                }
            }
            (BallPlacement::AnyPosition, Some(_)) => {}
            (BallPlacement::NoBall, _) => {}
            _ => return false,
        }

        // Check own players
        let mut available_ids = world
            .own_players
            .iter()
            .map(|p| p.id)
            .collect::<HashSet<_>>();
        for player in self.own_players.iter() {
            if let Some(id) = find_player(
                &player,
                &available_ids,
                &world.own_players,
                self.tolerance,
                self.yaw_tolerance,
            ) {
                available_ids.remove(&id);
            } else {
                return false;
            }
        }

        // Check opponent players
        let mut available_ids = world
            .opp_players
            .iter()
            .map(|p| p.id)
            .collect::<HashSet<_>>();
        for player in self.opp_players.iter() {
            if let Some(id) = find_player(
                &player,
                &available_ids,
                &world.opp_players,
                self.tolerance,
                self.yaw_tolerance,
            ) {
                available_ids.remove(&id);
            } else {
                return false;
            }
        }

        true
    }
}

fn player_into_simulation(
    placement: &PlayerPlacement,
    field_width: f64,
    field_length: f64,
) -> (Vector2, Angle) {
    let position = match placement.position {
        Some(pos) => pos,
        None => random_pos(field_width, field_length),
    };
    dbg!(placement.position, position);

    let yaw = match placement.yaw {
        Some(yaw) => yaw,
        None => Angle::default(),
    };

    (position, yaw)
}

/// Check whether there is a player that can be used to fill this slot.
fn find_player(
    placement: &PlayerPlacement,
    available_ids: &HashSet<PlayerId>,
    players: &Vec<PlayerData>,
    tolerance: f64,
    yaw_tolerance: f64,
) -> Option<PlayerId> {
    let id = if let Some(target) = placement.position {
        let mut closest_player = None;
        let mut closest_distance = f64::INFINITY;
        for player in players.iter() {
            if available_ids.contains(&player.id) {
                let distance = (player.position - target).norm();
                if distance < closest_distance {
                    closest_distance = distance;
                    closest_player = Some(player.id);
                }
            }
        }
        closest_player
    } else {
        available_ids.iter().next().copied()
    };

    if let Some(player) = id.map(|id| players.iter().find(|p| p.id == id)).flatten() {
        match placement.position {
            Some(target) => {
                if (player.position - target).norm() > tolerance {
                    return None;
                }
            }
            None => {}
        }

        match placement.yaw {
            Some(target) => {
                if (player.yaw - target).radians().abs() > yaw_tolerance {
                    return None;
                }
            }
            None => {}
        }
        id
    } else {
        None
    }
}

fn random_pos(field_width: f64, field_length: f64) -> Vector2 {
    let w = field_width - 2.0 * (SIMULATION_FIELD_MARGIN * field_width);
    let l = field_length - 2.0 * (SIMULATION_FIELD_MARGIN * field_length);
    Vector2::new(
        (rand::random::<f64>() - 0.5) * w,
        (rand::random::<f64>() - 0.5) * l,
    )
}

#[cfg(test)]
mod tests {
    use crate::strategy::AdHocStrategy;

    use super::*;
    use dies_core::Angle;
    use dies_core::BallData;
    use dies_core::WorldData;

    #[test]
    fn test_scenario_setup_check_live() {
        let setup = ScenarioSetup {
            ball: BallPlacement::Position(Vector3::new(0.0, 0.0, 0.0)),
            own_players: vec![PlayerPlacement {
                position: Some(Vector2::new(100.0, 0.0)),
                yaw: Some(Angle::from_degrees(0.0)),
            }],
            opp_players: vec![],
            tolerance: 10.0,
            yaw_tolerance: 10.0f64.to_radians(),
            strategy: Box::new(AdHocStrategy::new()),
        };

        let mut world = WorldData {
            ball: None,
            own_players: vec![PlayerData {
                position: Vector2::new(0.0, 0.0),
                velocity: Vector2::zeros(),
                yaw: Angle::from_degrees(0.0),
                ..PlayerData::new(PlayerId::new(0))
            }],
            opp_players: vec![],
            ..Default::default()
        };

        assert!(!setup.check_live(world.clone()));

        world.ball = Some(BallData {
            timestamp: 0.0,
            position: Vector3::zeros(),
            velocity: Vector3::zeros(),
        });
        world.own_players[0].position = Vector2::new(91.0, 0.0);

        assert!(setup.check_live(world.clone()));
    }
}
