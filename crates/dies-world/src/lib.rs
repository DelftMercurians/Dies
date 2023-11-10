use serde::Serialize;

use dies_protos::{
    ssl_vision_detection::SSL_DetectionFrame, ssl_vision_wrapper::SSL_WrapperPacket,
};

mod ball;
mod geom;
mod player;

use ball::{BallData, BallTracker};
use geom::{FieldCircularArc, FieldGeometry, FieldLineSegment};
use player::{PlayerData, PlayerTracker};

/// A struct to store the world state from a single frame.
#[derive(Serialize, Clone, Debug)]
struct WorldData<'a> {
    own_players: Vec<PlayerData>,
    opp_players: Vec<PlayerData>,
    ball: BallData,
    field_geom: &'a FieldGeometry,
    field_circular_arc: &'a FieldCircularArc,
    field_line_segment: &'a FieldLineSegment,
}
/// A struct to track the world state.
struct WorldTracker {
    own_players_tracker: Vec<PlayerTracker>,
    opp_players_tracker: Vec<PlayerTracker>,
    ball_tracker: BallTracker,
    field_geometry: Option<FieldGeometry>,
}

impl WorldTracker {
    pub fn new(num_players: usize) -> Self {
        // Initialize the WorldTracker with the given number of players.
        let own_players_tracker = (0..num_players).map(|_| PlayerTracker::new()).collect();
        let opp_players_tracker = (0..num_players).map(|_| PlayerTracker::new()).collect();

        Self {
            own_players_tracker,
            opp_players_tracker,
            ball_tracker: BallTracker::new(),
            field_geometry: None,
        }
    }

    pub fn update_from_protobuf(&mut self, data: &SSL_WrapperPacket) -> WorldData {
        todo!()
    }
}
