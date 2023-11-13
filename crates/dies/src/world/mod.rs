use nalgebra::{Vector2, Vector3};
use serde::{de, Deserialize, Serialize};

use crate::protos::{ssl_detection::SSL_DetectionFrame, ssl_wrapper::SSL_WrapperPacket};

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
        let own_players_tracker = (0..num_players).map(|_| PlayerTracker::update()).collect();
        let opp_players_tracker = (0..num_players).map(|_| PlayerTracker::update()).collect();
        // QUESTION parameters for the update function

        Self {
            own_players_tracker,
            opp_players_tracker,
            ball_tracker: BallTracker::new(),
            field_geometry: None,
        }
    }

    pub fn update_from_protobuf(&mut self, data: &SSL_WrapperPacket) -> WorldData {
        // If no geom data is in WorldTracker
        // and we received a wrapper packet with geom data
        // extract it into our geometry structs

        // if we received a wrapper packet with no geom data return
        // QUESTION how should I handle this?
        // if data.geometry.is_none() {
        //     return None;
        // }

        let detection_frame = data.detection.as_ref().unwrap();

        let own_players: Vec<PlayerData> = self
            .own_players_tracker
            .iter_mut()
            .filter_map(|tracker| tracker.update(detection_frame))
            .collect();

        let opp_players: Vec<PlayerData> = self
            .opp_players_tracker
            .iter_mut()
            .filter_map(|tracker| tracker.update(detection_frame))
            .collect();

        let ball: BallData = self.ball_tracker.update(detection_frame).unwrap();

        let fild_geom = self.field_geometry.as_ref().unwrap();

        // QUESTION where do I get these from?
        // let field_circular_arc = &fild_geom.field_circular_arc;
        // let field_line_segment = &fild_geom.field_line_segment;

        WorldData {
            own_players,
            opp_players,
            ball,
            field_geom: fild_geom,
            field_circular_arc,
            field_line_segment,
        }
    }
}
