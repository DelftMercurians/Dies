use serde::{Serialize, Deserialize};
use nalgebra::Vector3; // Import the Vector3 type from nalgebra.
use nalgebra::Vector2; // Import the Vector3 type from nalgebra.

// Define your smaller container types for players, balls, and field.

#[derive(Serialize, Deserialize, Clone, Debug)]
struct PlayerData {
    // 
    is_blue: bool,
    // Position of the player
    position: Vector2<f64>,
    // Velocity of the player
    velocity: Vector2<f64>,
    // Orientation of the robot [-pi, pi]
    orientation: f64,
    // Angular speed of the robot
    angular_speed: f64
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct BallData {
    // Position of the ball
    position: Vector3<f64>,
    // Velocity of the ball
    velocity: Vector3<f64>
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct FieldGeometry {
    field_length: f64,
    field_width: f64,
    goal_width: f64,
    goal_depth: f64,
    boundary_width: f64,
    penalty_area_depth: f64,
    penalty_area_width: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct FieldCircularArc {
    // A structure to store a single field arc
    // Params:
    // a1: Start angle in counter-clockwise order.
    // a2: End angle in counter-clockwise order.

    index: i64,
    name: String,
    center: Vector2<f64>,
    radius: f64,
    a1: f64,
    a2: f64,
    thickness: f64
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct FieldLineSegment {
    // A single field line segment
    // Params:
    // p1: Start point of segment 
    // p2: End point of segment

    index: i64,
    name: String,
    p1: Vector2<f64>,
    p2: Vector2<f64>,
    thickness: f64
}

// Define the WorldData struct.
#[derive(Serialize, Clone, Debug)]
struct WorldData {
    own_players: Vec<PlayerData>,
    opp_players: Vec<PlayerData>,
    ball: BallData,
    field_geom: FieldGeometry,
    field_circular_arc: FieldCircularArc,
    field_line_segment: FieldLineSegment
}

// Create tracker types for players and the ball.
#[derive(Serialize, Clone, Debug)]
struct PlayerTracker {
    position: Vector2<f64>,
    // Velocity of the player
    velocity: Vector2<f64>,
    // Orientation of the robot [-pi, pi]
    orientation: f64,
    // Angular speed of the robot
    angular_speed: f64 
}

#[derive(Serialize, Clone, Debug)]
struct BallTracker {
    // Position of the ball
    position: Vector3<f64>,
    prev_position: Vector3<f64>,
    velocity: Vector3<f64>
}

impl PlayerTracker {
    pub fn update(&mut self, player_data: &PlayerData) {
        // Update the player tracker with new data.
        // Update fields in the PlayerTracker based on player_data.
        self.position = player_data.position.clone(); 
        self.velocity = player_data.velocity.clone(); 
        self.orientation = player_data.orientation; 
        self.angular_speed = player_data.angular_speed; 
    }
}

impl BallTracker {
    pub fn update(&mut self, frame: &SSL_DetectionFrame) -> BallData {
        let ball_detection = frame.balls.get(0); // Assuming the first ball detection is the one you're interested in.

        let prev_position = self.prev_position;
        let current_position = Vector3::new(ball_detection.x, ball_detection.y, 0.0);
        let velocity = current_position - prev_position;

        // Update the internal state of the tracker.
        self.position = current_position;

        // Construct and return a BallData instance.
        BallData {
            position: current_position,
            velocity,
        }
    }
}

// Define the WorldTracker struct.
struct WorldTracker {
    own_players_tracker: Vec<PlayerTracker>,
    opp_players_tracker: Vec<PlayerTracker>,
    ball_tracker: BallTracker,
    field_geometry: FieldGeometry,
    // Add other fields as needed.
}

impl WorldTracker {
    pub fn new(num_players: usize) -> Self {
        // Initialize the WorldTracker with the given number of players.
        let own_players_tracker = (0..num_players).map(|_| PlayerTracker::update()).collect();
        let opp_players_tracker = (0..num_players).map(|_| PlayerTracker::update()).collect();

        Self {
            own_players_tracker,
            opp_players_tracker,
            ball_tracker: BallTracker::new(),
            field_geometry: FieldGeometry {
                // Initialize field geometry fields here.
            }
        }
    }

    pub fn update_from_protobuf(&mut self, data: SSL_WrapperPacket) -> WorldData {
        // Implement the update logic here to update player and ball trackers.
        // Update the WorldTracker's fields based on the data.
        // Then, create and return a WorldData struct based on the updated information.

        // Example:
        // for (i, player_data) in data.players.iter().enumerate() {
        //     self.own_players_tracker[i].update(PlayerData { /* Extract data from player_data */ });
        // }

        // Update the ball tracker similarly.

        // After updating the trackers, create a WorldData instance and return it.
        WorldData {
            own_players: self.own_players_tracker.iter().map(|tracker| /* Extract player data from tracker */ ).collect(),
            opp_players: self.opp_players_tracker.iter().map(|tracker| /* Extract player data from tracker */ ).collect(),
            ball: /* Extract ball data from ball_tracker */,
            field_geom: self.field_geometry.clone(), // If it doesn't change.
            field_circular_arc: self.field
        }
    }
}
