//! Read-only world state API for strategies.
//!
//! The [`World`] struct provides read-only access to the current world state.
//! All coordinates are in the normalized team-relative frame:
//! - **+x**: Toward opponent's goal (attacking direction)
//! - **-x**: Toward our own goal (defending direction)
//!
//! Strategies never see absolute world coordinates or team color.

use dies_core::{FieldGeometry, Vector2};
use dies_strategy_protocol::{BallState, GameState, PlayerId, PlayerState, WorldSnapshot};

/// A rectangle defined by min and max corners.
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    /// Minimum corner (bottom-left in standard orientation).
    pub min: Vector2,
    /// Maximum corner (top-right in standard orientation).
    pub max: Vector2,
}

impl Rect {
    /// Create a new rectangle from min and max corners.
    pub fn new(min: Vector2, max: Vector2) -> Self {
        Self { min, max }
    }

    /// Create a rectangle centered at a point with given half-width and half-height.
    pub fn from_center(center: Vector2, half_width: f64, half_height: f64) -> Self {
        Self {
            min: Vector2::new(center.x - half_width, center.y - half_height),
            max: Vector2::new(center.x + half_width, center.y + half_height),
        }
    }

    /// Check if a point is inside the rectangle.
    pub fn contains(&self, point: Vector2) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
    }

    /// Get the center of the rectangle.
    pub fn center(&self) -> Vector2 {
        (self.min + self.max) / 2.0
    }

    /// Get the width of the rectangle.
    pub fn width(&self) -> f64 {
        self.max.x - self.min.x
    }

    /// Get the height of the rectangle.
    pub fn height(&self) -> f64 {
        self.max.y - self.min.y
    }
}

/// Read-only access to the current world state.
///
/// All coordinates are in the normalized team-relative frame where +x points
/// toward the opponent's goal.
///
/// # Example
///
/// ```ignore
/// fn update(&mut self, ctx: &mut TeamContext) {
///     let world = ctx.world();
///     
///     // Get ball position
///     if let Some(ball_pos) = world.ball_position() {
///         // Ball is at this position
///     }
///     
///     // Get field geometry
///     let goal = world.opp_goal_center();  // Always at +x
///     let our_goal = world.own_goal_center();  // Always at -x
/// }
/// ```
#[derive(Clone, Debug)]
pub struct World {
    snapshot: WorldSnapshot,
}

impl World {
    /// Create a new World from a WorldSnapshot.
    pub fn new(snapshot: WorldSnapshot) -> Self {
        Self { snapshot }
    }

    // ========== Field Geometry ==========

    /// Get the full field geometry, if available.
    pub fn field(&self) -> Option<&FieldGeometry> {
        self.snapshot.field_geom.as_ref()
    }

    /// Get the field length (distance between goal lines) in mm.
    pub fn field_length(&self) -> f64 {
        self.snapshot
            .field_geom
            .as_ref()
            .map(|f| f.field_length)
            .unwrap_or(9000.0)
    }

    /// Get the field width (distance between touch lines) in mm.
    pub fn field_width(&self) -> f64 {
        self.snapshot
            .field_geom
            .as_ref()
            .map(|f| f.field_width)
            .unwrap_or(6000.0)
    }

    /// Get the center of our own goal (always at -x in normalized coordinates).
    pub fn own_goal_center(&self) -> Vector2 {
        Vector2::new(-self.field_length() / 2.0, 0.0)
    }

    /// Get the center of the opponent's goal (always at +x in normalized coordinates).
    pub fn opp_goal_center(&self) -> Vector2 {
        Vector2::new(self.field_length() / 2.0, 0.0)
    }

    /// Get the goal width in mm.
    pub fn goal_width(&self) -> f64 {
        self.snapshot
            .field_geom
            .as_ref()
            .map(|f| f.goal_width)
            .unwrap_or(1000.0)
    }

    /// Get our penalty area as a rectangle.
    ///
    /// The penalty area is at the -x side of the field.
    pub fn own_penalty_area(&self) -> Rect {
        let field = self.snapshot.field_geom.as_ref();
        let depth = field.map(|f| f.penalty_area_depth).unwrap_or(1000.0);
        let width = field.map(|f| f.penalty_area_width).unwrap_or(2000.0);
        let half_length = self.field_length() / 2.0;

        Rect::new(
            Vector2::new(-half_length, -width / 2.0),
            Vector2::new(-half_length + depth, width / 2.0),
        )
    }

    /// Get the opponent's penalty area as a rectangle.
    ///
    /// The penalty area is at the +x side of the field.
    pub fn opp_penalty_area(&self) -> Rect {
        let field = self.snapshot.field_geom.as_ref();
        let depth = field.map(|f| f.penalty_area_depth).unwrap_or(1000.0);
        let width = field.map(|f| f.penalty_area_width).unwrap_or(2000.0);
        let half_length = self.field_length() / 2.0;

        Rect::new(
            Vector2::new(half_length - depth, -width / 2.0),
            Vector2::new(half_length, width / 2.0),
        )
    }

    /// Get the center circle radius in mm.
    pub fn center_circle_radius(&self) -> f64 {
        self.snapshot
            .field_geom
            .as_ref()
            .map(|f| f.center_circle_radius)
            .unwrap_or(500.0)
    }

    // ========== Ball ==========

    /// Get the ball state, if the ball is detected.
    pub fn ball(&self) -> Option<&BallState> {
        self.snapshot.ball.as_ref()
    }

    /// Get the ball position, if the ball is detected.
    pub fn ball_position(&self) -> Option<Vector2> {
        self.snapshot.ball.as_ref().map(|b| b.position)
    }

    /// Get the ball velocity, if the ball is detected.
    pub fn ball_velocity(&self) -> Option<Vector2> {
        self.snapshot.ball.as_ref().map(|b| b.velocity)
    }

    /// Predict the ball position at time `t` seconds in the future.
    ///
    /// Uses simple linear prediction (ball moves in a straight line).
    /// Returns `None` if the ball is not detected.
    pub fn predict_ball_position(&self, t: f64) -> Option<Vector2> {
        self.snapshot
            .ball
            .as_ref()
            .map(|b| b.position + b.velocity * t)
    }

    // ========== Players ==========

    /// Get all own players (our team's robots).
    pub fn own_players(&self) -> &[PlayerState] {
        &self.snapshot.own_players
    }

    /// Get all opponent players.
    pub fn opp_players(&self) -> &[PlayerState] {
        &self.snapshot.opp_players
    }

    /// Get a specific own player by ID.
    pub fn own_player(&self, id: PlayerId) -> Option<&PlayerState> {
        self.snapshot.own_players.iter().find(|p| p.id == id)
    }

    /// Get a specific opponent player by ID.
    pub fn opp_player(&self, id: PlayerId) -> Option<&PlayerState> {
        self.snapshot.opp_players.iter().find(|p| p.id == id)
    }

    /// Get the list of own player IDs.
    pub fn own_player_ids(&self) -> Vec<PlayerId> {
        self.snapshot.own_players.iter().map(|p| p.id).collect()
    }

    /// Get the list of opponent player IDs.
    pub fn opp_player_ids(&self) -> Vec<PlayerId> {
        self.snapshot.opp_players.iter().map(|p| p.id).collect()
    }

    /// Find the closest own player to a given position.
    pub fn closest_own_player_to(&self, position: Vector2) -> Option<&PlayerState> {
        self.snapshot.own_players.iter().min_by(|a, b| {
            let dist_a = (a.position - position).norm();
            let dist_b = (b.position - position).norm();
            dist_a
                .partial_cmp(&dist_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Find the closest opponent player to a given position.
    pub fn closest_opp_player_to(&self, position: Vector2) -> Option<&PlayerState> {
        self.snapshot.opp_players.iter().min_by(|a, b| {
            let dist_a = (a.position - position).norm();
            let dist_b = (b.position - position).norm();
            dist_a
                .partial_cmp(&dist_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    // ========== Game State ==========

    /// Get the current game state.
    pub fn game_state(&self) -> GameState {
        self.snapshot.game_state
    }

    /// Check if the ball is in play (can be contested).
    pub fn is_ball_in_play(&self) -> bool {
        self.snapshot.game_state.is_ball_in_play()
    }

    /// Check if it's our team's turn to act (e.g., we have a free kick).
    pub fn us_operating(&self) -> bool {
        self.snapshot.us_operating
    }

    /// Get our goalkeeper's player ID, if designated.
    pub fn our_keeper_id(&self) -> Option<PlayerId> {
        self.snapshot.our_keeper_id
    }

    /// Get the player who performed a free kick or kickoff (for double-touch tracking).
    /// Only `Some` until another player touches the ball.
    pub fn freekick_kicker(&self) -> Option<PlayerId> {
        self.snapshot.freekick_kicker
    }

    // ========== Timing ==========

    /// Get the time since the last frame, in seconds.
    pub fn dt(&self) -> f64 {
        self.snapshot.dt
    }

    /// Get the timestamp of this frame, in seconds.
    pub fn timestamp(&self) -> f64 {
        self.snapshot.timestamp
    }

    // ========== Raw Access ==========

    /// Get the raw world snapshot (for advanced use cases).
    pub fn raw_snapshot(&self) -> &WorldSnapshot {
        &self.snapshot
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dies_core::Angle;

    fn make_test_world() -> World {
        World::new(WorldSnapshot {
            timestamp: 1.0,
            dt: 0.016,
            field_geom: Some(FieldGeometry::default()),
            ball: Some(BallState {
                position: Vector2::new(100.0, 200.0),
                velocity: Vector2::new(50.0, 0.0),
                detected: true,
            }),
            own_players: vec![
                PlayerState::new(
                    PlayerId::new(1),
                    Vector2::new(1000.0, 500.0),
                    Vector2::new(0.0, 0.0),
                    Angle::from_radians(0.0),
                ),
                PlayerState::new(
                    PlayerId::new(2),
                    Vector2::new(-500.0, 0.0),
                    Vector2::new(0.0, 0.0),
                    Angle::from_radians(0.0),
                ),
            ],
            opp_players: vec![PlayerState::new(
                PlayerId::new(3),
                Vector2::new(-1000.0, -500.0),
                Vector2::new(0.0, 0.0),
                Angle::from_radians(0.0),
            )],
            game_state: GameState::Run,
            us_operating: true,
            our_keeper_id: Some(PlayerId::new(1)),
            freekick_kicker: None,
        })
    }

    #[test]
    fn test_field_geometry() {
        let world = make_test_world();

        assert_eq!(world.field_length(), 9000.0);
        assert_eq!(world.field_width(), 6000.0);

        // Own goal at -x
        assert!(world.own_goal_center().x < 0.0);
        assert_eq!(world.own_goal_center().y, 0.0);

        // Opponent goal at +x
        assert!(world.opp_goal_center().x > 0.0);
        assert_eq!(world.opp_goal_center().y, 0.0);
    }

    #[test]
    fn test_ball_access() {
        let world = make_test_world();

        assert!(world.ball().is_some());
        assert!(world.ball_position().is_some());

        let pos = world.ball_position().unwrap();
        assert_eq!(pos.x, 100.0);
        assert_eq!(pos.y, 200.0);

        // Predict ball position
        let predicted = world.predict_ball_position(1.0).unwrap();
        assert_eq!(predicted.x, 150.0); // 100 + 50*1
        assert_eq!(predicted.y, 200.0); // 200 + 0*1
    }

    #[test]
    fn test_player_access() {
        let world = make_test_world();

        assert_eq!(world.own_players().len(), 2);
        assert_eq!(world.opp_players().len(), 1);

        assert!(world.own_player(PlayerId::new(1)).is_some());
        assert!(world.own_player(PlayerId::new(99)).is_none());

        let closest = world.closest_own_player_to(Vector2::new(0.0, 0.0)).unwrap();
        assert_eq!(closest.id, PlayerId::new(2)); // Player at (-500, 0) is closer to origin
    }

    #[test]
    fn test_game_state() {
        let world = make_test_world();

        assert_eq!(world.game_state(), GameState::Run);
        assert!(world.is_ball_in_play());
        assert!(world.us_operating());
    }

    #[test]
    fn test_penalty_areas() {
        let world = make_test_world();

        let own_penalty = world.own_penalty_area();
        let opp_penalty = world.opp_penalty_area();

        // Own penalty area at -x
        assert!(own_penalty.min.x < 0.0);
        assert!(own_penalty.max.x < 0.0);

        // Opponent penalty area at +x
        assert!(opp_penalty.min.x > 0.0);
        assert!(opp_penalty.max.x > 0.0);
    }
}
