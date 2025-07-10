module static;

op *(Vec2, float) -> Vec2;

op *(Vec3, float) -> Vec3;

op *(float, Vec2) -> Vec2;

op *(float, Vec3) -> Vec3;

op +(Vec2, Vec2) -> Vec2;

op +(Vec2, Vec3) -> Vec2;

op +(Vec3, Vec2) -> Vec2;

op +(Vec3, Vec3) -> Vec3;

op -(Vec2) -> Vec2;

op -(Vec3) -> Vec3;

op -(Vec2, Vec2) -> Vec2;

op -(Vec2, Vec3) -> Vec2;

op -(Vec3, Vec2) -> Vec2;

op -(Vec3, Vec3) -> Vec3;

op /(Vec2, float) -> Vec2;

op /(Vec3, float) -> Vec3;

fn angle_to(_: Vec2, _: Vec2) -> float;

fn cast_ray(_: RobotSituation, _: ArrayStorage<float, 2, 1>>, _: ArrayStorage<float, 2, 1>>) -> map;

fn closest_opp_player_to_me(_: RobotSituation) -> ?;

fn closest_opp_player_to_position(_: RobotSituation, _: ArrayStorage<float, 2, 1>>) -> ?;

fn closest_own_player_to_ball(_: RobotSituation) -> ?;

fn closest_own_player_to_me(_: RobotSituation) -> ?;

fn closest_own_player_to_position(_: RobotSituation, _: ArrayStorage<float, 2, 1>>) -> ?;

fn count_opp_players_where(_: RobotSituation, _: Fn) -> int;

fn count_own_players_where(_: RobotSituation, _: Fn) -> int;

fn distance_to(_: Vec2, _: Vec2) -> float;

fn distance_to_ball(_: RobotSituation) -> float;

fn distance_to_nearest_wall(_: RobotSituation) -> float;

fn distance_to_opp_penalty_area(_: RobotSituation) -> float;

fn distance_to_own_penalty_area(_: RobotSituation) -> float;

fn distance_to_player(_: RobotSituation, _: PlayerId) -> float;

fn distance_to_position(_: RobotSituation, _: ArrayStorage<float, 2, 1>>) -> float;

fn distance_to_wall_in_direction(_: RobotSituation, _: float) -> float;

fn filter_opp_players_by(_: RobotSituation, _: Fn) -> PlayerData>;

fn filter_own_players_by(_: RobotSituation, _: Fn) -> PlayerData>;

fn find_opp_player_max_by(_: RobotSituation, _: Fn) -> ?;

fn find_opp_player_min_by(_: RobotSituation, _: Fn) -> ?;

fn find_own_player_max_by(_: RobotSituation, _: Fn) -> ?;

fn find_own_player_min_by(_: RobotSituation, _: Fn) -> ?;

fn get ball(_: World) -> ?;

fn get ball_radius(_: FieldGeometry) -> float;

fn get boundary_width(_: FieldGeometry) -> float;

fn get center_circle_radius(_: FieldGeometry) -> float;

fn get field_geom(_: World) -> ?;

fn get field_length(_: FieldGeometry) -> float;

fn get field_width(_: FieldGeometry) -> float;

fn get game_state(_: GameStateData) -> GameState;

fn get game_state(_: World) -> GameStateData;

fn get goal_depth(_: FieldGeometry) -> float;

fn get goal_line_to_penalty_mark(_: FieldGeometry) -> float;

fn get goal_width(_: FieldGeometry) -> float;

fn get heading(_: PlayerData) -> float;

fn get id(_: PlayerData) -> PlayerId;

fn get opp_players(_: World) -> PlayerData>;

fn get own_players(_: World) -> PlayerData>;

fn get penalty_area_depth(_: FieldGeometry) -> float;

fn get penalty_area_width(_: FieldGeometry) -> float;

fn get player(_: RobotSituation) -> PlayerData;

fn get player_id(_: RobotSituation) -> PlayerId;

fn get position(_: BallData) -> Vec2;

fn get position(_: PlayerData) -> Vec2;

fn get position3(_: BallData) -> Vec3;

fn get us_operating(_: GameStateData) -> bool;

fn get velocity(_: BallData) -> Vec3;

fn get velocity(_: PlayerData) -> Vec2;

fn get world(_: RobotSituation) -> World;

fn get x(_: Vec2) -> float;

fn get x(_: Vec3) -> float;

fn get y(_: Vec2) -> float;

fn get y(_: Vec3) -> float;

fn get z(_: Vec3) -> float;

fn get_corner_positions(_: RobotSituation) -> array;

fn get_field_bounds(_: RobotSituation) -> map;

fn get_field_center(_: RobotSituation) -> Vec2;

fn get_goal_corners(_: RobotSituation, _: bool) -> array;

fn get_opp_goal_corners(_: RobotSituation) -> array;

fn get_opp_goal_position(_: RobotSituation) -> Vec2;

fn get_opp_penalty_mark(_: RobotSituation) -> Vec2;

fn get_opp_players_within_radius(_: RobotSituation, _: ArrayStorage<float, 2, 1>>, _: float) -> PlayerData>;

fn get_own_goal_corners(_: RobotSituation) -> array;

fn get_own_goal_position(_: RobotSituation) -> Vec2;

fn get_own_penalty_mark(_: RobotSituation) -> Vec2;

fn get_own_players_within_radius(_: RobotSituation, _: ArrayStorage<float, 2, 1>>, _: float) -> PlayerData>;

fn get_players_within_radius(_: RobotSituation, _: ArrayStorage<float, 2, 1>>, _: float) -> PlayerData>;

fn halfway_to(_: Vec2, _: Vec2) -> Vec2;

fn has_ball(_: RobotSituation) -> bool;

fn interpolate(_: Vec2, _: Vec2, _: float) -> Vec2;

fn is_in_attacking_half(_: RobotSituation, _: ArrayStorage<float, 2, 1>>) -> bool;

fn is_in_center_circle(_: RobotSituation, _: ArrayStorage<float, 2, 1>>) -> bool;

fn is_in_defensive_half(_: RobotSituation, _: ArrayStorage<float, 2, 1>>) -> bool;

fn is_in_opp_penalty_area(_: RobotSituation, _: ArrayStorage<float, 2, 1>>) -> bool;

fn is_in_own_penalty_area(_: RobotSituation, _: ArrayStorage<float, 2, 1>>) -> bool;

fn is_in_penalty_area(_: RobotSituation, _: ArrayStorage<float, 2, 1>>) -> bool;

fn is_position_in_field(_: RobotSituation, _: ArrayStorage<float, 2, 1>>) -> bool;

fn norm(_: Vec2) -> float;

fn norm(_: Vec3) -> float;

fn predict_ball_collision_time(_: RobotSituation) -> float;

fn predict_ball_position(_: RobotSituation, _: float) -> ?;

fn rotate(_: Vec2, _: float) -> Vec2;

fn unit(_: Vec2) -> Vec2;

fn unit(_: Vec3) -> Vec3;

fn xy(_: Vec3) -> Vec2;

fn ApproachBall() -> Result<BehaviorNode, Box<EvalAltResult>>;

fn ApproachBall(description: string) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn AssignRoles(roles_array: array) -> Result<RoleAssignmentProblem, Box<EvalAltResult>>;

fn FaceAngle(angle_rad: ?) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn FaceAngle(angle_rad: ?, description: string) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn FaceAngle(angle_rad: ?, options: map) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn FaceAngle(angle_rad: ?, options: map, description: string) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn FaceTowardsOwnPlayer(player_id: ?) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn FaceTowardsOwnPlayer(player_id: ?, description: string) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn FaceTowardsOwnPlayer(player_id: ?, options: map) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn FaceTowardsOwnPlayer(player_id: ?, options: map, description: string) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn FaceTowardsPosition(target: ?) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn FaceTowardsPosition(target: ?, description: string) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn FaceTowardsPosition(target: ?, options: map) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn FaceTowardsPosition(target: ?, options: map, description: string) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn FetchBall() -> Result<BehaviorNode, Box<EvalAltResult>>;

fn FetchBall(description: string) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn FetchBallWithHeadingAngle(angle_rad: ?) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn FetchBallWithHeadingAngle(angle_rad: ?, description: string) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn FetchBallWithHeadingPlayer(player_id: ?) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn FetchBallWithHeadingPlayer(player_id: ?, description: string) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn FetchBallWithHeadingPosition(target: ?) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn FetchBallWithHeadingPosition(target: ?, description: string) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn GoToPosition(target: ?) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn GoToPosition(target: ?, description: string) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn GoToPosition(target: ?, options: map) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn GoToPosition(target: ?, options: map, description: string) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn Guard(condition_fn_ptr: Fn, child_node: BehaviorNode, cond_description: string) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn InterceptBall() -> Result<BehaviorNode, Box<EvalAltResult>>;

fn InterceptBall(description: string) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn Kick() -> Result<BehaviorNode, Box<EvalAltResult>>;

fn Kick(description: string) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn Role(name: string) -> RoleBuilder;

fn ScoringSelect(children_scorers_dyn: array, hysteresis_margin: float) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn ScoringSelect(children_scorers_dyn: array, hysteresis_margin: float, description: string) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn Select(children_dyn: array) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn Select(children_dyn: array, description: string) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn Semaphore(child_node: BehaviorNode, id: string, max_count: int) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn Semaphore(child_node: BehaviorNode, id: string, max_count: int, description: string) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn Sequence(children_dyn: array) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn Sequence(children_dyn: array, description: string) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn Wait(duration_secs: ?) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn Wait(duration_secs: ?, description: string) -> Result<BehaviorNode, Box<EvalAltResult>>;

fn behavior(builder: RoleBuilder, builder_fn: Fn) -> Result<RoleBuilder, Box<EvalAltResult>>;

fn build(builder: RoleBuilder) -> Result<Role, Box<EvalAltResult>>;

fn count(builder: RoleBuilder, count: int) -> RoleBuilder;

fn exclude(builder: RoleBuilder, filter_fn: Fn) -> Result<RoleBuilder, Box<EvalAltResult>>;

/// Returns a float between 0 and 1 based on the player id.
/// This is used to ensure that the same player id always produces the same hash.
/// This can be used to induce different behavior for different players.
fn hash_float(id: PlayerId) -> float;

fn max(builder: RoleBuilder, count: int) -> RoleBuilder;

fn min(builder: RoleBuilder, count: int) -> RoleBuilder;

fn require(builder: RoleBuilder, filter_fn: Fn) -> Result<RoleBuilder, Box<EvalAltResult>>;

fn score(builder: RoleBuilder, scorer_fn: Fn) -> Result<RoleBuilder, Box<EvalAltResult>>;

/// Returns the string representation of the player id.
fn to_string(id: PlayerId) -> String;

fn vec2(x: float, y: float) -> Vec2;