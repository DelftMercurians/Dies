use anyhow::Result;
use std::{
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};

use dies_core::{
    Angle, BallData, FieldCircularArc, FieldGeometry, FieldLineSegment, GameState, GameStateData,
    PlayerData, PlayerId, TeamColor, TeamData, Vector2, Vector3,
};
use rhai::{
    exported_module, module_resolvers::FileModuleResolver, Dynamic, Engine, ModuleResolver,
    OptimizationLevel, Scope, AST,
};

use crate::behavior_tree::{bt_rhai_plugin, BehaviorTree, RobotSituation};
use crate::ScriptError;

use super::rhai_types::RhaiBehaviorNode;

const PLAY_ENTRY_POINT: &str = "build_play_bt";
const KICKOFF_ENTRY_POINT: &str = "build_kickoff_bt";
const PENALTY_ENTRY_POINT: &str = "build_penalty_bt";

#[derive(Clone)]

struct SharedModuleResolver {
    resolver: Arc<RwLock<FileModuleResolver>>,
}

impl SharedModuleResolver {
    pub fn new(resolver: FileModuleResolver) -> Self {
        Self {
            resolver: Arc::new(RwLock::new(resolver)),
        }
    }

    pub fn set_scope(&self, scope: Scope<'static>) {
        let mut resolver = self.resolver.write().unwrap();
        resolver.set_scope(scope);
    }
}

impl ModuleResolver for SharedModuleResolver {
    fn resolve(
        &self,
        engine: &Engine,
        source: Option<&str>,
        path: &str,
        pos: rhai::Position,
    ) -> Result<rhai::Shared<rhai::Module>, Box<rhai::EvalAltResult>> {
        let resolver = self.resolver.read().unwrap();
        resolver.resolve(engine, source, path, pos)
    }
}

pub struct RhaiHost {
    engine: Arc<RwLock<Engine>>,
    resolver: SharedModuleResolver,
    script_path: String,
}

impl RhaiHost {
    pub fn new(script_path: impl AsRef<Path>) -> Self {
        let (engine, resolver) = create_engine();
        let path_str = script_path.as_ref().to_string_lossy().to_string();

        Self {
            engine: Arc::new(RwLock::new(engine)),
            resolver,
            script_path: path_str,
        }
    }

    pub fn engine(&self) -> Arc<RwLock<Engine>> {
        self.engine.clone()
    }

    /// Build behavior tree based on game state
    pub fn build_tree_for_state(
        &self,
        player_id: PlayerId,
        game_state: GameState,
        team_color: TeamColor,
        field_geom: FieldGeometry,
    ) -> Result<BehaviorTree, ScriptError> {
        let mut scope = Scope::new();
        scope.push_constant("FIELD_LENGTH", field_geom.field_length);
        scope.push_constant("FIELD_HALF_LENGTH", field_geom.field_length / 2.0);
        scope.push_constant("FIELD_WIDTH", field_geom.field_width);
        scope.push_constant("FIELD_HALF_WIDTH", field_geom.field_width / 2.0);
        scope.push_constant("GOAL_WIDTH", field_geom.goal_width);
        scope.push_constant("GOAL_HALF_WIDTH", field_geom.goal_width / 2.0);
        scope.push_constant("GOAL_DEPTH", field_geom.goal_depth);
        scope.push_constant("FIELD", field_geom);
        self.resolver.set_scope(scope.clone());

        let ast = match self
            .engine
            .read()
            .unwrap()
            .compile_file_with_scope(&scope, PathBuf::from(&self.script_path))
        {
            Ok(ast) => ast,
            Err(err) => {
                let script_error = ScriptError::Syntax {
                    script_path: self.script_path.clone(),
                    message: format!("{}", err),
                    line: None,
                    column: None,
                };
                log::error!(
                    "Script compilation failed for {}: {}",
                    self.script_path,
                    err
                );
                return Err(script_error);
            }
        };

        let entry_point = match game_state {
            GameState::Kickoff | GameState::PrepareKickoff => KICKOFF_ENTRY_POINT,
            GameState::Penalty | GameState::PreparePenalty => PENALTY_ENTRY_POINT,
            _ => PLAY_ENTRY_POINT, // Default to play tree for all other states
        };

        let result = self.engine.read().unwrap().call_fn::<RhaiBehaviorNode>(
            &mut scope,
            &ast,
            entry_point,
            (player_id,),
        );

        match result {
            Ok(node) => Ok(BehaviorTree::new(node.0)),
            Err(e) => {
                let script_error = ScriptError::Runtime {
                    script_path: self.script_path.clone(),
                    function_name: entry_point.to_string(),
                    message: format!("{}", e),
                    team_color,
                    player_id: Some(player_id),
                };
                log::error!(
                    "Script runtime error for {} in {}: {}",
                    entry_point,
                    self.script_path,
                    e
                );
                Err(script_error)
            }
        }
    }
}

fn create_engine() -> (Engine, SharedModuleResolver) {
    let mut engine = Engine::new();
    engine.set_max_expr_depths(64, 64);

    // Set up module resolver to support imports
    let mut file_resolver = FileModuleResolver::new_with_path(Path::new("strategies"));
    file_resolver.enable_cache(true);

    let resolver = SharedModuleResolver::new(file_resolver);
    engine.set_module_resolver(resolver.clone());
    engine.set_optimization_level(OptimizationLevel::Simple);

    engine.on_print(|text| log::info!("[RHAI SCRIPT] {}", text));

    engine.on_debug(|text, source, pos| {
        let src_info = source.map_or_else(String::new, |s| format!(" in '{}'", s));
        let pos_info = if pos.is_none() {
            String::new()
        } else {
            format!(" @ {}", pos)
        };
        log::debug!("[RHAI SCRIPT DEBUG]{}{}: {}", src_info, pos_info, text);
    });

    engine
        .register_type_with_name::<RobotSituation>("RobotSituation")
        .register_fn("has_ball", |rs: &mut RobotSituation| rs.has_ball())
        .register_fn("player", |rs: &mut RobotSituation| rs.player_data().clone())
        .register_get("world", |rs: &mut RobotSituation| rs.world.clone())
        .register_get("player_id", |rs: &mut RobotSituation| rs.player_id);

    engine
        .register_type_with_name::<Arc<TeamData>>("World")
        .register_get("ball", |wd: &mut Arc<TeamData>| {
            if let Some(ball) = &wd.ball {
                Dynamic::from(ball.clone())
            } else {
                Dynamic::from(())
            }
        })
        .register_get("own_players", |wd: &mut Arc<TeamData>| {
            wd.own_players.clone()
        })
        .register_get("opp_players", |wd: &mut Arc<TeamData>| {
            wd.opp_players.clone()
        })
        .register_get("game_state", |wd: &mut Arc<TeamData>| {
            wd.current_game_state.clone()
        })
        .register_get("field_geom", |wd: &mut Arc<TeamData>| {
            if let Some(field_geom) = &wd.field_geom {
                Dynamic::from(field_geom.clone())
            } else {
                Dynamic::from(())
            }
        });

    engine
        .register_type_with_name::<PlayerData>("PlayerData")
        .register_get("id", |pd: &mut PlayerData| pd.id)
        .register_get("position", |pd: &mut PlayerData| pd.position)
        .register_get("velocity", |pd: &mut PlayerData| pd.velocity)
        .register_get("heading", |pd: &mut PlayerData| pd.yaw.radians());

    engine
        .register_type_with_name::<BallData>("BallData")
        .register_get("position", |bd: &mut BallData| bd.position)
        .register_get("velocity", |bd: &mut BallData| bd.velocity);

    engine
        .register_type_with_name::<Vector2>("Vec2")
        .register_get("x", |v: &mut Vector2| v.x)
        .register_get("y", |v: &mut Vector2| v.y);

    engine
        .register_type_with_name::<Vector3>("Vec3")
        .register_get("x", |v: &mut Vector3| v.x)
        .register_get("y", |v: &mut Vector3| v.y)
        .register_get("z", |v: &mut Vector3| v.z);

    // Register GameStateData
    engine
        .register_type_with_name::<GameStateData>("GameStateData")
        .register_get("game_state", |gsd: &mut GameStateData| gsd.game_state)
        .register_get("us_operating", |gsd: &mut GameStateData| gsd.us_operating);

    // Register GameState enum
    engine.register_type_with_name::<GameState>("GameState");

    // Register FieldGeometry and related types
    engine
        .register_type_with_name::<FieldGeometry>("FieldGeometry")
        .register_get("field_length", |fg: &mut FieldGeometry| fg.field_length)
        .register_get("field_width", |fg: &mut FieldGeometry| fg.field_width)
        .register_get("goal_width", |fg: &mut FieldGeometry| fg.goal_width)
        .register_get("goal_depth", |fg: &mut FieldGeometry| fg.goal_depth)
        .register_get("boundary_width", |fg: &mut FieldGeometry| fg.boundary_width)
        .register_get("penalty_area_depth", |fg: &mut FieldGeometry| {
            fg.penalty_area_depth
        })
        .register_get("penalty_area_width", |fg: &mut FieldGeometry| {
            fg.penalty_area_width
        })
        .register_get("center_circle_radius", |fg: &mut FieldGeometry| {
            fg.center_circle_radius
        })
        .register_get("goal_line_to_penalty_mark", |fg: &mut FieldGeometry| {
            fg.goal_line_to_penalty_mark
        })
        .register_get("ball_radius", |fg: &mut FieldGeometry| fg.ball_radius);

    let bt_module = Arc::new(exported_module!(bt_rhai_plugin));
    engine.register_global_module(bt_module);

    (engine, resolver)
}
