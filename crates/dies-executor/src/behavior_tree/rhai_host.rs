use anyhow::Result;
use std::{
    path::Path,
    sync::{Arc, RwLock},
};

use dies_core::{
    Angle, BallData, GameState, GameStateData, PlayerData, PlayerId, TeamColor, TeamData, Vector2,
    Vector3,
};
use rhai::{exported_module, module_resolvers::FileModuleResolver, Dynamic, Engine, Scope, AST};

use crate::behavior_tree::{bt_rhai_plugin, BehaviorTree, RobotSituation};
use crate::ScriptError;

use super::rhai_types::RhaiBehaviorNode;

const PLAY_ENTRY_POINT: &str = "build_play_bt";
const KICKOFF_ENTRY_POINT: &str = "build_kickoff_bt";
const PENALTY_ENTRY_POINT: &str = "build_penalty_bt";

pub struct RhaiHost {
    engine: Arc<RwLock<Engine>>,
    ast: Arc<RwLock<Option<AST>>>,
    script_path: String,
    compilation_error: Arc<RwLock<Option<ScriptError>>>,
}

impl RhaiHost {
    pub fn new(script_path: impl AsRef<Path>) -> Self {
        let engine = create_engine();
        let path_str = script_path.as_ref().to_string_lossy().to_string();

        let (ast, compilation_error) = match engine.compile_file(script_path.as_ref().into()) {
            Ok(ast) => (Some(ast), None),
            Err(err) => {
                let script_error = ScriptError::Syntax {
                    script_path: path_str.clone(),
                    message: format!("{}", err),
                    line: None, // Rhai doesn't provide line info in compile errors easily
                    column: None,
                };
                log::error!("Script compilation failed for {}: {}", path_str, err);
                (None, Some(script_error))
            }
        };

        Self {
            engine: Arc::new(RwLock::new(engine)),
            ast: Arc::new(RwLock::new(ast)),
            script_path: path_str,
            compilation_error: Arc::new(RwLock::new(compilation_error)),
        }
    }

    pub fn engine(&self) -> Arc<RwLock<Engine>> {
        self.engine.clone()
    }

    /// Get the compilation error if any
    pub fn compilation_error(&self) -> Option<ScriptError> {
        self.compilation_error.read().unwrap().clone()
    }

    /// Check if the script compiled successfully
    pub fn is_compiled(&self) -> bool {
        self.ast.read().unwrap().is_some()
    }

    /// Build behavior tree based on game state
    pub fn build_tree_for_state(
        &self,
        player_id: PlayerId,
        game_state: GameState,
        team_color: TeamColor,
    ) -> Result<BehaviorTree, ScriptError> {
        // Check if script is compiled
        let ast_guard = self.ast.read().unwrap();
        let ast = match ast_guard.as_ref() {
            Some(ast) => ast,
            None => {
                // Return compilation error if script didn't compile
                return Err(self
                    .compilation_error()
                    .unwrap_or_else(|| ScriptError::Syntax {
                        script_path: self.script_path.clone(),
                        message: "Script compilation failed".to_string(),
                        line: None,
                        column: None,
                    }));
            }
        };

        let entry_point = match game_state {
            GameState::Kickoff | GameState::PrepareKickoff => KICKOFF_ENTRY_POINT,
            GameState::Penalty | GameState::PreparePenalty => PENALTY_ENTRY_POINT,
            _ => PLAY_ENTRY_POINT, // Default to play tree for all other states
        };

        let mut scope = Scope::new();
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

fn create_engine() -> Engine {
    let mut engine = Engine::new();
    engine.set_max_expr_depths(64, 64);

    // Set up module resolver to support imports
    let mut resolver = FileModuleResolver::new_with_path(Path::new("strategies"));
    resolver.enable_cache(true);
    engine.set_module_resolver(resolver);

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

    let bt_module = Arc::new(exported_module!(bt_rhai_plugin));
    engine.register_global_module(bt_module);

    engine
}
