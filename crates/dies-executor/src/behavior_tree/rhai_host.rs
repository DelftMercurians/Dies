use anyhow::Result;
use std::{
    cell::RefCell,
    path::{Path, PathBuf},
    rc::Rc,
    sync::{Arc, Mutex, RwLock},
};

use dies_core::{
    Angle, BallData, FieldCircularArc, FieldGeometry, FieldLineSegment, GameState, GameStateData,
    PlayerData, PlayerId, TeamColor, TeamData, Vector2, Vector3,
};
use rhai::{
    exported_module, module_resolvers::FileModuleResolver, Dynamic, Engine, ModuleResolver,
    OptimizationLevel, Scope, AST,
};

use crate::{
    behavior_tree::{
        role_assignment::{Role, RoleAssignmentProblem, RoleBuilder},
        BehaviorTree, RobotSituation,
    },
    ScriptError,
};

use super::{rhai_type_registration, rhai_types::RhaiBehaviorNode};

const SCRIPT_ENTRY_POINT: &str = "main";

/// Game context passed to strategy scripts
#[derive(Clone)]
pub struct GameContext {
    pub game_state: GameState,
    pub num_own_players: usize,
    pub num_opp_players: usize,
    pub field_geom: Option<FieldGeometry>,
    role_builders: Arc<Mutex<Vec<RoleBuilder>>>,
}

impl GameContext {
    pub fn new(team_data: &TeamData) -> Self {
        Self {
            game_state: team_data.current_game_state.game_state,
            num_own_players: team_data.own_players.len(),
            num_opp_players: team_data.opp_players.len(),
            field_geom: team_data.field_geom.clone(),
            role_builders: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Add a role and return a builder for configuration
    pub fn add_role(&self, name: &str) -> RoleBuilder {
        RoleBuilder::new(name)
    }

    /// Extract the final role assignment problem
    pub fn into_role_assignment_problem(self) -> RoleAssignmentProblem {
        let mut role_builders = self.role_builders.lock().unwrap();
        let roles = std::mem::take(&mut *role_builders)
            .into_iter()
            .filter_map(|builder| builder.build().ok())
            .collect();

        RoleAssignmentProblem { roles }
    }
}

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
    pub(crate) script_path: String,
    team_color: TeamColor,
    compiled_ast: Option<AST>,
}

impl RhaiHost {
    pub fn new(script_path: impl AsRef<Path>) -> Self {
        let (engine, resolver) = create_engine();
        let path_str = script_path.as_ref().to_string_lossy().to_string();

        let mut host = Self {
            engine: Arc::new(RwLock::new(engine)),
            resolver,
            script_path: path_str,
            team_color: TeamColor::Yellow, // Default, should be set later
            compiled_ast: None,
        };

        // Compile the script immediately
        if let Err(e) = host.compile() {
            log::error!("Failed to compile script on creation: {:?}", e);
        }

        host
    }

    pub fn set_team_color(&mut self, team_color: TeamColor) {
        self.team_color = team_color;
    }

    pub fn engine(&self) -> Arc<RwLock<Engine>> {
        self.engine.clone()
    }

    /// Compile the script once and store the AST
    pub fn compile(&mut self) -> Result<(), ScriptError> {
        let scope = self.create_scope(&None); // Create scope without field geometry for compilation
        self.resolver.set_scope(scope.clone());

        let ast = self
            .engine
            .read()
            .unwrap()
            .compile_file_with_scope(&scope, PathBuf::from(&self.script_path))
            .map_err(|err| ScriptError::Syntax {
                script_path: self.script_path.clone(),
                message: format!("{}", err),
                line: None,
                column: None,
            })?;

        self.compiled_ast = Some(ast);
        Ok(())
    }

    /// Get role assignment configuration from script
    pub fn get_role_assignment(
        &self,
        team_data: &TeamData,
    ) -> Result<RoleAssignmentProblem, ScriptError> {
        let ast = self
            .compiled_ast
            .as_ref()
            .ok_or_else(|| ScriptError::Runtime {
                script_path: self.script_path.clone(),
                function_name: "get_role_assignment".to_string(),
                message: "Script not compiled. Call compile() first.".to_string(),
                team_color: self.team_color,
                player_id: None,
            })?;

        let mut scope = self.create_scope(&team_data.field_geom);

        // Create game context
        let game_context = GameContext::new(team_data);

        // Add game context and team data to scope
        scope.push_constant("GAME_CONTEXT", game_context.clone());
        scope.push_constant("TEAM_DATA", team_data.clone());

        self.resolver.set_scope(scope.clone());

        let engine = self.engine.read().unwrap();

        // Call main function without expecting a return value
        let result =
            engine.call_fn::<Dynamic>(&mut scope, ast, SCRIPT_ENTRY_POINT, (game_context.clone(),));

        match result {
            Ok(_) => {
                // Extract the role assignment problem from the game context
                Ok(game_context.into_role_assignment_problem())
            }
            Err(e) => Err(ScriptError::Runtime {
                script_path: self.script_path.clone(),
                function_name: SCRIPT_ENTRY_POINT.to_string(),
                message: format!("{}", e),
                team_color: self.team_color,
                player_id: None,
            }),
        }
    }

    /// Build behavior tree for a specific role
    pub fn build_tree_for_role(
        &self,
        player_id: PlayerId,
        role_name: &str,
        team_data: &TeamData,
    ) -> Result<BehaviorTree, ScriptError> {
        let ast = self
            .compiled_ast
            .as_ref()
            .ok_or_else(|| ScriptError::Runtime {
                script_path: self.script_path.clone(),
                function_name: "build_tree_for_role".to_string(),
                message: "Script not compiled. Call compile() first.".to_string(),
                team_color: self.team_color,
                player_id: Some(player_id),
            })?;

        let mut scope = self.create_scope(&team_data.field_geom);

        // Call role-specific builder function
        let builder_name = format!("build_{}_tree", role_name);
        let result = self.engine.read().unwrap().call_fn::<RhaiBehaviorNode>(
            &mut scope,
            ast,
            &builder_name,
            (player_id,),
        );

        match result {
            Ok(node) => Ok(BehaviorTree::new(node.0)),
            Err(e) => {
                // Fallback to generic builder
                let fallback_result = self.engine.read().unwrap().call_fn::<RhaiBehaviorNode>(
                    &mut scope,
                    ast,
                    "build_default_tree",
                    (player_id,),
                );

                match fallback_result {
                    Ok(node) => Ok(BehaviorTree::new(node.0)),
                    Err(_) => Err(ScriptError::Runtime {
                        script_path: self.script_path.clone(),
                        function_name: builder_name,
                        message: format!("{}", e),
                        team_color: self.team_color,
                        player_id: Some(player_id),
                    }),
                }
            }
        }
    }

    fn create_scope(&self, field_geom: &Option<FieldGeometry>) -> Scope<'static> {
        let mut scope = Scope::new();
        if let Some(geom) = field_geom {
            scope.push_constant("FIELD_LENGTH", geom.field_length);
            scope.push_constant("FIELD_HALF_LENGTH", geom.field_length / 2.0);
            scope.push_constant("FIELD_WIDTH", geom.field_width);
            scope.push_constant("FIELD_HALF_WIDTH", geom.field_width / 2.0);
            scope.push_constant("GOAL_WIDTH", geom.goal_width);
            scope.push_constant("GOAL_HALF_WIDTH", geom.goal_width / 2.0);
            scope.push_constant("GOAL_DEPTH", geom.goal_depth);
            scope.push_constant("FIELD", geom.clone());
        }
        scope
    }
}

fn create_engine() -> (Engine, SharedModuleResolver) {
    let mut engine = Engine::new();
    engine.set_max_expr_depths(64, 64);
    engine.set_fast_operators(false);

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

    // Register all types and their extended APIs
    rhai_type_registration::register_all_types(&mut engine);

    // Disable fast operators to allow operator overloading
    engine.set_fast_operators(false);

    let bt_module = Arc::new(exported_module!(super::rhai_plugin::bt_rhai_plugin));
    engine.register_global_module(bt_module);

    (engine, resolver)
}
