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

use crate::{
    behavior_tree::{
        role_assignment::{Role, RoleAssignmentProblem},
        BehaviorTree, RobotSituation, RoleBuilder,
    },
    ScriptError,
};

use super::{rhai_type_registration, rhai_types::RhaiBehaviorNode};

const SCRIPT_ENTRY_POINT: &str = "main";

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
}

impl RhaiHost {
    pub fn new(script_path: impl AsRef<Path>) -> Self {
        let (engine, resolver) = create_engine();
        let path_str = script_path.as_ref().to_string_lossy().to_string();

        Self {
            engine: Arc::new(RwLock::new(engine)),
            resolver,
            script_path: path_str,
            team_color: TeamColor::Yellow, // Default, should be set later
        }
    }

    pub fn set_team_color(&mut self, team_color: TeamColor) {
        self.team_color = team_color;
    }

    pub fn engine(&self) -> Arc<RwLock<Engine>> {
        self.engine.clone()
    }

    /// Get role assignment configuration from script
    pub fn get_role_assignment(
        &self,
        team_data: &TeamData,
        game_state: GameState,
    ) -> Result<RoleAssignmentProblem, ScriptError> {
        let mut scope = self.create_scope(&team_data.field_geom);

        // Add game state to scope
        scope.push_constant("GAME_STATE", game_state);
        scope.push_constant("TEAM_DATA", team_data.clone());

        self.resolver.set_scope(scope.clone());

        let ast = match self
            .engine
            .read()
            .unwrap()
            .compile_file_with_scope(&scope, PathBuf::from(&self.script_path))
        {
            Ok(ast) => ast,
            Err(err) => {
                return Err(ScriptError::Syntax {
                    script_path: self.script_path.clone(),
                    message: format!("{}", err),
                    line: None,
                    column: None,
                });
            }
        };

        let result = self
            .engine
            .read()
            .unwrap()
            .call_fn::<RoleAssignmentProblem>(&mut scope, &ast, SCRIPT_ENTRY_POINT, ());

        match result {
            Ok(problem) => Ok(problem),
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
        let mut scope = self.create_scope(&team_data.field_geom);

        let ast = self.compile_script(&scope)?;

        // Call role-specific builder function
        let builder_name = format!("build_{}_tree", role_name);
        let result = self.engine.read().unwrap().call_fn::<RhaiBehaviorNode>(
            &mut scope,
            &ast,
            &builder_name,
            (player_id,),
        );

        match result {
            Ok(node) => Ok(BehaviorTree::new(node.0)),
            Err(e) => {
                // Fallback to generic builder
                let fallback_result = self.engine.read().unwrap().call_fn::<RhaiBehaviorNode>(
                    &mut scope,
                    &ast,
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

    fn compile_script(&self, scope: &Scope) -> Result<AST, ScriptError> {
        self.engine
            .read()
            .unwrap()
            .compile_file_with_scope(scope, PathBuf::from(&self.script_path))
            .map_err(|err| ScriptError::Syntax {
                script_path: self.script_path.clone(),
                message: format!("{}", err),
                line: None,
                column: None,
            })
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
