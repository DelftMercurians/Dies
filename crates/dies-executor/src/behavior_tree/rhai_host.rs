use anyhow::Result;
use std::{
    path::Path,
    sync::{Arc, RwLock},
};

use dies_core::{Angle, BallData, PlayerData, PlayerId, Vector2, Vector3, TeamData};
use rhai::{exported_module, Dynamic, Engine, Scope, AST};

use crate::behavior_tree::{bt_rhai_plugin, BehaviorTree, RobotSituation};

use super::rhai_types::RhaiBehaviorNode;

const ENTRY_POINT_NAME: &str = "build_player_bt";

pub struct RhaiHost {
    engine: Arc<RwLock<Engine>>,
    ast: Arc<RwLock<AST>>,
}

impl RhaiHost {
    pub fn new(script_path: impl AsRef<Path>) -> Self {
        let engine = create_engine();
        let ast = engine.compile_file(script_path.as_ref().into()).unwrap();

        Self {
            engine: Arc::new(RwLock::new(engine)),
            ast: Arc::new(RwLock::new(ast)),
        }
    }

    pub fn engine(&self) -> Arc<RwLock<Engine>> {
        self.engine.clone()
    }

    pub fn build_player_bt(&self, player_id: PlayerId) -> Result<BehaviorTree> {
        let mut scope = Scope::new();
        let ast = self.ast.read().unwrap();
        let result = self.engine.read().unwrap().call_fn::<RhaiBehaviorNode>(
            &mut scope,
            &ast,
            ENTRY_POINT_NAME,
            (player_id,),
        );
        result
            .map(|node| BehaviorTree::new(node.0))
            .map_err(|e| anyhow::anyhow!("Failed to build player BT: {:?}", e))
    }
}

fn create_engine() -> Engine {
    let mut engine = Engine::new();
    engine.set_max_expr_depths(64, 64);

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

    let bt_module = Arc::new(exported_module!(bt_rhai_plugin));
    engine.register_global_module(bt_module);

    engine
}
