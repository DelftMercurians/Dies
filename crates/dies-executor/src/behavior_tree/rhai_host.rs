use anyhow::Result;
use std::{
    path::Path,
    sync::{Arc, RwLock},
};

use dies_core::PlayerId;
use rhai::{exported_module, Engine, Scope, AST};

use crate::behavior_tree::{bt_rhai_plugin, BehaviorTree};

use super::rhai_plugin::RhaiBehaviorNode;

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

    let bt_module = Arc::new(exported_module!(bt_rhai_plugin));
    engine.register_global_module(bt_module);

    engine
}
