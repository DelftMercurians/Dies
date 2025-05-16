use rhai::{Engine, Scope};

pub struct ScriptHost {
    engine: Engine,
    scope: Scope<'static>,
}

impl ScriptHost {
    pub fn new() -> Self {
        let engine = Engine::new_raw();
        let scope = Scope::new();
        Self { engine, scope }
    }
}
