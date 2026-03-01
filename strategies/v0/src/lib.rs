use dies_strategy_api::prelude::*;

pub struct V0Strategy;

impl V0Strategy {
    pub fn new() -> Self {
        Self
    }
}

impl Default for V0Strategy {
    fn default() -> Self {
        Self::new()
    }
}

impl Strategy for V0Strategy {
    fn update(&mut self, ctx: &mut TeamContext) {
        // TODO: Port v0 roles to IPC API
        //
        // Original roles (see reference/ directory for BT implementations):
        //   - goalkeeper        (reference/v0/keeper.rs)
        //   - striker_1/2/3     (reference/v0/striker.rs)
        //   - harasser_1        (reference/v0/harasser.rs)
        //   - secondary_harasser (reference/v0/secondary_harasser.rs)
        //   - waller_1/2        (reference/v0/waller.rs)
        //   - penalty_kicker    (reference/v0/penalty_kicker.rs)
        //   - kickoff_kicker    (reference/v0/kickoff_kicker.rs)
        //   - freekick_kicker   (reference/v0/freekick_kicker.rs)
        //   - freekick_interference (reference/v0/freekick_interference.rs)
        //
        // Supporting code:
        //   - reference/behavior_tree/  — BT runtime (nodes, role assignment, rhai host)
        //   - reference/legacy_skills/  — legacy skill implementations (fetchball, shoot, kick, etc.)
        //   - reference/testing/        — test strategies
        //   - reference/v0/utils/       — utility functions (goal_shot, ball_calc, pressure, etc.)
        //
        // For now, just stop all robots.
        for player in ctx.players() {
            player.stop();
        }
    }
}
