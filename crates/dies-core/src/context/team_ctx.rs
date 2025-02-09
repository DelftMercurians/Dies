use crate::{PlayerFrame, TeamColor, Vector2, WorldFrame};

use super::{
    base::{BaseContext, BaseCtx},
    WorldView,
};

#[derive(Clone)]
pub struct TeamCtx {
    base: BaseCtx,
    color: TeamColor,
}

impl TeamCtx {
    pub fn new(base: BaseCtx, color: TeamColor) -> Self {
        Self { base, color }
    }
}

impl WorldView for TeamCtx {
    fn world_frame(&self) -> &WorldFrame {
        &self.base.world_frame()
    }
}

impl BaseContext for TeamCtx {
    fn settings_manager(&self) -> &crate::SettingsHandle {
        self.base.settings_manager()
    }

    fn dbg_send(&self, key: &str, value: Option<crate::debug::DebugValue>) {
        self.base.dbg_send(key, value)
    }

    fn dbg_scoped(&self, subkey: &str) -> Self {
        Self {
            base: self.base.dbg_scoped(subkey),
            color: self.color,
        }
    }
}

impl TeamView for TeamCtx {
    fn own_players(&self) -> &Vec<PlayerFrame> {
        &self.world_frame().get_team(self.color)
    }

    fn opp_players(&self) -> &Vec<PlayerFrame> {
        &self.world_frame().get_team(self.color.opposite())
    }
}

pub trait TeamView: WorldView {
    fn own_players(&self) -> &Vec<PlayerFrame>;
    fn opp_players(&self) -> &Vec<PlayerFrame>;

    fn get_nearest_own_player(&self, pos: Vector2) -> Option<&PlayerFrame> {
        self.own_players()
            .iter()
            .min_by_key(|p| p.position.metric_distance(&pos) as i64)
    }

    fn get_nearest_opp_player(&self, pos: Vector2) -> Option<&PlayerFrame> {
        self.opp_players()
            .iter()
            .min_by_key(|p| p.position.metric_distance(&pos) as i64)
    }
}
