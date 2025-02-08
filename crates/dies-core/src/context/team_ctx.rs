use std::sync::Arc;

use crate::{PlayerFrame, TeamColor, Vector2, WorldFrame};

use super::WorldView;

pub struct TeamCtx {
    world: Arc<WorldFrame>,
    color: TeamColor,
}

impl TeamCtx {
    pub fn new(world: Arc<WorldFrame>, color: TeamColor) -> Self {
        Self { world, color }
    }
}

impl WorldView for TeamCtx {
    fn world_frame(&self) -> &WorldFrame {
        &self.world
    }
}

impl TeamView for TeamCtx {
    fn own_players(&self) -> &Vec<PlayerFrame> {
        &self.world.get_team(self.color)
    }

    fn opp_players(&self) -> &Vec<PlayerFrame> {
        &self.world.get_team(self.color.opposite())
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
