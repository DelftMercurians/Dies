use crate::{PlayerFrame, Vector2};

use super::{WithBall, WorldView};

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

pub trait TeamViewWithBall: TeamView + WithBall {
    fn get_own_player_nearest_to_ball(&self) -> Option<&PlayerFrame> {
        self.get_nearest_own_player(self.ball().position.xy())
    }

    fn get_opp_player_nearest_to_ball(&self) -> Option<&PlayerFrame> {
        self.get_nearest_opp_player(self.ball().position.xy())
    }
}

impl<W: TeamView + WithBall> TeamViewWithBall for W {}
