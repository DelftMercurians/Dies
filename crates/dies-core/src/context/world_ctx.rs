use crate::{PlayerFrame, TeamColor, Vector2, WorldFrame};

pub trait WorldView {
    /// Get a reference to the current world frame.
    fn world_frame(&self) -> &WorldFrame;

    /// Get the nearest player to the given position with the given color.
    fn nearest_player_with_color(&self, pos: Vector2, color: TeamColor) -> Option<&PlayerFrame> {
        let players = self.world_frame().get_team(color);
        players
            .iter()
            .min_by_key(|p| p.position.metric_distance(&pos) as i64)
    }

    /// Get the nearest player to the given position, and the color of the team they belong to.
    fn nearest_player(&self, pos: Vector2) -> Option<(TeamColor, &PlayerFrame)> {
        let blue = self
            .world_frame()
            .blue_team
            .iter()
            .map(|p| (p, p.position.metric_distance(&pos) as i64))
            .min_by_key(|(_, dist)| *dist);

        let yellow = self
            .world_frame()
            .yellow_team
            .iter()
            .map(|p| (p, p.position.metric_distance(&pos) as i64))
            .min_by_key(|(_, dist)| *dist);

        match (blue, yellow) {
            (Some((blue, blue_dist)), Some((yellow, yellow_dist))) => {
                if blue_dist < yellow_dist {
                    Some((TeamColor::Blue, blue))
                } else {
                    Some((TeamColor::Yellow, yellow))
                }
            }
            (Some((blue, _)), None) => Some((TeamColor::Blue, blue)),
            (None, Some((yellow, _))) => Some((TeamColor::Yellow, yellow)),
            (None, None) => None,
        }
    }
}

impl WorldView for WorldFrame {
    fn world_frame(&self) -> &WorldFrame {
        self
    }
}
