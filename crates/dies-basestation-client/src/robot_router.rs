use dies_core::{ColoredPlayerId, PlayerId, TeamColor, VecMap};

/// Routes player commands and feedback messages between team color and player IDs and
/// underlying robot IDs.
pub(crate) struct RobotRouter {
    default_team_color: TeamColor,
    id_map: VecMap<u32, Option<ColoredPlayerId>>,
}

impl RobotRouter {
    pub(crate) fn new(default_team_color: TeamColor) -> Self {
        Self {
            default_team_color,
            id_map: VecMap::new(),
        }
    }

    pub(crate) fn set_id_map(&mut self, id_map: Vec<(u32, Option<ColoredPlayerId>)>) {
        for (rid, player_id) in id_map {
            self.id_map.insert(rid, player_id);
        }
    }

    pub(crate) fn update_robot_ids(&mut self, rids: Vec<u32>) {
        // Keep existing mappings that have Some value
        let mut new_map = VecMap::new();
        for (rid, player_id) in self.id_map.iter() {
            if player_id.is_some() {
                new_map.insert(*rid, *player_id);
            }
        }

        // Add all robot IDs from rids, preserving existing mappings
        for &rid in rids.iter() {
            if !new_map.contains_key(&rid) {
                new_map.insert(rid, None);
            }
        }

        self.id_map = new_map;
    }

    pub(crate) fn id_map(&self) -> Vec<(u32, Option<ColoredPlayerId>)> {
        self.id_map.iter().map(|(k, v)| (*k, *v)).collect()
    }

    pub(crate) fn get_by_player_id(&self, (team, id): ColoredPlayerId) -> u32 {
        self.id_map
            .iter()
            .find(|(_, &player_id)| player_id == Some((team, id)))
            .map(|(robot_id, _)| *robot_id)
            .unwrap_or(id.as_u32())
    }

    pub(crate) fn get_by_robot_id(&self, rid: u32) -> ColoredPlayerId {
        self.id_map
            .get(&rid)
            .copied()
            .flatten()
            .unwrap_or((self.default_team_color, PlayerId::new(rid)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_router() -> RobotRouter {
        RobotRouter::new(TeamColor::Yellow)
    }

    #[test]
    fn test_add_robot_id() {
        let mut router = create_test_router();

        // Add a new robot
        router.add_robot_id(1);
        assert_eq!(router.id_map.len(), 1);
        assert_eq!(router.id_map.get(&1), Some(&None));

        // Add same robot again - should not duplicate
        router.add_robot_id(1);
        assert_eq!(router.id_map.len(), 1);
    }

    #[test]
    fn test_set_id_map() {
        let mut router = create_test_router();

        // Initial mapping
        let initial_map = vec![
            (1, Some((TeamColor::Blue, PlayerId::new(1)))),
            (2, Some((TeamColor::Yellow, PlayerId::new(2)))),
        ];
        router.set_id_map(initial_map);
        assert_eq!(router.id_map.len(), 2);

        // Update with new mapping
        let new_map = vec![
            (2, Some((TeamColor::Blue, PlayerId::new(2)))), // Update existing
            (3, Some((TeamColor::Yellow, PlayerId::new(3)))), // Add new
        ];
        router.set_id_map(new_map);
        assert_eq!(router.id_map.len(), 3);

        // Verify the updates
        assert_eq!(
            router.id_map.get(&2),
            Some(&Some((TeamColor::Blue, PlayerId::new(2))))
        );
    }

    #[test]
    fn test_id_map() {
        let mut router = create_test_router();
        let test_map = vec![(1, Some((TeamColor::Blue, PlayerId::new(1)))), (2, None)];
        router.set_id_map(test_map.clone());

        let returned_map = router.id_map();
        assert_eq!(returned_map.len(), 2);
        assert!(returned_map.contains(&(1, Some((TeamColor::Blue, PlayerId::new(1))))));
        assert!(returned_map.contains(&(2, None)));
    }

    #[test]
    fn test_get_by_player_id() {
        let mut router = create_test_router();
        let player_id = (TeamColor::Blue, PlayerId::new(5));

        // Test with empty map - should return player ID as robot ID
        assert_eq!(router.get_by_player_id(player_id), 5);

        // Test with mapped ID
        router.set_id_map(vec![(10, Some(player_id))]);
        assert_eq!(router.get_by_player_id(player_id), 10);

        // Test with unmapped player ID
        let unmapped_id = (TeamColor::Yellow, PlayerId::new(7));
        assert_eq!(router.get_by_player_id(unmapped_id), 7);
    }

    #[test]
    fn test_get_by_robot_id() {
        let mut router = create_test_router();

        // Test unmapped robot ID - should return default team color
        let result = router.get_by_robot_id(1);
        assert_eq!(result, (TeamColor::Yellow, PlayerId::new(1)));

        // Test mapped robot ID
        let mapped_player = (TeamColor::Blue, PlayerId::new(5));
        router.set_id_map(vec![(1, Some(mapped_player))]);
        assert_eq!(router.get_by_robot_id(1), mapped_player);

        // Test robot ID with None mapping
        router.set_id_map(vec![(2, None)]);
        assert_eq!(
            router.get_by_robot_id(2),
            (TeamColor::Yellow, PlayerId::new(2))
        );
    }

    #[test]
    fn test_complex_routing_scenario() {
        let mut router = create_test_router();

        // Set up a complex mapping
        let initial_map = vec![
            (1, Some((TeamColor::Blue, PlayerId::new(3)))),
            (2, Some((TeamColor::Yellow, PlayerId::new(1)))),
            (3, None),
            (4, Some((TeamColor::Blue, PlayerId::new(2)))),
        ];
        router.set_id_map(initial_map);

        // Test various routing scenarios
        assert_eq!(
            router.get_by_robot_id(1),
            (TeamColor::Blue, PlayerId::new(3))
        );
        assert_eq!(
            router.get_by_robot_id(3),
            (TeamColor::Yellow, PlayerId::new(3))
        );
        assert_eq!(
            router.get_by_player_id((TeamColor::Blue, PlayerId::new(3))),
            1
        );
        assert_eq!(
            router.get_by_player_id((TeamColor::Yellow, PlayerId::new(1))),
            2
        );
    }
}
