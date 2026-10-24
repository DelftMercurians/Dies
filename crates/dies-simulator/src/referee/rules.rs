//! Touch tracking, goals, boundaries, and restart-position checks.
use super::*;
use crate::ball::BALL_RADIUS;

impl AutoReferee {
    pub(crate) fn observe_touches(&mut self, ctx: &mut RefereeContext<'_>) {
        // Update last touch info
        if matches!(self.game_state, SimulationGameState::Run { .. }) {
            if let Some(ball_handle) = ctx.physics.ball.as_ref().map(|b| b._rigid_body_handle) {
                let ball_body = ctx.physics.rigid_body_set.get(ball_handle).unwrap();
                let ball_position = ball_body.position().translation.vector;
                let touch_threshold = ctx.config.player_radius + BALL_RADIUS + 5.0;

                // Find the closest player touching the ball in this step
                let mut closest_touching_player: Option<(PlayerId, TeamColor, f64)> = None;

                for player in ctx.robots.players.iter() {
                    let player_body = ctx
                        .physics
                        .rigid_body_set
                        .get(player.rigid_body_handle)
                        .unwrap();
                    let player_position = player_body.position().translation.vector;
                    let distance = (player_position - ball_position).norm();

                    if distance < touch_threshold
                        && (closest_touching_player.is_none()
                            || distance < closest_touching_player.unwrap().2)
                    {
                        closest_touching_player = Some((player.id, player.team_color, distance));
                    }
                }

                if let Some((id, team_color, _)) = closest_touching_player {
                    // Check for double touch during kicks
                    if self.kick_in_progress {
                        if self.check_double_touch((id, team_color)) {
                            // Double touch violation - stop game and award free kick to opponent
                            self.kick_in_progress = false;
                            self.last_kicker_info = None;
                            self.kick_ball_position = None;

                            self.push_referee_event(
                                RefereeMessage::DoubleTouchViolation,
                                Some(team_color),
                            );

                            let opponent_team = if team_color == TeamColor::Blue {
                                TeamColor::Yellow
                            } else {
                                TeamColor::Blue
                            };
                            self.game_state = SimulationGameState::free_kick(opponent_team);
                            let cmd = if opponent_team == TeamColor::Blue {
                                referee::Command::DIRECT_FREE_BLUE
                            } else {
                                referee::Command::DIRECT_FREE_YELLOW
                            };
                            self.update_referee_command_full(ctx, cmd, None, Some(10.0));
                        } else if self.last_kicker_info.is_none() {
                            // Set the first kicker
                            self.last_kicker_info = Some((id, team_color));
                        }
                    }

                    self.last_touch_info = Some((id, team_color));
                }
            }
        } else {
            self.last_touch_info = None;
        }
    }

    pub(crate) fn check_robot_allowances(&mut self, ctx: &mut RefereeContext<'_>) {
        // Flag a team that is fielding more robots than its current card-adjusted
        // allowance (SSL TOO_MANY_ROBOTS). Grace-limited like other §8.4.3 fouls.
        for team in [TeamColor::Blue, TeamColor::Yellow] {
            if self.on_field_count(ctx, team) > self.max_allowed_bots_for(ctx, team) {
                self.raise_foul(ctx, RefereeMessage::TooManyRobots, Some(team));
            }
        }
    }

    pub(crate) fn ball_out(&mut self, ctx: &mut RefereeContext<'_>) -> bool {
        let ball_handle = ctx
            .physics
            .ball
            .as_ref()
            .map(|ball| ball._rigid_body_handle);
        if let Some(ball_handle) = ball_handle {
            let ball_body = ctx.physics.rigid_body_set.get(ball_handle).unwrap();
            let ball_position = ball_body.position().translation.vector;

            let half_length = ctx.config.field_geometry.field_length / 2.0;
            let half_width = ctx.config.field_geometry.field_width / 2.0;

            // Clamp a candidate free-kick spot to the reachable field margin so
            // the placement is always inside the playable area.
            let clamp_designated = |mut p: Vector<f64>| {
                p.x = p.x.clamp(
                    -(half_length - FREE_KICK_PLACEMENT_MARGIN),
                    half_length - FREE_KICK_PLACEMENT_MARGIN,
                );
                p.y = p.y.clamp(
                    -(half_width - FREE_KICK_PLACEMENT_MARGIN),
                    half_width - FREE_KICK_PLACEMENT_MARGIN,
                );
                p
            };

            // A ball is out only once its centre has actually crossed a real
            // field line. The out-of-bounds line must be the true field edge,
            // not a shrunk one: the placement margin (FREE_KICK_PLACEMENT_MARGIN)
            // already keeps a placed free-kick ball inside the field, so testing
            // against any smaller boundary would flag a legally-placed corner
            // free kick as immediately out and re-stop the game.
            if ball_position.x.abs() > half_length || ball_position.y.abs() > half_width {
                dies_core::debug_string(
                    "RefereeMessage",
                    format!("Ball out of bounds at position: {:?}", ball_position),
                );
                // Recompute the designated position from where the ball actually
                // is, so a ball that is already out (e.g. came to rest just past
                // the line, or crossed in a single fast frame) does not reuse a
                // stale designated position from an earlier out-of-bounds on the
                // other side of the field.
                self.designated_ball_position = clamp_designated(ball_position);
                // The actual reposition happens in `update_game_state` once the
                // free kick is awarded (see `place_ball`); here we only report
                // that the ball has left the field.
                return true;
            }

            // Still inside: predictively record where it *will* leave so the
            // free-kick placement is ready, but keep the game running this frame.
            let ball_velocity: Vector<f64> = *ball_body.linvel(); // ball's current velocity as Vector3
            let next_ball_position =
                ball_position + ball_velocity * ctx.physics.integration_parameters.dt;
            if next_ball_position.x.abs() > half_length || next_ball_position.y.abs() > half_width {
                dies_core::debug_string(
                    "RefereeMessage.BallOut",
                    format!("Ball out of bounds at position: {:?}", next_ball_position),
                );
                // Linear interpolation toward where it crosses the line.
                let designated = ball_position + (next_ball_position - ball_position) * 0.5;
                self.designated_ball_position = clamp_designated(designated);
            }
            false
        } else {
            false
        }
    }

    pub(crate) fn goal(&mut self, ctx: &mut RefereeContext<'_>) -> bool {
        let ball_handle = ctx
            .physics
            .ball
            .as_ref()
            .map(|ball| ball._rigid_body_handle);
        if let Some(ball_handle) = ball_handle {
            let ball_body = ctx.physics.rigid_body_set.get(ball_handle).unwrap();
            let ball_position = ball_body.position().translation.vector; // Center of the ball

            let goal_height = 150.0;
            // A goal is scored once the ball crosses the goal line (field_length/2)
            // into the net, which extends back to field_length/2 + goal_depth.
            if ball_position.x.abs() > ctx.config.field_geometry.field_length / 2.0
                && ball_position.x.abs()
                    < ctx.config.field_geometry.field_length / 2.0
                        + ctx.config.field_geometry.goal_depth
                && ball_position.y.abs() < ctx.config.field_geometry.goal_width / 2.0
                && ball_position.z < goal_height
            {
                // Log which side scored
                let scoring_side = if ball_position.x > 0.0 {
                    "positive x side (blue goal)"
                } else {
                    "negative x side (yellow goal)"
                };
                dies_core::debug_string(
                    "RefereeMessage.Goal",
                    format!("Goal scored at {}", scoring_side),
                );

                // The ball is re-centered for the kick-off in `update_game_state`
                // (see `place_ball`); here we only report that a goal was scored.
                return true;
            }
        }
        false
    }

    pub(crate) fn check_kickoff_positions(
        &self,
        ctx: &mut RefereeContext<'_>,
        attacking_team: TeamColor,
    ) -> bool {
        let ball_handle = ctx
            .physics
            .ball
            .as_ref()
            .map(|ball| ball._rigid_body_handle);
        if let Some(ball_handle) = ball_handle {
            let ball_body = ctx.physics.rigid_body_set.get(ball_handle).unwrap();
            let ball_position = ball_body.position().translation.vector;

            let center_circle_radius = 500.0; // 1m diameter = 0.5m radius
            let touch_distance = ctx.config.player_radius + BALL_RADIUS;
            let mut attacking_robots_in_circle = 0;

            for player in ctx.robots.players.iter() {
                let rigid_body = ctx
                    .physics
                    .rigid_body_set
                    .get(player.rigid_body_handle)
                    .unwrap();
                let player_position = rigid_body.position().translation.vector;

                // Check if robot is touching the ball (not allowed)
                if (player_position - ball_position).norm() < touch_distance {
                    dies_core::debug_string(
                        "RefereeMessage",
                        RefereeMessage::KickoffPositionViolation.to_string(),
                    );
                    return false;
                }

                // Every robot must be on its own half. The only exception is a
                // single attacking robot allowed inside the center circle (which
                // straddles the halfway line) so it can take the kick.
                let is_on_own_side = self
                    .side_assignment
                    .is_on_own_side_vec3(player.team_color, &player_position);
                if !is_on_own_side {
                    let distance_to_center = (player_position.xy() - Vector2::new(0.0, 0.0)).norm();
                    let is_kicker_in_circle = player.team_color == attacking_team
                        && distance_to_center <= center_circle_radius;
                    if is_kicker_in_circle {
                        attacking_robots_in_circle += 1;
                    }
                    if !is_kicker_in_circle || attacking_robots_in_circle > 1 {
                        dies_core::debug_string(
                            "RefereeMessage",
                            RefereeMessage::KickoffPositionViolation.to_string(),
                        );
                        return false;
                    }
                }
            }
        }
        true
    }

    pub(crate) fn check_free_kick_positions(
        &self,
        ctx: &mut RefereeContext<'_>,
        attacking_team: TeamColor,
    ) -> bool {
        let ball_handle = ctx
            .physics
            .ball
            .as_ref()
            .map(|ball| ball._rigid_body_handle);
        if let Some(ball_handle) = ball_handle {
            let ball_body = ctx.physics.rigid_body_set.get(ball_handle).unwrap();
            let ball_position = ball_body.position().translation.vector;

            let required_distance = 500.0; // 0.5m

            for player in ctx.robots.players.iter() {
                // Only check defending team robots
                if player.team_color != attacking_team {
                    let rigid_body = ctx
                        .physics
                        .rigid_body_set
                        .get(player.rigid_body_handle)
                        .unwrap();
                    let player_position = rigid_body.position().translation.vector;

                    if (player_position - ball_position).norm() < required_distance {
                        dies_core::debug_string(
                            "RefereeMessage",
                            RefereeMessage::DefenderTooCloseToBall.to_string(),
                        );
                        return false;
                    }
                }
            }
        }
        true
    }

    pub(crate) fn check_penalty_positions(
        &self,
        ctx: &mut RefereeContext<'_>,
        attacking_team: TeamColor,
    ) -> bool {
        let ball_handle = ctx
            .physics
            .ball
            .as_ref()
            .map(|ball| ball._rigid_body_handle);
        if let Some(ball_handle) = ball_handle {
            let ball_body = ctx.physics.rigid_body_set.get(ball_handle).unwrap();
            let ball_position = ball_body.position().translation.vector;

            let defending_team = if attacking_team == TeamColor::Blue {
                TeamColor::Yellow
            } else {
                TeamColor::Blue
            };
            let goal_x = if self
                .side_assignment
                .is_on_opp_side_vec3(defending_team, &ball_position)
            {
                ctx.config.field_geometry.field_length / 2.0
            } else {
                -ctx.config.field_geometry.field_length / 2.0
            };

            let mut keeper_found = false;
            let mut keeper_position_valid = false;

            // One attacking robot — the kicker — is allowed to approach the ball
            // (but not touch it) during the penalty setup (rules §5.3.5). Exempt
            // the attacker closest to the ball from the 1m-behind check; without
            // this exemption a legally staged kicker deadlocks the penalty state
            // (NORMAL_START is never issued).
            let kicker_index = ctx
                .robots
                .players
                .iter()
                .enumerate()
                .filter(|(_, p)| p.team_color == attacking_team)
                .min_by(|(_, a), (_, b)| {
                    let da = (ctx
                        .physics
                        .rigid_body_set
                        .get(a.rigid_body_handle)
                        .unwrap()
                        .position()
                        .translation
                        .vector
                        - ball_position)
                        .norm();
                    let db = (ctx
                        .physics
                        .rigid_body_set
                        .get(b.rigid_body_handle)
                        .unwrap()
                        .position()
                        .translation
                        .vector
                        - ball_position)
                        .norm();
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i);

            for (index, player) in ctx.robots.players.iter().enumerate() {
                let rigid_body = ctx
                    .physics
                    .rigid_body_set
                    .get(player.rigid_body_handle)
                    .unwrap();
                let player_position = rigid_body.position().translation.vector;

                if Some(index) == kicker_index {
                    // The kicker may approach but must not touch the ball.
                    let touch_threshold = ctx.config.player_radius + BALL_RADIUS + 30.0;
                    if (player_position - ball_position).norm() < touch_threshold {
                        dies_core::debug_string(
                            "RefereeMessage",
                            RefereeMessage::PenaltyRobotPositionViolation.to_string(),
                        );
                        return false;
                    }
                    continue;
                }

                // The keeper is the robot the sim's own referee designated.
                if player.team_color == defending_team && player.id.as_u32() == SIM_GOALKEEPER_ID {
                    keeper_found = true;
                    // "On the goal line" = the robot body touches the line: centre
                    // within one robot radius. The old 2cm tolerance was tighter
                    // than the rule and than control jitter can hold, deadlocking
                    // the penalty state.
                    let goal_line_tolerance = 90.0;
                    let goal_width = ctx.config.field_geometry.goal_width;

                    if (player_position.x - goal_x).abs() < goal_line_tolerance
                        && player_position.y.abs() < goal_width / 2.0
                    {
                        keeper_position_valid = true;
                    } else {
                        dies_core::debug_string(
                            "RefereeMessage",
                            RefereeMessage::PenaltyKeeperPositionViolation.to_string(),
                        );
                        return false;
                    }
                } else {
                    // Check if other robots are 1m behind ball
                    let required_distance = 1000.0; // 1m
                    let ball_to_goal_direction = if goal_x > ball_position.x { 1.0 } else { -1.0 };
                    let robot_behind_ball =
                        (player_position.x - ball_position.x) * ball_to_goal_direction < 0.0;

                    if !robot_behind_ball
                        || (player_position - ball_position).norm() < required_distance
                    {
                        dies_core::debug_string(
                            "RefereeMessage",
                            RefereeMessage::PenaltyRobotPositionViolation.to_string(),
                        );
                        return false;
                    }
                }
            }

            if !keeper_found || !keeper_position_valid {
                dies_core::debug_string(
                    "RefereeMessage",
                    RefereeMessage::PenaltyKeeperPositionViolation.to_string(),
                );
                return false;
            }
        }
        true
    }

    pub(crate) fn check_ball_in_play(&self, ctx: &mut RefereeContext<'_>) -> bool {
        if let Some(kick_ball_pos) = self.kick_ball_position {
            let ball_handle = ctx
                .physics
                .ball
                .as_ref()
                .map(|ball| ball._rigid_body_handle);
            if let Some(ball_handle) = ball_handle {
                let ball_body = ctx.physics.rigid_body_set.get(ball_handle).unwrap();
                let current_ball_pos = ball_body.position().translation.vector.xy();

                // Check if ball moved 0.05m (50mm)
                if (current_ball_pos - kick_ball_pos).norm() >= 50.0 {
                    return true;
                }
            }
        }
        false
    }

    pub(crate) fn check_double_touch(&self, touching_robot: (PlayerId, TeamColor)) -> bool {
        if let Some(last_kicker) = self.last_kicker_info {
            if last_kicker.0 == touching_robot.0 && last_kicker.1 == touching_robot.1 {
                dies_core::debug_string(
                    "RefereeMessage",
                    RefereeMessage::DoubleTouchViolation.to_string(),
                );
                return true;
            }
        }
        false
    }

    pub(crate) fn check_penalty_ball_direction(
        &self,
        ctx: &mut RefereeContext<'_>,
        attacking_team: TeamColor,
    ) -> bool {
        let ball_handle = ctx
            .physics
            .ball
            .as_ref()
            .map(|ball| ball._rigid_body_handle);
        if let Some(ball_handle) = ball_handle {
            let ball_body = ctx.physics.rigid_body_set.get(ball_handle).unwrap();
            let ball_velocity = ball_body.linvel();

            // Check if ball is moving toward opponent goal
            let defending_team = if attacking_team == TeamColor::Blue {
                TeamColor::Yellow
            } else {
                TeamColor::Blue
            };
            let ball_position = ball_body.position().translation.vector;
            let goal_x = if self
                .side_assignment
                .is_on_opp_side_vec3(defending_team, &ball_position)
            {
                ctx.config.field_geometry.field_length / 2.0
            } else {
                -ctx.config.field_geometry.field_length / 2.0
            };

            let toward_goal = (goal_x - ball_position.x).signum();
            let ball_direction = ball_velocity.x.signum();

            if ball_velocity.norm() > 0.1 && ball_direction != toward_goal {
                dies_core::debug_string(
                    "RefereeMessage",
                    RefereeMessage::PenaltyBallDirectionViolation.to_string(),
                );
                return false;
            }
        }
        true
    }
}
