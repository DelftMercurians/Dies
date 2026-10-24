//! State dispatch and state-specific match transitions.
use super::state::{NO_PROGRESS_DISTANCE, STOP_DURATION};
use super::*;

impl AutoReferee {
    pub(crate) fn update_game_state(&mut self, ctx: &mut RefereeContext<'_>) {
        dies_core::debug_string("Auto Game State", self.game_state.to_string());

        let mut game_state = std::mem::take(&mut self.game_state);
        self.game_state = match game_state {
            SimulationGameState::StartGame => {
                self.update_referee_command_full(
                    ctx,
                    referee::Command::STOP,
                    Some(referee::Command::PREPARE_KICKOFF_BLUE),
                    Some(STOP_DURATION),
                );
                SimulationGameState::stop_and_kickoff(TeamColor::Blue)
            }
            SimulationGameState::Stop => game_state,
            SimulationGameState::Halt => game_state,
            SimulationGameState::StopAndKickOff {
                team_color,
                ref mut stop_timer,
            } => {
                // Wait for 1s before starting the game
                if stop_timer.tick(ctx.physics.integration_parameters.dt) {
                    let cmd = match team_color {
                        TeamColor::Blue => referee::Command::PREPARE_KICKOFF_BLUE,
                        TeamColor::Yellow => referee::Command::PREPARE_KICKOFF_YELLOW,
                    };
                    self.update_referee_command_full(
                        ctx,
                        cmd,
                        Some(referee::Command::NORMAL_START),
                        Some(10.0),
                    );
                    SimulationGameState::prepare_kickoff(team_color)
                } else {
                    game_state
                }
            }
            SimulationGameState::StopAndForceStart { ref mut stop_timer } => {
                // Wait out the stop period before resuming the game.
                if stop_timer.tick(ctx.physics.integration_parameters.dt) {
                    self.update_referee_command(ctx, referee::Command::FORCE_START);
                    SimulationGameState::run()
                } else {
                    game_state
                }
            }
            SimulationGameState::StopAndFreeKick {
                team_color,
                ref mut stop_timer,
            } => {
                // Wait out the stop period, then award the free kick.
                if stop_timer.tick(ctx.physics.integration_parameters.dt) {
                    let cmd = match team_color {
                        TeamColor::Blue => referee::Command::DIRECT_FREE_BLUE,
                        TeamColor::Yellow => referee::Command::DIRECT_FREE_YELLOW,
                    };
                    self.update_referee_command_full(ctx, cmd, None, Some(10.0));
                    SimulationGameState::free_kick(team_color)
                } else {
                    game_state
                }
            }
            game_state @ SimulationGameState::PrepareKickOff { .. } => {
                self.update_kickoff(ctx, game_state)
            }
            game_state @ SimulationGameState::Run { .. } => self.update_running(ctx, game_state),
            game_state @ SimulationGameState::FreeKick { .. } => {
                self.update_free_kick(ctx, game_state)
            }
            game_state @ SimulationGameState::Penalty { .. } => {
                self.update_penalty(ctx, game_state)
            }
            game_state @ SimulationGameState::BallPlacement { .. } => {
                self.update_ball_placement(ctx, game_state)
            }
            SimulationGameState::Null => {
                // Do nothing, this is a placeholder state
                game_state
            }
        }
    }

    fn update_kickoff(
        &mut self,
        ctx: &mut RefereeContext<'_>,
        mut game_state: SimulationGameState,
    ) -> SimulationGameState {
        let SimulationGameState::PrepareKickOff {
            team_color,
            ref mut prepare_timer,
        } = game_state
        else {
            unreachable!("state-specific referee dispatch");
        };
        // If positions never become valid (e.g. the scoring team's
        // robots don't retreat to their own half), don't hang forever —
        // force the kickoff after the timer so the match keeps running.
        if prepare_timer.tick(ctx.physics.integration_parameters.dt) {
            self.kick_start_time = ctx.now;
            if let Some(ball_handle) = ctx
                .physics
                .ball
                .as_ref()
                .map(|ball| ball._rigid_body_handle)
            {
                let ball_body = ctx.physics.rigid_body_set.get(ball_handle).unwrap();
                self.kick_ball_position = Some(ball_body.position().translation.vector.xy());
            }
            self.kick_in_progress = true;
            self.last_kicker_info = None;
            self.update_referee_command(ctx, referee::Command::NORMAL_START);
            SimulationGameState::run()
        } else if self.check_kickoff_positions(ctx, team_color) {
            // Initialize kick tracking
            self.kick_start_time = ctx.now;
            if let Some(ball_handle) = ctx
                .physics
                .ball
                .as_ref()
                .map(|ball| ball._rigid_body_handle)
            {
                let ball_body = ctx.physics.rigid_body_set.get(ball_handle).unwrap();
                self.kick_ball_position = Some(ball_body.position().translation.vector.xy());
            }
            self.kick_in_progress = true;
            self.last_kicker_info = None; // Will be set when robot touches ball
            self.update_referee_command(ctx, referee::Command::NORMAL_START);
            SimulationGameState::run()
        } else {
            game_state
        }
    }

    fn update_running(
        &mut self,
        ctx: &mut RefereeContext<'_>,
        mut game_state: SimulationGameState,
    ) -> SimulationGameState {
        let SimulationGameState::Run {
            ref mut no_progress_timer,
            ref mut last_ball_position,
            ..
        } = game_state
        else {
            unreachable!("state-specific referee dispatch");
        };
        // Check if we're in a kick and should clear tracking due to time or ball movement
        if self.kick_in_progress {
            let ball_in_play = self.check_ball_in_play(ctx);
            let time_limit = 10.0; // 10 seconds for kickoff
            let time_exceeded = (ctx.now - self.kick_start_time) > time_limit;

            if ball_in_play || time_exceeded {
                self.kick_in_progress = false;
                self.last_kicker_info = None;
                self.kick_ball_position = None;
            }
        }

        if self.goal(ctx) {
            let ball_pos = {
                let ball_handle = ctx
                    .physics
                    .ball
                    .as_ref()
                    .map(|ball| ball._rigid_body_handle);
                if let Some(ball_handle) = ball_handle {
                    let ball_body = ctx.physics.rigid_body_set.get(ball_handle).unwrap();
                    ball_body.position().translation.vector.xy()
                } else {
                    Vector2::new(0.0, 0.0)
                }
            };

            // Clear kick tracking
            self.kick_in_progress = false;
            self.last_kicker_info = None;
            self.kick_ball_position = None;

            // Attribute the goal. The goal the ball entered belongs to the
            // team defending that side; that team conceded, the opponent
            // scored, and the conceding team restarts with a kick-off.
            let positive_defender = match self.side_assignment {
                SideAssignment::BlueOnPositive => TeamColor::Blue,
                SideAssignment::YellowOnPositive => TeamColor::Yellow,
            };
            let conceding = if ball_pos.x > 0.0 {
                positive_defender
            } else {
                positive_defender.opposite()
            };
            let scorer = conceding.opposite();
            let kickoff_team = conceding;

            match scorer {
                TeamColor::Blue => self.blue_score += 1,
                TeamColor::Yellow => self.yellow_score += 1,
            }
            self.push_referee_event(RefereeMessage::Goal, Some(scorer));

            // Re-center the ball for the kick-off so the match keeps
            // running (otherwise it stays in the goal and re-triggers
            // the goal check every frame). Go through a Stop period
            // first so robots slow down before the kick-off.
            self.place_ball(ctx, 0.0, 0.0);
            let next = match kickoff_team {
                TeamColor::Blue => referee::Command::PREPARE_KICKOFF_BLUE,
                TeamColor::Yellow => referee::Command::PREPARE_KICKOFF_YELLOW,
            };
            self.update_referee_command_full(
                ctx,
                referee::Command::STOP,
                Some(next),
                Some(STOP_DURATION),
            );
            SimulationGameState::stop_and_kickoff(kickoff_team)
        } else if self.ball_out(ctx) {
            // Clear kick tracking
            self.kick_in_progress = false;
            self.last_kicker_info = None;
            self.kick_ball_position = None;

            // Snap the ball back onto a reachable spot inside the field
            // so the free kick can actually be taken. Without this the
            // ball stays pinned against the boundary wall and the match
            // gets stuck. Also enforce the rule 5.3.3 defense-area
            // clearance so the kicker can legally get behind the ball.
            // Go through a Stop period first so robots slow down and back
            // off before the free kick.
            let placement = self.valid_free_kick_position(ctx, self.designated_ball_position.xy());
            self.place_ball(ctx, placement.x, placement.y);

            if let Some((_kicker_id, kicker_color)) = self.last_touch_info {
                // Free kick is awarded to the opponent of the last toucher.
                let awarded = kicker_color.opposite();
                let next = match awarded {
                    TeamColor::Blue => referee::Command::DIRECT_FREE_BLUE,
                    TeamColor::Yellow => referee::Command::DIRECT_FREE_YELLOW,
                };
                self.update_referee_command_full(
                    ctx,
                    referee::Command::STOP,
                    Some(next),
                    Some(STOP_DURATION),
                );
                SimulationGameState::stop_and_free_kick(awarded)
            } else {
                // No known last toucher — neutral restart with a force start.
                self.update_referee_command_full(
                    ctx,
                    referee::Command::STOP,
                    Some(referee::Command::FORCE_START),
                    Some(STOP_DURATION),
                );
                SimulationGameState::stop_and_force_start()
            }
        } else {
            // Check for no progress (rule 8.1): the timer only fires if the
            // ball stays within NO_PROGRESS_DISTANCE of a sticky reference
            // point for the whole window. As soon as the ball travels far
            // enough from that reference, we move the reference up and reset
            // the timer, so any real progress keeps the game running.
            let ball_handle = ctx
                .physics
                .ball
                .as_ref()
                .map(|ball| ball._rigid_body_handle);
            if let Some(ball_handle) = ball_handle {
                let ball_body = ctx.physics.rigid_body_set.get(ball_handle).unwrap();
                let ball_position_2d = ball_body.position().translation.vector.xy();
                match last_ball_position {
                    Some(reference)
                        if (ball_position_2d - *reference).norm() < NO_PROGRESS_DISTANCE =>
                    {
                        // Ball hasn't made progress — count down.
                        if no_progress_timer.tick(ctx.physics.integration_parameters.dt) {
                            // tick() resets the timer on fire; advance the
                            // reference so the next window starts cleanly.
                            *last_ball_position = Some(ball_position_2d);
                            self.push_referee_event(RefereeMessage::NoProgress, None);
                            self.update_referee_command_full(
                                ctx,
                                referee::Command::STOP,
                                Some(referee::Command::FORCE_START),
                                Some(STOP_DURATION),
                            );
                            SimulationGameState::stop_and_force_start()
                        } else {
                            game_state
                        }
                    }
                    _ => {
                        // First frame, or the ball moved far enough to count
                        // as progress: advance the reference and reset.
                        *last_ball_position = Some(ball_position_2d);
                        no_progress_timer.reset();
                        game_state
                    }
                }
            } else {
                game_state
            }
        }
    }

    fn update_free_kick(
        &mut self,
        ctx: &mut RefereeContext<'_>,
        mut game_state: SimulationGameState,
    ) -> SimulationGameState {
        let SimulationGameState::FreeKick {
            team_color,
            ref mut kick_timer,
            ball_is_kicked: _,
        } = game_state
        else {
            unreachable!("state-specific referee dispatch");
        };
        // Check if positions are valid
        if !self.check_free_kick_positions(ctx, team_color) {
            // Defending robots are crowding the ball. Raised at most once
            // per team per FOUL_GRACE_SECS (SSL §8.4.3) rather than every
            // physics frame — a sustained violation (e.g. a keeper pinned
            // by a ball behind its own goal line) is one foul per 2 s, not
            // hundreds.
            self.raise_foul(
                ctx,
                RefereeMessage::DefenderTooCloseToBall,
                Some(team_color.opposite()),
            );
            // Don't wait forever: tick the kick timer as a hard ceiling
            // and force the game back to running if they never clear out,
            // so self-play can't stall on a free kick.
            if kick_timer.tick(ctx.physics.integration_parameters.dt) {
                self.update_referee_command(ctx, referee::Command::FORCE_START);
                self.kick_in_progress = false;
                SimulationGameState::run()
            } else {
                game_state
            }
        } else {
            // Initialize kick tracking if not already done
            if !self.kick_in_progress {
                self.kick_start_time = ctx.now;
                if let Some(ball_handle) = ctx
                    .physics
                    .ball
                    .as_ref()
                    .map(|ball| ball._rigid_body_handle)
                {
                    let ball_body = ctx.physics.rigid_body_set.get(ball_handle).unwrap();
                    self.kick_ball_position = Some(ball_body.position().translation.vector.xy());
                }
                self.kick_in_progress = true;
                self.last_kicker_info = None;
            }

            // Check if ball is in play
            let ball_in_play = self.check_ball_in_play(ctx);

            let time_limit = 10.0;
            let time_exceeded = (ctx.now - self.kick_start_time) > time_limit;

            let is_free_kick_time_exceeded = kick_timer.tick(ctx.physics.integration_parameters.dt);

            if ball_in_play || time_exceeded {
                // Ball is in play or time exceeded - transition to run
                self.kick_in_progress = false;
                SimulationGameState::run()
            } else if is_free_kick_time_exceeded {
                // Original timer logic - keep for compatibility
                self.push_referee_event(RefereeMessage::FreekickTimeExceeded, None);
                self.kick_in_progress = false;
                SimulationGameState::run()
            } else {
                game_state
            }
        }
    }

    fn update_penalty(
        &mut self,
        ctx: &mut RefereeContext<'_>,
        game_state: SimulationGameState,
    ) -> SimulationGameState {
        let SimulationGameState::Penalty { team_color } = game_state else {
            unreachable!("state-specific referee dispatch");
        };
        // Positions only gate the *start* of the kick (NORMAL_START).
        // Once the kick is under way the attacker is allowed to approach
        // and manipulate the ball, so re-running the position check
        // would stall the timer/in-play tracking mid-kick.
        if !self.kick_in_progress && !self.check_penalty_positions(ctx, team_color) {
            // Stay in penalty state until positions are correct
            game_state
        } else {
            // Initialize kick tracking if not already done
            if !self.kick_in_progress {
                self.kick_start_time = ctx.now;
                if let Some(ball_handle) = ctx
                    .physics
                    .ball
                    .as_ref()
                    .map(|ball| ball._rigid_body_handle)
                {
                    let ball_body = ctx.physics.rigid_body_set.get(ball_handle).unwrap();
                    self.kick_ball_position = Some(ball_body.position().translation.vector.xy());
                }
                self.kick_in_progress = true;
                self.last_kicker_info = None;

                // Issue normal start command
                self.update_referee_command(ctx, referee::Command::NORMAL_START);
            }

            // Check if ball is in play
            let ball_in_play = self.check_ball_in_play(ctx);

            // Check 10 second time limit
            let time_exceeded = (ctx.now - self.kick_start_time) > 10.0;

            // Check if ball direction is valid (toward goal)
            let direction_valid = self.check_penalty_ball_direction(ctx, team_color);

            if time_exceeded {
                self.push_referee_event(RefereeMessage::PenaltyTimeExceeded, None);
                self.kick_in_progress = false;
                // Award goal kick to defending team
                let defending_team = if team_color == TeamColor::Blue {
                    TeamColor::Yellow
                } else {
                    TeamColor::Blue
                };
                match team_color {
                    TeamColor::Blue => {
                        self.update_referee_command(ctx, referee::Command::DIRECT_FREE_BLUE);
                    }
                    TeamColor::Yellow => {
                        self.update_referee_command(ctx, referee::Command::DIRECT_FREE_YELLOW);
                    }
                }
                SimulationGameState::free_kick(defending_team)
            } else if ball_in_play && !direction_valid {
                // Ball moved but in wrong direction - award goal kick to defending team
                self.kick_in_progress = false;
                let defending_team = if team_color == TeamColor::Blue {
                    TeamColor::Yellow
                } else {
                    TeamColor::Blue
                };
                match team_color {
                    TeamColor::Blue => {
                        self.update_referee_command(ctx, referee::Command::DIRECT_FREE_BLUE);
                    }
                    TeamColor::Yellow => {
                        self.update_referee_command(ctx, referee::Command::DIRECT_FREE_YELLOW);
                    }
                }
                SimulationGameState::free_kick(defending_team)
            } else if ball_in_play {
                // Ball in play and direction is valid - continue with run state
                self.kick_in_progress = false;
                SimulationGameState::run()
            } else {
                game_state
            }
        }
    }

    fn update_ball_placement(
        &mut self,
        ctx: &mut RefereeContext<'_>,
        game_state: SimulationGameState,
    ) -> SimulationGameState {
        let SimulationGameState::BallPlacement {
            team_color,
            position,
        } = game_state
        else {
            unreachable!("state-specific referee dispatch");
        };
        match team_color {
            TeamColor::Blue => {
                self.update_referee_command(ctx, referee::Command::BALL_PLACEMENT_BLUE);
            }
            TeamColor::Yellow => {
                self.update_referee_command(ctx, referee::Command::BALL_PLACEMENT_YELLOW);
            }
        }
        let ball_pos = {
            let ball_handle = ctx
                .physics
                .ball
                .as_ref()
                .map(|ball| ball._rigid_body_handle);
            if let Some(ball_handle) = ball_handle {
                let ball_body = ctx.physics.rigid_body_set.get(ball_handle).unwrap();
                ball_body.position().translation.vector.xy()
            } else {
                Vector2::new(0.0, 0.0)
            }
        };
        if (ball_pos - position).norm() < 100.0 {
            let opposite_team = team_color.opposite();
            match opposite_team {
                TeamColor::Blue => {
                    self.update_referee_command(ctx, referee::Command::DIRECT_FREE_BLUE);
                }
                TeamColor::Yellow => {
                    self.update_referee_command(ctx, referee::Command::DIRECT_FREE_YELLOW);
                }
            }
            SimulationGameState::free_kick(opposite_team)
        } else {
            game_state
        }
    }
}
