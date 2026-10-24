//! Card expiry, robot allowances, and physical substitutions.
use super::*;
/// Yellow-card lifetime in seconds (SSL §5.5.1: a card expires after 120 s of
/// playing time, restoring one robot to the team's allowance).
const YELLOW_CARD_DURATION_SECS: f64 = 120.0;
/// Base robot allowance per team before any cards (SSL division B: 6).
const BASE_MAX_BOTS: u32 = 6;

/// Compute a team's referee `TeamInfo` card fields from its active card
/// expiries: the current `max_allowed_bots` (base minus active cards, floored
/// at 1) and the `yellow_card_times` list (one positive entry per active card,
/// holding whole seconds remaining — the world tracker only counts entries > 0).
pub(crate) fn card_team_info(expiries: &[f64], now: f64) -> (u32, Vec<u32>) {
    let mut times: Vec<u32> = expiries
        .iter()
        .filter(|&&e| e > now)
        .map(|&e| ((e - now).ceil() as u32).max(1))
        .collect();
    times.sort_unstable_by(|a, b| b.cmp(a));
    let max_bots = BASE_MAX_BOTS.saturating_sub(times.len() as u32).max(1);
    (max_bots, times)
}

impl AutoReferee {
    /// Show a yellow card to `team` (SSL §5.5.1): record a 120 s expiry, lowering
    /// the team's `max_allowed_bots` by one until it expires, and immediately
    /// re-broadcast the current referee command so the new allowance reaches the
    /// world tracker without waiting for the next game-state transition. The
    /// executor's compliance layer then parks the excess robot off-field; on
    /// expiry the allowance is restored (see [`AutoReferee::update_cards`]).
    pub(crate) fn add_yellow_card(&mut self, ctx: &mut RefereeContext<'_>, team: TeamColor) {
        let expiry = ctx.now + YELLOW_CARD_DURATION_SECS;
        match team {
            TeamColor::Blue => self.blue_card_expiries.push(expiry),
            TeamColor::Yellow => self.yellow_card_expiries.push(expiry),
        }
        self.push_referee_event(RefereeMessage::YellowCard, Some(team));
        self.rebroadcast_referee(ctx);
    }

    /// Number of robots a team is currently allowed on the field.
    pub(crate) fn max_allowed_bots_for(
        &self,
        ctx: &mut RefereeContext<'_>,
        team: TeamColor,
    ) -> u32 {
        let expiries = match team {
            TeamColor::Blue => &self.blue_card_expiries,
            TeamColor::Yellow => &self.yellow_card_expiries,
        };
        card_team_info(expiries, ctx.now).0
    }

    /// Re-emit the last referee command with refreshed `TeamInfo` (card / max-bot
    /// fields). Game state is unaffected: the world game-state tracker
    /// early-returns on an unchanged command but still re-reads the card fields.
    pub(crate) fn rebroadcast_referee(&mut self, ctx: &mut RefereeContext<'_>) {
        let cmd = self.last_command;
        self.update_referee_command_full(ctx, cmd, None, None);
    }

    /// Prune expired yellow cards and, if any expired this step, re-broadcast the
    /// referee command so the restored allowance reaches the world tracker.
    /// Driven off sim time, so it is deterministic and FTR-safe.
    pub(crate) fn update_cards(&mut self, ctx: &mut RefereeContext<'_>) {
        let now = ctx.now;
        let before = self.blue_card_expiries.len() + self.yellow_card_expiries.len();
        self.blue_card_expiries.retain(|&e| e > now);
        self.yellow_card_expiries.retain(|&e| e > now);
        let after = self.blue_card_expiries.len() + self.yellow_card_expiries.len();
        if after < before {
            self.rebroadcast_referee(ctx);
        }
    }

    /// Model the physical substitution that a yellow card forces: once the
    /// executor has parked a carded robot beyond the touchline (its compliance
    /// layer drives it to the substitution line), the sim removes it so the
    /// strategy sees the true reduced roster — exactly as a real robot leaving the
    /// cameras' view would. When a card later expires the robot is re-added at the
    /// halfway line, where the strategy picks it back up and assigns it a role.
    /// Deterministic: keyed only on sim-time card state + robot positions.
    pub(crate) fn update_card_substitutions(&mut self, ctx: &mut RefereeContext<'_>) {
        let beyond = ctx.config.field_geometry.field_width / 2.0 + 30.0;
        let reentry_y = ctx.config.field_geometry.field_width / 2.0 - 200.0;
        for team in [TeamColor::Blue, TeamColor::Yellow] {
            let (target, out_len) = match team {
                TeamColor::Blue => (self.blue_card_expiries.len(), self.blue_carded_out.len()),
                TeamColor::Yellow => (
                    self.yellow_card_expiries.len(),
                    self.yellow_carded_out.len(),
                ),
            };
            if out_len < target {
                // Need to serve more cards: remove parked robots (those the executor
                // has driven beyond the touchline) not already taken out.
                let already: &[PlayerId] = match team {
                    TeamColor::Blue => &self.blue_carded_out,
                    TeamColor::Yellow => &self.yellow_carded_out,
                };
                let mut cands: Vec<PlayerId> = ctx
                    .robots
                    .players
                    .iter()
                    .filter(|p| p.team_color == team && !already.contains(&p.id))
                    .filter_map(|p| {
                        let y = ctx
                            .physics
                            .rigid_body_set
                            .get(p.rigid_body_handle)
                            .map(|b| b.position().translation.vector.y)
                            .unwrap_or(0.0);
                        if y.abs() > beyond {
                            Some(p.id)
                        } else {
                            None
                        }
                    })
                    .collect();
                cands.sort();
                cands.truncate(target - out_len);
                for id in cands {
                    ctx.actions.push(RefereeAction::RemoveRobot { team, id });
                    match team {
                        TeamColor::Blue => self.blue_carded_out.push(id),
                        TeamColor::Yellow => self.yellow_carded_out.push(id),
                    }
                }
            } else if out_len > target {
                // Cards expired: re-add robots at the halfway-line substitution zone.
                for k in 0..(out_len - target) {
                    let id = match team {
                        TeamColor::Blue => self.blue_carded_out.pop(),
                        TeamColor::Yellow => self.yellow_carded_out.pop(),
                    };
                    if let Some(id) = id {
                        let pos = Vector2::new(-300.0 * k as f64, reentry_y);
                        ctx.actions.push(RefereeAction::AddRobot {
                            team,
                            id,
                            position: pos,
                            yaw: Angle::from_radians(0.0),
                        });
                    }
                }
            }
        }
    }

    /// Count of a team's robots currently inside the touch/goal lines (parked,
    /// off-field robots at the substitution line are excluded). Used for the
    /// too-many-robots flag.
    pub(crate) fn on_field_count(&self, ctx: &mut RefereeContext<'_>, team: TeamColor) -> u32 {
        let half_w = ctx.config.field_geometry.field_width / 2.0;
        let half_l = ctx.config.field_geometry.field_length / 2.0;
        ctx.robots
            .players
            .iter()
            .filter(|p| p.team_color == team)
            .filter(|p| {
                let pos = ctx
                    .physics
                    .rigid_body_set
                    .get(p.rigid_body_handle)
                    .map(|b| b.position().translation.vector)
                    .unwrap_or_default();
                pos.x.abs() <= half_l && pos.y.abs() <= half_w
            })
            .count() as u32
    }
}
