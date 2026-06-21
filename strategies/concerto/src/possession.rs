//! Possession tracker — the root stability surface.
//!
//! Everything downstream keys off one debounced classification of who controls
//! the ball. The raw, memoryless classification is noisy (single-frame breakbeam
//! blips, vision id swaps, momentary detection dropouts), so a latching filter
//! masks transients before any decision sees the result.
//!
//! This temporal filtering is *sensor conditioning*, not the forbidden
//! decision-hysteresis: it estimates a latent physical fact (who has the ball)
//! that cannot change every 16 ms. Requiring N agreeing frames is a
//! maximum-likelihood estimate of that fact, not a stay-bonus on a choice. The
//! planner downstream carries no preference for its previous decision.

use dies_strategy_api::prelude::*;
use dies_strategy_api::World;

use crate::config;

/// Stable, debounced possession state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Possession {
    /// We control the ball (the given teammate).
    We(PlayerId),
    /// An opponent controls the ball.
    Opp(PlayerId),
    /// Nobody clearly controls the ball.
    Loose,
}

/// Raw, memoryless classification for a single frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawPossession {
    /// A teammate controls the ball. `breakbeam` = hard sensor evidence.
    We {
        id: PlayerId,
        breakbeam: bool,
    },
    Opp(PlayerId),
    Loose,
    /// Ball not detected this frame.
    Unknown,
}

/// Classify the current frame with no memory.
///
/// Priority: our breakbeam → our proximity → opponent proximity → loose.
/// "We" beats "Opp" when the ball is contested.
pub fn classify_raw(world: &World) -> RawPossession {
    let ball_pos = match world.ball_position() {
        Some(p) => p,
        None => return RawPossession::Unknown,
    };

    // Hard evidence: a teammate's breakbeam is triggered.
    if let Some(p) = world.own_players().iter().find(|p| p.has_ball) {
        return RawPossession::We {
            id: p.id,
            breakbeam: true,
        };
    }

    // A fast ball can't be possessed by proximity — only the breakbeam (above)
    // can claim a fast-moving ball (a robot genuinely carrying it).
    let ball_speed = world.ball_velocity().map(|v| v.norm()).unwrap_or(0.0);
    if ball_speed >= config::POSSESSION_MAX_BALL_SPEED {
        return RawPossession::Loose;
    }

    // Soft evidence: nearest teammate within possession range.
    if let Some(p) = world.closest_own_player_to(ball_pos) {
        if (p.position - ball_pos).norm() < config::WE_POSSESSION_DIST {
            return RawPossession::We {
                id: p.id,
                breakbeam: false,
            };
        }
    }

    // Opponent possession.
    if let Some(p) = world.closest_opp_player_to(ball_pos) {
        if (p.position - ball_pos).norm() < config::OPP_POSSESSION_DIST {
            return RawPossession::Opp(p.id);
        }
    }

    RawPossession::Loose
}

/// Debouncing tracker. Hold across ticks; feed `update` the raw classification.
pub struct PossessionTracker {
    stable: Possession,
    candidate: Option<Possession>,
    candidate_count: u32,
    last_concrete_ts: f64,
    changed_this_tick: bool,
    /// The robot whose kick we were just told about (efference copy). While within
    /// `RELEASE_SUPPRESS_SECS`, its *proximity* re-acquisition of `We` is ignored.
    released_by: Option<PlayerId>,
    released_at: f64,
}

impl Default for PossessionTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl PossessionTracker {
    pub fn new() -> Self {
        Self {
            stable: Possession::Loose,
            candidate: None,
            candidate_count: 0,
            last_concrete_ts: 0.0,
            changed_this_tick: false,
            released_by: None,
            released_at: 0.0,
        }
    }

    /// Current stable possession.
    pub fn possession(&self) -> Possession {
        self.stable
    }

    /// Whether the stable possession committed a change on the most recent `update`.
    pub fn changed_this_tick(&self) -> bool {
        self.changed_this_tick
    }

    /// Advance the filter by one frame and return the stable possession.
    pub fn update(&mut self, raw: RawPossession, now: f64) -> Possession {
        self.changed_this_tick = false;

        match raw {
            RawPossession::Unknown => {
                // Detection dropout: hold the last stable state briefly, then decay
                // to Loose. Don't touch candidate accumulation.
                if now - self.last_concrete_ts > config::POSSESSION_HOLD_SECS
                    && self.stable != Possession::Loose
                {
                    self.commit(Possession::Loose);
                }
            }
            RawPossession::We { id, breakbeam } => {
                self.last_concrete_ts = now;
                // Suppress proximity re-acquisition by a robot that just kicked
                // (efference copy). Breakbeam is hard evidence and always passes.
                let suppressed = !breakbeam
                    && self.released_by == Some(id)
                    && now - self.released_at < config::RELEASE_SUPPRESS_SECS;
                if suppressed {
                    self.accumulate(Possession::Loose, false);
                } else {
                    self.accumulate(Possession::We(id), breakbeam);
                }
            }
            RawPossession::Opp(id) => {
                self.last_concrete_ts = now;
                self.accumulate(Possession::Opp(id), false);
            }
            RawPossession::Loose => {
                self.last_concrete_ts = now;
                self.accumulate(Possession::Loose, false);
            }
        }

        self.stable
    }

    /// Tell the tracker we commanded `kicker` to release the ball (efference copy).
    ///
    /// Immediately drops `We(kicker)` — bypassing the loss-debounce, because this
    /// loss is commanded, not sensed — and opens a short window during which the
    /// kicker can't re-acquire `We` by proximity (breakbeam still can, so a misfire
    /// where the ball never left self-corrects within a frame).
    pub fn notify_release(&mut self, kicker: PlayerId, now: f64) {
        self.released_by = Some(kicker);
        self.released_at = now;
        if self.stable == Possession::We(kicker) {
            self.commit(Possession::Loose);
        }
    }

    /// Accumulate evidence toward `observed`; commit once it is stable enough.
    /// `fast` (breakbeam-confirmed possession gain) commits in a single frame.
    fn accumulate(&mut self, observed: Possession, fast: bool) {
        if observed == self.stable {
            self.candidate = None;
            self.candidate_count = 0;
            return;
        }

        if self.candidate == Some(observed) {
            self.candidate_count += 1;
        } else {
            self.candidate = Some(observed);
            self.candidate_count = 1;
        }

        let needed = if fast { 1 } else { config::DEBOUNCE_FRAMES };
        if self.candidate_count >= needed {
            self.commit(observed);
        }
    }

    fn commit(&mut self, new: Possession) {
        if new != self.stable {
            self.stable = new;
            self.changed_this_tick = true;
        }
        self.candidate = None;
        self.candidate_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pid(n: u32) -> PlayerId {
        PlayerId::new(n)
    }

    #[test]
    fn breakbeam_gain_commits_in_one_frame() {
        let mut t = PossessionTracker::new();
        let p = t.update(
            RawPossession::We {
                id: pid(1),
                breakbeam: true,
            },
            0.0,
        );
        assert_eq!(p, Possession::We(pid(1)));
        assert!(t.changed_this_tick());
    }

    #[test]
    fn single_frame_blip_is_masked() {
        let mut t = PossessionTracker::new();
        t.update(
            RawPossession::We {
                id: pid(1),
                breakbeam: true,
            },
            0.0,
        );
        // One frame of Loose should NOT flip us (needs DEBOUNCE_FRAMES).
        let p = t.update(RawPossession::Loose, 0.016);
        assert_eq!(p, Possession::We(pid(1)));
        assert!(!t.changed_this_tick());
    }

    #[test]
    fn sustained_loss_commits_after_debounce() {
        let mut t = PossessionTracker::new();
        t.update(
            RawPossession::We {
                id: pid(1),
                breakbeam: true,
            },
            0.0,
        );
        let mut last = Possession::We(pid(1));
        for i in 0..config::DEBOUNCE_FRAMES {
            last = t.update(RawPossession::Loose, 0.1 + i as f64 * 0.016);
        }
        assert_eq!(last, Possession::Loose);
    }

    #[test]
    fn dropout_holds_then_decays_to_loose() {
        let mut t = PossessionTracker::new();
        t.update(
            RawPossession::We {
                id: pid(1),
                breakbeam: true,
            },
            0.0,
        );
        // Within hold window: keep We.
        assert_eq!(
            t.update(RawPossession::Unknown, 0.1),
            Possession::We(pid(1))
        );
        // Past hold window: decay to Loose.
        assert_eq!(t.update(RawPossession::Unknown, 1.0), Possession::Loose);
    }

    fn world_one_own(ball_pos: Vector2, ball_vel: Vector2, robot_pos: Vector2) -> World {
        use dies_strategy_protocol::WorldSnapshot;
        World::new(WorldSnapshot {
            timestamp: 0.0,
            dt: 0.016,
            field_geom: Some(FieldGeometry::default()),
            ball: Some(BallState {
                position: ball_pos,
                velocity: ball_vel,
                detected: true,
            }),
            own_players: vec![PlayerState::new(
                pid(1),
                robot_pos,
                Vector2::new(0.0, 0.0),
                Angle::from_radians(0.0),
            )],
            opp_players: vec![],
            game_state: GameState::Run,
            us_operating: true,
            our_keeper_id: None,
            freekick_kicker: None,
        })
    }

    #[test]
    fn fast_ball_near_robot_is_not_possessed() {
        // Robot right on the ball, but ball is rocketing away → Loose, not We.
        let w = world_one_own(
            Vector2::new(0.0, 0.0),
            Vector2::new(3000.0, 0.0),
            Vector2::new(50.0, 0.0),
        );
        assert_eq!(classify_raw(&w), RawPossession::Loose);
    }

    #[test]
    fn slow_ball_near_robot_is_we_by_proximity() {
        let w = world_one_own(
            Vector2::new(0.0, 0.0),
            Vector2::new(50.0, 0.0),
            Vector2::new(50.0, 0.0),
        );
        assert_eq!(
            classify_raw(&w),
            RawPossession::We {
                id: pid(1),
                breakbeam: false,
            }
        );
    }

    #[test]
    fn notify_release_drops_we_immediately() {
        let mut t = PossessionTracker::new();
        t.update(
            RawPossession::We {
                id: pid(1),
                breakbeam: true,
            },
            0.0,
        );
        t.notify_release(pid(1), 0.1);
        assert_eq!(t.possession(), Possession::Loose);
        assert!(t.changed_this_tick());
    }

    #[test]
    fn release_suppresses_proximity_but_not_breakbeam() {
        let mut t = PossessionTracker::new();
        t.update(
            RawPossession::We {
                id: pid(1),
                breakbeam: true,
            },
            0.0,
        );
        t.notify_release(pid(1), 0.1);

        // Proximity re-acquire by the kicker within the window is ignored.
        for i in 0..(config::DEBOUNCE_FRAMES + 2) {
            let p = t.update(
                RawPossession::We {
                    id: pid(1),
                    breakbeam: false,
                },
                0.1 + i as f64 * 0.005,
            );
            assert_eq!(p, Possession::Loose);
        }
        // But a breakbeam re-acquire (ball never left / real re-catch) overrides.
        let p = t.update(
            RawPossession::We {
                id: pid(1),
                breakbeam: true,
            },
            0.13,
        );
        assert_eq!(p, Possession::We(pid(1)));
    }

    #[test]
    fn suppression_expires() {
        let mut t = PossessionTracker::new();
        t.notify_release(pid(1), 0.0);
        // After the window, proximity We re-acquires (debounced as normal).
        let mut last = Possession::Loose;
        for i in 0..config::DEBOUNCE_FRAMES {
            last = t.update(
                RawPossession::We {
                    id: pid(1),
                    breakbeam: false,
                },
                1.0 + i as f64 * 0.016,
            );
        }
        assert_eq!(last, Possession::We(pid(1)));
    }
}
