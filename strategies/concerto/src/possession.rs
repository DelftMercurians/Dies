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
                self.accumulate(Possession::We(id), breakbeam);
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
}
