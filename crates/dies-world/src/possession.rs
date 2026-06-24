//! Unified ball-possession tracker — the single place possession is decided.
//!
//! Evidence sources depend on whether the robot is one we control:
//!   * **controlled robots** (our own team) use *only* their **breakbeam**,
//!     *gated* on the ball actually being near the dribbler — a breakbeam trigger
//!     with the ball nowhere near is rejected. Proximity never contributes for a
//!     controlled robot, so a controlled robot that merely brushes the ball is
//!     never credited with possession.
//!   * **uncontrolled robots** (opponents, where we have no breakbeam) fall back
//!     to **proximity** of the ball to the robot, with a two-band hysteresis (gain
//!     at `acquire_dist`, retain until the ball leaves `release_dist`).
//!
//! Edges are confidence-driven: a breakbeam-backed gain commits in one frame,
//! while a proximity-only change must persist `acquire_frames`. A vision/contact
//! dropout holds the last belief (marked `stale`) for `hold_secs` before decaying
//! to `Loose`. Multiple equally-plausible owners surface as `Contested`. A
//! commanded kick (efference copy) drops the kicker and suppresses its *proximity*
//! re-acquisition briefly — breakbeam re-acquisition is never suppressed, so a
//! misfire where the ball never left self-corrects within a frame.

use dies_core::{
    BallData, PlayerData, PossessionConfig, PossessionState, TeamColor, TeamPlayerId, Vector2,
};

/// Raw, memoryless classification for a single frame.
#[derive(Debug, Clone, PartialEq)]
enum RawClass {
    Owned {
        who: TeamPlayerId,
        breakbeam: bool,
    },
    Contested(Vec<TeamPlayerId>),
    Loose,
    /// Ball not detected this frame.
    Unknown,
}

/// Sort key giving `Contested` candidate lists a deterministic order (so equality
/// across frames is stable).
fn order_key(t: &TeamPlayerId) -> (u8, u32) {
    let team = match t.team_color {
        TeamColor::Blue => 0,
        TeamColor::Yellow => 1,
    };
    (team, t.player_id.as_u32())
}

/// Debouncing possession tracker. Hold across ticks; feed `update` each frame.
pub struct PossessionTracker {
    stable: PossessionState,
    candidate: Option<PossessionState>,
    candidate_count: u32,
    /// Last time we had a concrete (non-`Unknown`) classification.
    last_concrete_ts: f64,
    stale: bool,
    /// The robot whose kick we were told about (efference copy) and when. While
    /// within `release_suppress_secs`, its *proximity* re-acquisition is ignored.
    released_by: Option<TeamPlayerId>,
    released_at: f64,
    /// A kick reported since the last `update`, consumed on the next one.
    pending_release: Option<TeamPlayerId>,
}

impl Default for PossessionTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl PossessionTracker {
    pub fn new() -> Self {
        Self {
            stable: PossessionState::Loose,
            candidate: None,
            candidate_count: 0,
            last_concrete_ts: 0.0,
            stale: false,
            released_by: None,
            released_at: 0.0,
            pending_release: None,
        }
    }

    /// Current stable possession with its staleness flag.
    pub fn possession(&self) -> dies_core::Possession {
        dies_core::Possession {
            state: self.stable.clone(),
            stale: self.stale,
        }
    }

    /// Tell the tracker `who` was commanded to kick (efference copy). Consumed on
    /// the next `update`: drops the kicker and opens the proximity-suppress window.
    pub fn notify_kick(&mut self, who: TeamPlayerId) {
        self.pending_release = Some(who);
    }

    /// Advance the filter by one frame and return the stable possession.
    pub fn update(
        &mut self,
        ball: Option<&BallData>,
        blue: &[PlayerData],
        yellow: &[PlayerData],
        controlled_blue: bool,
        controlled_yellow: bool,
        cfg: &PossessionConfig,
        now: f64,
    ) -> dies_core::Possession {
        if let Some(k) = self.pending_release.take() {
            self.released_by = Some(k);
            self.released_at = now;
            if self.stable.owner() == Some(k) {
                self.commit(PossessionState::Loose);
            }
        }

        let raw = self.classify(ball, blue, yellow, controlled_blue, controlled_yellow, cfg);
        self.stale = false;

        match raw {
            RawClass::Unknown => {
                // Detection dropout: hold the last belief briefly (marked stale),
                // then decay to Loose. Don't touch candidate accumulation.
                if now - self.last_concrete_ts > cfg.hold_secs {
                    if self.stable != PossessionState::Loose {
                        self.commit(PossessionState::Loose);
                    }
                } else if self.stable != PossessionState::Loose {
                    self.stale = true;
                }
            }
            RawClass::Owned { who, breakbeam } => {
                self.last_concrete_ts = now;
                let suppressed = !breakbeam
                    && self.released_by == Some(who)
                    && now - self.released_at < cfg.release_suppress_secs;
                if suppressed {
                    self.accumulate(PossessionState::Loose, false, cfg);
                } else {
                    self.accumulate(PossessionState::Owned { owner: who }, breakbeam, cfg);
                }
            }
            RawClass::Contested(candidates) => {
                self.last_concrete_ts = now;
                self.accumulate(PossessionState::Contested { candidates }, false, cfg);
            }
            RawClass::Loose => {
                self.last_concrete_ts = now;
                self.accumulate(PossessionState::Loose, false, cfg);
            }
        }

        self.possession()
    }

    /// Memoryless classification, hysteresis-aware via the current stable owner
    /// (who gets the looser `release_dist` retain band).
    fn classify(
        &self,
        ball: Option<&BallData>,
        blue: &[PlayerData],
        yellow: &[PlayerData],
        controlled_blue: bool,
        controlled_yellow: bool,
        cfg: &PossessionConfig,
    ) -> RawClass {
        let ball = match ball {
            Some(b) if b.detected => b,
            _ => return RawClass::Unknown,
        };
        let ball_pos = ball.position.xy();
        let ball_speed = ball.velocity.xy().norm();

        // (id, distance-to-ball, breakbeam, controlled) for every robot.
        let cands: Vec<(TeamPlayerId, f64, bool, bool)> = blue
            .iter()
            .map(|p| (TeamColor::Blue, controlled_blue, p))
            .chain(
                yellow
                    .iter()
                    .map(|p| (TeamColor::Yellow, controlled_yellow, p)),
            )
            .map(|(team_color, controlled, p)| {
                (
                    TeamPlayerId {
                        team_color,
                        player_id: p.id,
                    },
                    (p.position - ball_pos).norm(),
                    p.breakbeam_ball_detected,
                    controlled,
                )
            })
            .collect();

        // Hard evidence: breakbeam, gated on the ball being near the dribbler.
        let breakbeam_owner = cands
            .iter()
            .filter(|(_, dist, bb, _)| *bb && *dist <= cfg.breakbeam_gate_dist)
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((who, _, _, _)) = breakbeam_owner {
            return RawClass::Owned {
                who: *who,
                breakbeam: true,
            };
        }

        // A fast ball can't be possessed by proximity (only breakbeam, above).
        if ball_speed >= cfg.max_ball_speed {
            return RawClass::Loose;
        }

        // Soft evidence: proximity, but *only* for robots we don't control —
        // controlled robots are credited solely via breakbeam (above). The
        // stable owner is retained out to the looser `release_dist` band
        // (hysteresis against oscillation).
        let stable_owner = self.stable.owner();
        let mut in_control: Vec<(TeamPlayerId, f64)> = cands
            .iter()
            .filter(|(_, _, _, controlled)| !*controlled)
            .filter(|(who, dist, _, _)| {
                let threshold = if Some(*who) == stable_owner {
                    cfg.release_dist
                } else {
                    cfg.acquire_dist
                };
                *dist < threshold
            })
            .map(|(who, dist, _, _)| (*who, *dist))
            .collect();
        in_control.sort_by(|a, b| a.1.total_cmp(&b.1));

        let (closest, closest_dist) = match in_control.first() {
            Some(&(who, dist)) => (who, dist),
            None => return RawClass::Loose,
        };

        // Everyone within the ambiguity margin of the closest is a co-candidate.
        let mut candidates: Vec<TeamPlayerId> = in_control
            .iter()
            .filter(|(_, dist)| dist - closest_dist < cfg.ambiguity_margin)
            .map(|(who, _)| *who)
            .collect();
        if candidates.len() >= 2 {
            candidates.sort_by_key(order_key);
            RawClass::Contested(candidates)
        } else {
            RawClass::Owned {
                who: closest,
                breakbeam: false,
            }
        }
    }

    /// Accumulate evidence toward `observed`; commit once stable enough. `fast`
    /// (breakbeam-confirmed gain) commits in a single frame.
    fn accumulate(&mut self, observed: PossessionState, fast: bool, cfg: &PossessionConfig) {
        if observed == self.stable {
            self.candidate = None;
            self.candidate_count = 0;
            return;
        }
        if self.candidate.as_ref() == Some(&observed) {
            self.candidate_count += 1;
        } else {
            self.candidate = Some(observed.clone());
            self.candidate_count = 1;
        }
        let needed = if fast { 1 } else { cfg.acquire_frames };
        if self.candidate_count >= needed {
            self.commit(observed);
        }
    }

    fn commit(&mut self, new: PossessionState) {
        self.stable = new;
        self.candidate = None;
        self.candidate_count = 0;
    }
}
