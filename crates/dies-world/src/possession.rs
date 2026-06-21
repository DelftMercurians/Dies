//! Unified ball-possession tracker — the single place possession is decided.
//!
//! One latched signal is fused from two evidence sources per robot:
//!   * **breakbeam** (own robots only), *gated* on the ball actually being near
//!     the dribbler — a breakbeam trigger with the ball nowhere near is rejected;
//!   * **proximity** of the ball to the robot, with a two-band hysteresis (gain
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
    Owned { who: TeamPlayerId, breakbeam: bool },
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

        let raw = self.classify(ball, blue, yellow, cfg);
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
        cfg: &PossessionConfig,
    ) -> RawClass {
        let ball = match ball {
            Some(b) if b.detected => b,
            _ => return RawClass::Unknown,
        };
        let ball_pos = ball.position.xy();
        let ball_speed = ball.velocity.xy().norm();

        // (id, distance-to-ball, breakbeam) for every robot on the field.
        let cands: Vec<(TeamPlayerId, f64, bool)> = blue
            .iter()
            .map(|p| (TeamColor::Blue, p))
            .chain(yellow.iter().map(|p| (TeamColor::Yellow, p)))
            .map(|(team_color, p)| {
                (
                    TeamPlayerId {
                        team_color,
                        player_id: p.id,
                    },
                    (p.position - ball_pos).norm(),
                    p.breakbeam_ball_detected,
                )
            })
            .collect();

        // Hard evidence: breakbeam, gated on the ball being near the dribbler.
        let breakbeam_owner = cands
            .iter()
            .filter(|(_, dist, bb)| *bb && *dist <= cfg.breakbeam_gate_dist)
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((who, _, _)) = breakbeam_owner {
            return RawClass::Owned {
                who: *who,
                breakbeam: true,
            };
        }

        // A fast ball can't be possessed by proximity (only breakbeam, above).
        if ball_speed >= cfg.max_ball_speed {
            return RawClass::Loose;
        }

        // Soft evidence: proximity, with the stable owner retained out to the
        // looser `release_dist` band (hysteresis against oscillation).
        let stable_owner = self.stable.owner();
        let mut in_control: Vec<(TeamPlayerId, f64)> = cands
            .iter()
            .filter(|(who, dist, _)| {
                let threshold = if Some(*who) == stable_owner {
                    cfg.release_dist
                } else {
                    cfg.acquire_dist
                };
                *dist < threshold
            })
            .map(|(who, dist, _)| (*who, *dist))
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

#[cfg(test)]
mod tests {
    use super::*;
    use dies_core::{PlayerId, Vector3};

    fn pid(n: u32) -> PlayerId {
        PlayerId::new(n)
    }

    fn tpid(team: TeamColor, n: u32) -> TeamPlayerId {
        TeamPlayerId {
            team_color: team,
            player_id: pid(n),
        }
    }

    fn player(id: u32, pos: Vector2, breakbeam: bool) -> PlayerData {
        let mut p = PlayerData::new(pid(id));
        p.position = pos;
        p.breakbeam_ball_detected = breakbeam;
        p
    }

    fn ball(pos: Vector2, vel: Vector2) -> BallData {
        BallData {
            timestamp: 0.0,
            position: Vector3::new(pos.x, pos.y, 0.0),
            raw_position: vec![],
            velocity: Vector3::new(vel.x, vel.y, 0.0),
            detected: true,
        }
    }

    fn cfg() -> PossessionConfig {
        PossessionConfig::default()
    }

    /// One blue robot sitting on the ball with breakbeam lit.
    fn one_blue_breakbeam() -> (Vec<PlayerData>, BallData) {
        let b = ball(Vector2::new(0.0, 0.0), Vector2::zeros());
        let p = player(1, Vector2::new(50.0, 0.0), true);
        (vec![p], b)
    }

    #[test]
    fn breakbeam_gain_commits_in_one_frame() {
        let mut t = PossessionTracker::new();
        let (blue, b) = one_blue_breakbeam();
        let p = t.update(Some(&b), &blue, &[], &cfg(), 0.0);
        assert_eq!(p.state, PossessionState::Owned { owner: tpid(TeamColor::Blue, 1) });
        assert!(!p.stale);
    }

    #[test]
    fn breakbeam_with_ball_far_is_rejected() {
        // Breakbeam lit, but the ball is well beyond the gate distance → ignored.
        let mut t = PossessionTracker::new();
        let b = ball(Vector2::new(1000.0, 0.0), Vector2::zeros());
        let p = player(1, Vector2::new(0.0, 0.0), true);
        let res = t.update(Some(&b), &[p], &[], &cfg(), 0.0);
        assert_eq!(res.state, PossessionState::Loose);
    }

    #[test]
    fn single_frame_blip_is_masked() {
        let mut t = PossessionTracker::new();
        let (blue, b) = one_blue_breakbeam();
        t.update(Some(&b), &blue, &[], &cfg(), 0.0);
        // One frame with the ball gone far (loss) must NOT flip us yet.
        let b2 = ball(Vector2::new(2000.0, 0.0), Vector2::zeros());
        let blue2 = vec![player(1, Vector2::new(50.0, 0.0), false)];
        let p = t.update(Some(&b2), &blue2, &[], &cfg(), 0.016);
        assert_eq!(p.state, PossessionState::Owned { owner: tpid(TeamColor::Blue, 1) });
    }

    #[test]
    fn sustained_loss_commits_after_debounce() {
        let mut t = PossessionTracker::new();
        let (blue, b) = one_blue_breakbeam();
        t.update(Some(&b), &blue, &[], &cfg(), 0.0);
        let b2 = ball(Vector2::new(2000.0, 0.0), Vector2::zeros());
        let blue2 = vec![player(1, Vector2::new(50.0, 0.0), false)];
        let mut last = t.possession().state;
        for i in 0..cfg().acquire_frames {
            last = t.update(Some(&b2), &blue2, &[], &cfg(), 0.1 + i as f64 * 0.016).state;
        }
        assert_eq!(last, PossessionState::Loose);
    }

    #[test]
    fn dropout_holds_then_stale_then_loose() {
        let mut t = PossessionTracker::new();
        let (blue, b) = one_blue_breakbeam();
        t.update(Some(&b), &blue, &[], &cfg(), 0.0);
        // Within the hold window with no ball: keep owner, marked stale.
        let held = t.update(None, &blue, &[], &cfg(), 0.1);
        assert_eq!(held.state, PossessionState::Owned { owner: tpid(TeamColor::Blue, 1) });
        assert!(held.stale);
        // Past the hold window: decay to Loose.
        let gone = t.update(None, &blue, &[], &cfg(), 1.0);
        assert_eq!(gone.state, PossessionState::Loose);
    }

    #[test]
    fn fast_ball_near_robot_is_not_possessed() {
        let mut t = PossessionTracker::new();
        let b = ball(Vector2::new(0.0, 0.0), Vector2::new(3000.0, 0.0));
        let p = player(1, Vector2::new(50.0, 0.0), false);
        let res = t.update(Some(&b), &[p], &[], &cfg(), 0.0);
        assert_eq!(res.state, PossessionState::Loose);
    }

    #[test]
    fn slow_ball_near_robot_is_owned_by_proximity() {
        let mut t = PossessionTracker::new();
        let b = ball(Vector2::new(0.0, 0.0), Vector2::new(50.0, 0.0));
        let blue = vec![player(1, Vector2::new(50.0, 0.0), false)];
        // Proximity gain needs the debounce frames.
        let mut last = PossessionState::Loose;
        for i in 0..cfg().acquire_frames {
            last = t.update(Some(&b), &blue, &[], &cfg(), i as f64 * 0.016).state;
        }
        assert_eq!(last, PossessionState::Owned { owner: tpid(TeamColor::Blue, 1) });
    }

    #[test]
    fn opponent_proximity_is_owned_opp() {
        let mut t = PossessionTracker::new();
        let b = ball(Vector2::new(0.0, 0.0), Vector2::zeros());
        let yellow = vec![player(3, Vector2::new(40.0, 0.0), false)];
        let mut last = PossessionState::Loose;
        for i in 0..cfg().acquire_frames {
            last = t.update(Some(&b), &[], &yellow, &cfg(), i as f64 * 0.016).state;
        }
        assert_eq!(last, PossessionState::Owned { owner: tpid(TeamColor::Yellow, 3) });
    }

    #[test]
    fn two_robots_equidistant_is_contested() {
        let mut t = PossessionTracker::new();
        let b = ball(Vector2::new(0.0, 0.0), Vector2::zeros());
        let blue = vec![player(1, Vector2::new(40.0, 0.0), false)];
        let yellow = vec![player(2, Vector2::new(-40.0, 0.0), false)];
        let mut last = PossessionState::Loose;
        for i in 0..cfg().acquire_frames {
            last = t.update(Some(&b), &blue, &yellow, &cfg(), i as f64 * 0.016).state;
        }
        assert_eq!(
            last,
            PossessionState::Contested {
                candidates: vec![tpid(TeamColor::Blue, 1), tpid(TeamColor::Yellow, 2)],
            }
        );
    }

    #[test]
    fn release_suppresses_proximity_but_not_breakbeam() {
        let mut t = PossessionTracker::new();
        let (blue, b) = one_blue_breakbeam();
        t.update(Some(&b), &blue, &[], &cfg(), 0.0);
        // Command a kick; next update drops us to Loose.
        t.notify_kick(tpid(TeamColor::Blue, 1));
        let near = ball(Vector2::new(0.0, 0.0), Vector2::zeros());
        let blue_prox = vec![player(1, Vector2::new(50.0, 0.0), false)];
        let dropped = t.update(Some(&near), &blue_prox, &[], &cfg(), 0.1);
        assert_eq!(dropped.state, PossessionState::Loose);

        // Proximity re-acquire by the kicker within the window stays suppressed.
        for i in 0..(cfg().acquire_frames + 2) {
            let p = t.update(Some(&near), &blue_prox, &[], &cfg(), 0.1 + i as f64 * 0.005);
            assert_eq!(p.state, PossessionState::Loose);
        }
        // But a breakbeam re-acquire (ball never left) overrides immediately.
        let blue_bb = vec![player(1, Vector2::new(50.0, 0.0), true)];
        let p = t.update(Some(&near), &blue_bb, &[], &cfg(), 0.13);
        assert_eq!(p.state, PossessionState::Owned { owner: tpid(TeamColor::Blue, 1) });
    }

    #[test]
    fn suppression_expires() {
        let mut t = PossessionTracker::new();
        let near = ball(Vector2::new(0.0, 0.0), Vector2::zeros());
        let blue_prox = vec![player(1, Vector2::new(50.0, 0.0), false)];
        t.notify_kick(tpid(TeamColor::Blue, 1));
        // First update consumes the kick, stamping the suppress window at t=1.0.
        t.update(Some(&near), &blue_prox, &[], &cfg(), 1.0);
        // Past the suppress window, proximity re-acquires (debounced as normal).
        let mut last = PossessionState::Loose;
        for i in 0..cfg().acquire_frames {
            last = t
                .update(Some(&near), &blue_prox, &[], &cfg(), 1.25 + i as f64 * 0.016)
                .state;
        }
        assert_eq!(last, PossessionState::Owned { owner: tpid(TeamColor::Blue, 1) });
    }
}
