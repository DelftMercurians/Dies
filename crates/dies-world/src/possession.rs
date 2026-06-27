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
    BallContest, BallData, PlayerData, PossessionConfig, PossessionState, TeamColor, TeamPlayerId,
    Vector2,
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

/// Memoryless contest observation from the per-robot candidate list: at least one
/// robot from *each* team within `contest_dist` of a ball slower than
/// `contest_speed_max`. `None` otherwise. The near set spans both teams.
fn classify_contest(
    cands: &[(TeamPlayerId, f64, bool, bool)],
    ball_speed: f64,
    cfg: &PossessionConfig,
) -> Option<BallContest> {
    if ball_speed >= cfg.contest_speed_max {
        return None;
    }
    let near: Vec<TeamPlayerId> = cands
        .iter()
        .filter(|(_, dist, _, _)| *dist <= cfg.contest_dist)
        .map(|(who, _, _, _)| *who)
        .collect();
    let has_blue = near.iter().any(|w| w.team_color == TeamColor::Blue);
    let has_yellow = near.iter().any(|w| w.team_color == TeamColor::Yellow);
    if has_blue && has_yellow {
        Some(BallContest { near })
    } else {
        None
    }
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
    /// Committed contest belief — a *separate* debounce track from ownership, so
    /// it never perturbs the `accumulate`/`commit` state machine above.
    contest_stable: Option<BallContest>,
    /// Consecutive frames the live contest observation has disagreed with
    /// `contest_stable` (drives the enter/exit hysteresis).
    contest_count: u32,
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
            contest_stable: None,
            contest_count: 0,
        }
    }

    /// Current stable possession with its staleness flag.
    pub fn possession(&self) -> dies_core::Possession {
        dies_core::Possession {
            state: self.stable.clone(),
            stale: self.stale,
            contest: self.contest_stable.clone(),
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

        let (raw, contest_obs) =
            self.classify(ball, blue, yellow, controlled_blue, controlled_yellow, cfg);
        self.stale = false;

        // Contest is a separate, parallel track (orthogonal to ownership). On a
        // detection dropout we clear it outright — a contest is a live geometric
        // fact, not something to coast.
        if matches!(raw, RawClass::Unknown) {
            self.contest_stable = None;
            self.contest_count = 0;
        } else {
            self.update_contest(contest_obs, cfg);
        }

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
    /// (who gets the looser `release_dist` retain band). Also returns the raw,
    /// memoryless contest observation (debounced separately by the caller).
    fn classify(
        &self,
        ball: Option<&BallData>,
        blue: &[PlayerData],
        yellow: &[PlayerData],
        controlled_blue: bool,
        controlled_yellow: bool,
        cfg: &PossessionConfig,
    ) -> (RawClass, Option<BallContest>) {
        let ball = match ball {
            Some(b) if b.detected => b,
            _ => return (RawClass::Unknown, None),
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

        // Contest (orthogonal to ownership): both teams crowding a slow ball.
        let contest = classify_contest(&cands, ball_speed, cfg);

        // Hard evidence: breakbeam, gated on the ball being near the dribbler.
        let breakbeam_owner = cands
            .iter()
            .filter(|(_, dist, bb, _)| *bb && *dist <= cfg.breakbeam_gate_dist)
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((who, _, _, _)) = breakbeam_owner {
            return (
                RawClass::Owned {
                    who: *who,
                    breakbeam: true,
                },
                contest,
            );
        }

        // A fast ball can't be possessed by proximity (only breakbeam, above).
        if ball_speed >= cfg.max_ball_speed {
            return (RawClass::Loose, contest);
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
            None => return (RawClass::Loose, contest),
        };

        // Everyone within the ambiguity margin of the closest is a co-candidate.
        let mut candidates: Vec<TeamPlayerId> = in_control
            .iter()
            .filter(|(_, dist)| dist - closest_dist < cfg.ambiguity_margin)
            .map(|(who, _)| *who)
            .collect();
        let raw = if candidates.len() >= 2 {
            candidates.sort_by_key(order_key);
            RawClass::Contested(candidates)
        } else {
            RawClass::Owned {
                who: closest,
                breakbeam: false,
            }
        };
        (raw, contest)
    }

    /// Debounce the raw contest observation into `contest_stable` with asymmetric
    /// enter/exit hysteresis. While active, the member list is refreshed every
    /// frame (only the active/inactive *edge* is debounced, not set membership —
    /// that would thrash as opponent positions jitter).
    fn update_contest(&mut self, observed: Option<BallContest>, cfg: &PossessionConfig) {
        let active_now = observed.is_some();
        let active_before = self.contest_stable.is_some();
        if active_now == active_before {
            // No edge pending: reset the counter and refresh members if active.
            self.contest_count = 0;
            if observed.is_some() {
                self.contest_stable = observed;
            }
            return;
        }
        // An edge is pending; commit once it has persisted long enough.
        self.contest_count += 1;
        let needed = if active_now {
            cfg.contest_enter_frames
        } else {
            cfg.contest_exit_frames
        };
        if self.contest_count >= needed {
            self.contest_stable = observed;
            self.contest_count = 0;
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

    fn ball(speed_x: f64) -> BallData {
        BallData {
            timestamp: 0.0,
            position: Vector3::zeros(),
            raw_position: vec![],
            velocity: Vector3::new(speed_x, 0.0, 0.0),
            detected: true,
        }
    }

    fn player(id: u32, x: f64, y: f64, breakbeam: bool) -> PlayerData {
        let mut p = PlayerData::new(PlayerId::new(id));
        p.position = Vector2::new(x, y);
        p.breakbeam_ball_detected = breakbeam;
        p
    }

    fn contest_teams(c: &BallContest) -> (bool, bool) {
        (
            c.near.iter().any(|w| w.team_color == TeamColor::Blue),
            c.near.iter().any(|w| w.team_color == TeamColor::Yellow),
        )
    }

    /// The headline case: our (blue, controlled) robot holds the ball via breakbeam
    /// while an opponent presses from close range on a near-stationary ball. Owner
    /// stays us, *and* the contest signal fires — the deadlock the strategy was
    /// previously blind to.
    #[test]
    fn breakbeam_owner_under_opponent_press_is_owned_and_contested() {
        let cfg = PossessionConfig::default();
        let mut t = PossessionTracker::new();
        let blue = vec![player(1, 0.0, 0.0, true)];
        let yellow = vec![player(2, 200.0, 0.0, false)]; // within contest_dist (220)
        let b = ball(0.0);

        // Breakbeam owns in one frame; contest needs `contest_enter_frames`.
        let mut poss = dies_core::Possession::default();
        for i in 0..cfg.contest_enter_frames {
            poss = t.update(
                Some(&b),
                &blue,
                &yellow,
                true,
                false,
                &cfg,
                i as f64 * 0.016,
            );
        }

        match poss.state {
            PossessionState::Owned { owner } => {
                assert_eq!(owner.team_color, TeamColor::Blue);
                assert_eq!(owner.player_id, PlayerId::new(1));
            }
            other => panic!("expected Owned by us, got {other:?}"),
        }
        let contest = poss.contest.expect("contest should fire");
        assert_eq!(contest_teams(&contest), (true, true), "both teams crowding");
    }

    /// Carrying the ball in open space (no opponent near) is not a contest.
    #[test]
    fn clean_possession_is_not_contested() {
        let cfg = PossessionConfig::default();
        let mut t = PossessionTracker::new();
        let blue = vec![player(1, 0.0, 0.0, true)];
        let yellow = vec![player(2, 3000.0, 0.0, false)]; // far away
        let b = ball(0.0);

        let mut poss = dies_core::Possession::default();
        for i in 0..(cfg.contest_enter_frames + 2) {
            poss = t.update(
                Some(&b),
                &blue,
                &yellow,
                true,
                false,
                &cfg,
                i as f64 * 0.016,
            );
        }
        assert!(poss.contest.is_none(), "no opponent near → no contest");
    }

    /// A fast ball can't be a static pin, so it never reads as contested even with
    /// both teams nearby.
    #[test]
    fn fast_ball_is_not_contested() {
        let cfg = PossessionConfig::default();
        let mut t = PossessionTracker::new();
        let blue = vec![player(1, 0.0, 0.0, false)];
        let yellow = vec![player(2, 150.0, 0.0, false)];
        let b = ball(cfg.contest_speed_max + 100.0); // above the slow-ball gate

        let mut poss = dies_core::Possession::default();
        for i in 0..(cfg.contest_enter_frames + 2) {
            poss = t.update(
                Some(&b),
                &blue,
                &yellow,
                true,
                false,
                &cfg,
                i as f64 * 0.016,
            );
        }
        assert!(poss.contest.is_none(), "fast ball → no contest");
    }

    /// Enter needs `contest_enter_frames`; exit needs the (longer) `contest_exit_frames`.
    #[test]
    fn contest_debounce_respects_enter_and_exit_frames() {
        let cfg = PossessionConfig::default();
        let mut t = PossessionTracker::new();
        let blue = vec![player(1, 0.0, 0.0, true)];
        let near = vec![player(2, 200.0, 0.0, false)];
        let far = vec![player(2, 3000.0, 0.0, false)];
        let b = ball(0.0);
        let mut now = 0.0;
        let mut tick = |t: &mut PossessionTracker, yellow: &[PlayerData]| {
            now += 0.016;
            t.update(Some(&b), &blue, yellow, true, false, &cfg, now)
        };

        // Enter: not contested until the enter threshold.
        for _ in 0..(cfg.contest_enter_frames - 1) {
            assert!(tick(&mut t, &near).contest.is_none());
        }
        assert!(
            tick(&mut t, &near).contest.is_some(),
            "commits at enter_frames"
        );

        // Exit: opponent leaves, but the belief is held for exit_frames.
        for _ in 0..(cfg.contest_exit_frames - 1) {
            assert!(
                tick(&mut t, &far).contest.is_some(),
                "held during exit window"
            );
        }
        assert!(
            tick(&mut t, &far).contest.is_none(),
            "clears at exit_frames"
        );
    }
}
