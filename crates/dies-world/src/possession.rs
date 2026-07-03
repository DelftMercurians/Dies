//! Unified ball-possession tracker — the single place possession is decided.
//!
//! Evidence sources depend on whether the robot is one we control:
//!   * **controlled robots** (our own team) use *only* their **breakbeam**,
//!     *gated* on the ball actually being near the dribbler — a breakbeam trigger
//!     with the ball nowhere near is rejected. Proximity never contributes for a
//!     controlled robot, so a controlled robot that merely brushes the ball is
//!     never credited with possession.
//!   * **uncontrolled robots** (opponents, where we have no breakbeam) fall back
//!     to a **dribbler-point hold test**: the ball within a band of the robot's
//!     dribbler mouth (centre + `dribbler_offset` along yaw), gained only while
//!     also moving *with* the robot (relative speed below
//!     `prox_acquire_rel_speed`), with a two-band spatial hysteresis (gain at
//!     `prox_enter_dist`, retain until the ball leaves `prox_stay_dist`).
//!
//! Edges are confidence-driven and asymmetric: a breakbeam-backed gain commits in
//! one frame, a proximity-only gain must persist `acquire_frames`, and a proximity
//! owner is only released after `release_frames` (a genuine holder's ball jitters
//! out of band constantly) — unless the ball departs at kick speed
//! (`prox_kick_rel_speed`), which releases in one frame. A vision dropout holds
//! the last belief (marked `stale`) for `hold_secs` — or `owned_coast_secs` when
//! someone owns the ball, because a held ball is *usually* occluded by its holder
//! (measured ~55% of held frames on real logs) — while a controlled robot's
//! breakbeam keeps counting as fresh evidence right through the dropout. Multiple
//! equally-plausible owners surface as `Contested`. A commanded kick (efference
//! copy) drops the kicker and suppresses its *proximity* re-acquisition briefly —
//! breakbeam re-acquisition is never suppressed, so a misfire where the ball
//! never left self-corrects within a frame.
//!
//! The proximity thresholds are tuned on breakbeam-labeled real-hardware logs
//! (2026-07-03, `.analysis/possession_proto.py`). Do not re-tune them in sim: the
//! sim dribble model holds the ball perfectly and will validate any values.

use std::collections::HashMap;

use dies_core::{
    BallContest, BallData, PlayerData, PossessionConfig, PossessionState, TeamColor, TeamPlayerId,
    Vector2,
};

/// Per-robot latch state for the ToF-backup breakbeam substitute — an asymmetric
/// enter/exit frame debounce (a time-domain Schmitt trigger) over the raw ToF
/// detection condition. Only tracked for controlled robots whose per-robot
/// toggle is on.
#[derive(Debug, Clone, Copy, Default)]
struct TofBackupState {
    /// Latched output: the ToF signal currently substituting for the breakbeam.
    on: bool,
    /// Consecutive frames the raw condition has disagreed with `on` (drives the
    /// enter/exit hysteresis).
    count: u32,
}

/// The instantaneous (memoryless) ToF-backup detection condition for a robot:
/// `confidence > c_th && |x| < x_th && y < y_th`, all in raw sensor units.
/// Fail-safe: a missing confidence byte or missing position yields `false`, so
/// an enabled robot never spuriously claims the ball on absent telemetry.
fn tof_backup_raw(p: &PlayerData, cfg: &PossessionConfig) -> bool {
    let conf_ok = p
        .tof_confidence
        .is_some_and(|c| c as f64 > cfg.tof_backup_conf_th);
    let geom_ok = p.tof_xy.is_some_and(|[x, y]| {
        (x as f64).abs() < cfg.tof_backup_x_th && (y as f64) < cfg.tof_backup_y_th
    });
    conf_ok && geom_ok
}

/// Raw, memoryless classification for a single frame.
#[derive(Debug, Clone, PartialEq)]
enum RawClass {
    Owned {
        who: TeamPlayerId,
        breakbeam: bool,
    },
    Contested(Vec<TeamPlayerId>),
    Loose {
        /// Kick signature on the stable owner (ball left the retain band at
        /// `prox_kick_rel_speed`): commit the loss in one frame instead of
        /// waiting out `release_frames`.
        fast: bool,
    },
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
    cands: &[(TeamPlayerId, f64, bool)],
    ball_speed: f64,
    cfg: &PossessionConfig,
) -> Option<BallContest> {
    if ball_speed >= cfg.contest_speed_max {
        return None;
    }
    let near: Vec<TeamPlayerId> = cands
        .iter()
        .filter(|(_, dist, _)| *dist <= cfg.contest_dist)
        .map(|(who, _, _)| *who)
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
    /// Per-robot ToF-backup breakbeam latch state, keyed by robot. Entries exist
    /// only for controlled robots with the toggle currently on; pruned otherwise.
    tof_backup: HashMap<TeamPlayerId, TofBackupState>,
    /// Ball position from the last frame it was actually detected. Gates
    /// breakbeam evidence while the ball is occluded (vision can't confirm or
    /// deny, but a beam trip next to where the ball vanished is a real hold).
    last_ball_pos: Option<Vector2>,
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
            tof_backup: HashMap::new(),
            last_ball_pos: None,
        }
    }

    /// Latched ToF-backup detection for a robot (false if it has no toggle / no
    /// entry). Used to stamp `PlayerData::tof_backup_ball_detected` for logging.
    pub fn tof_backup_detected(&self, who: TeamPlayerId) -> bool {
        self.tof_backup.get(&who).map(|s| s.on).unwrap_or(false)
    }

    /// Advance the per-robot ToF-backup Schmitt latches by one frame. Runs before
    /// `classify`, which reads the latched output as the effective breakbeam for
    /// toggled robots. Only controlled robots with the toggle on are tracked;
    /// others are pruned so a later toggle-off can't leave a stale latch.
    fn advance_tof_backup(
        &mut self,
        blue: &[PlayerData],
        yellow: &[PlayerData],
        controlled_blue: bool,
        controlled_yellow: bool,
        cfg: &PossessionConfig,
    ) {
        for (team_color, controlled, players) in [
            (TeamColor::Blue, controlled_blue, blue),
            (TeamColor::Yellow, controlled_yellow, yellow),
        ] {
            for p in players {
                let who = TeamPlayerId {
                    team_color,
                    player_id: p.id,
                };
                if !(controlled && p.tof_backup_enabled) {
                    self.tof_backup.remove(&who);
                    continue;
                }
                let raw = tof_backup_raw(p, cfg);
                let state = self.tof_backup.entry(who).or_default();
                if raw == state.on {
                    // Agreement resets the disagreement counter.
                    state.count = 0;
                } else {
                    // Persist the disagreeing edge for the asymmetric threshold
                    // (fast to acquire, slow to release) before flipping.
                    state.count += 1;
                    let need = if state.on {
                        cfg.tof_backup_exit_frames
                    } else {
                        cfg.tof_backup_enter_frames
                    };
                    if state.count >= need {
                        state.on = raw;
                        state.count = 0;
                    }
                }
            }
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

        if let Some(b) = ball {
            if b.detected {
                self.last_ball_pos = Some(b.position.xy());
            }
        }

        // Advance the ToF-backup Schmitt latches first so `classify` sees this
        // frame's effective breakbeam for toggled robots.
        self.advance_tof_backup(blue, yellow, controlled_blue, controlled_yellow, cfg);

        let (raw, contest_obs) =
            self.classify(ball, blue, yellow, controlled_blue, controlled_yellow, cfg);
        self.stale = false;

        // Whether losing the current stable owner takes the slow `release_frames`
        // debounce: only proximity-credited (uncontrolled) owners get it — a
        // controlled owner's breakbeam loss is authoritative.
        let stable_prox_owner = self.stable.owner().is_some_and(|o| {
            let controlled = match o.team_color {
                TeamColor::Blue => controlled_blue,
                TeamColor::Yellow => controlled_yellow,
            };
            !controlled
        });

        // Contest is a separate, parallel track (orthogonal to ownership). On a
        // detection dropout we clear it outright — a contest is a live geometric
        // fact, not something to coast.
        if matches!(raw, RawClass::Unknown) {
            self.contest_stable = None;
            self.contest_count = 0;
        } else {
            self.update_contest(contest_obs, cfg);
        }

        // Frames a change away from the stable belief must persist. Losing a
        // proximity owner is deliberately slower than any gain (the holder's ball
        // jitters out of band constantly on real vision).
        let change_frames = if stable_prox_owner {
            cfg.release_frames
        } else {
            cfg.acquire_frames
        };

        match raw {
            RawClass::Unknown => {
                // Detection dropout: hold the last belief (marked stale), then
                // decay to Loose. An `Owned` belief coasts far longer — a held
                // ball is usually occluded *by its holder*, so the dropout is
                // weak evidence for the hold, not against it. Don't touch
                // candidate accumulation.
                let coast = if matches!(self.stable, PossessionState::Owned { .. }) {
                    cfg.owned_coast_secs
                } else {
                    cfg.hold_secs
                };
                if now - self.last_concrete_ts > coast {
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
                    self.accumulate(PossessionState::Loose, change_frames);
                } else {
                    // A breakbeam-confirmed gain commits in one frame; a
                    // proximity gain (incl. owner→owner handoff) needs
                    // `acquire_frames` — the new owner's enter band is strong
                    // evidence, so the slow release doesn't apply.
                    let needed = if breakbeam { 1 } else { cfg.acquire_frames };
                    self.accumulate(PossessionState::Owned { owner: who }, needed);
                }
            }
            RawClass::Contested(candidates) => {
                self.last_concrete_ts = now;
                self.accumulate(PossessionState::Contested { candidates }, change_frames);
            }
            RawClass::Loose { fast } => {
                self.last_concrete_ts = now;
                let needed = if fast { 1 } else { change_frames };
                self.accumulate(PossessionState::Loose, needed);
            }
        }

        self.possession()
    }

    /// Memoryless classification, hysteresis-aware via the current stable owner
    /// (who gets the looser `prox_stay_dist` retain band). Also returns the raw,
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
            _ => {
                // Ball undetected: vision can neither confirm nor deny — but the
                // breakbeam is vision-independent. A controlled robot tripping it
                // still owns, *fresh* (not coasted), provided the trip is
                // plausible: the robot already is the stable owner (mid-carry —
                // the holder itself occludes the ball), or the ball was last
                // seen within the usual breakbeam gate of it.
                let claimant = [
                    (TeamColor::Blue, controlled_blue, blue),
                    (TeamColor::Yellow, controlled_yellow, yellow),
                ]
                .into_iter()
                .filter(|(_, controlled, _)| *controlled)
                .flat_map(|(team_color, _, players)| {
                    players
                        .iter()
                        .filter(|p| p.sideline.is_none())
                        .map(move |p| (team_color, p))
                })
                .filter_map(|(team_color, p)| {
                    let who = TeamPlayerId {
                        team_color,
                        player_id: p.id,
                    };
                    let bb = if p.tof_backup_enabled {
                        self.tof_backup_detected(who)
                    } else {
                        p.breakbeam_ball_detected
                    };
                    if !bb {
                        return None;
                    }
                    let plausible = self.stable.owner() == Some(who)
                        || self
                            .last_ball_pos
                            .is_some_and(|b| (p.position - b).norm() <= cfg.breakbeam_gate_dist);
                    plausible.then_some(who)
                })
                .next();
                return match claimant {
                    Some(who) => (
                        RawClass::Owned {
                            who,
                            breakbeam: true,
                        },
                        None,
                    ),
                    None => (RawClass::Unknown, None),
                };
            }
        };
        let ball_pos = ball.position.xy();
        let ball_vel = ball.velocity.xy();
        let ball_speed = ball_vel.norm();

        // (id, distance-to-ball, breakbeam) for every robot.
        // Sidelined robots (radio-lost / card-removed) can't meaningfully own the
        // ball — we can't act on it — so they're excluded from ownership.
        let cands: Vec<(TeamPlayerId, f64, bool)> = blue
            .iter()
            .filter(|p| p.sideline.is_none())
            .map(|p| (TeamColor::Blue, p))
            .chain(
                yellow
                    .iter()
                    .filter(|p| p.sideline.is_none())
                    .map(|p| (TeamColor::Yellow, p)),
            )
            .map(|(team_color, p)| {
                let who = TeamPlayerId {
                    team_color,
                    player_id: p.id,
                };
                // ToF-backup replacement: when a robot's toggle is on, ignore its
                // hardware breakbeam entirely and use the latched ToF signal in
                // its place. Both still ride the `breakbeam_gate_dist` sanity gate
                // (applied below), so this behaves exactly like a breakbeam.
                let breakbeam = if p.tof_backup_enabled {
                    self.tof_backup_detected(who)
                } else {
                    p.breakbeam_ball_detected
                };
                (who, (p.position - ball_pos).norm(), breakbeam)
            })
            .collect();

        // Contest (orthogonal to ownership): both teams crowding a slow ball.
        let contest = classify_contest(&cands, ball_speed, cfg);

        // Hard evidence: breakbeam, gated on the ball being near the dribbler.
        let breakbeam_owner = cands
            .iter()
            .filter(|(_, dist, bb)| *bb && *dist <= cfg.breakbeam_gate_dist)
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((who, _, _)) = breakbeam_owner {
            return (
                RawClass::Owned {
                    who: *who,
                    breakbeam: true,
                },
                contest,
            );
        }

        // Soft evidence: the ball at the dribbler mouth, but *only* for robots we
        // don't control — controlled robots are credited solely via breakbeam
        // (above). Distances are to the *dribbler point* (centre +
        // `dribbler_offset` along yaw), which separates a held ball from one
        // merely beside the robot. Acquisition additionally requires the ball to
        // move *with* the robot; the stable owner is retained out to the looser
        // `prox_stay_dist` band with no speed gate (velocity-estimate lag spikes
        // during the holder's accelerations must not release it).
        let stable_owner = self.stable.owner();
        let mut owner_kicked = false;
        let mut in_control: Vec<(TeamPlayerId, f64)> = Vec::new();
        for (team_color, controlled, players) in [
            (TeamColor::Blue, controlled_blue, blue),
            (TeamColor::Yellow, controlled_yellow, yellow),
        ] {
            if controlled {
                continue;
            }
            for p in players.iter().filter(|p| p.sideline.is_none()) {
                let who = TeamPlayerId {
                    team_color,
                    player_id: p.id,
                };
                let dribbler_point = p.position + p.yaw.to_vector() * cfg.dribbler_offset;
                let dist = (ball_pos - dribbler_point).norm();
                let rel_speed = (ball_vel - p.velocity).norm();
                if Some(who) == stable_owner {
                    if dist < cfg.prox_stay_dist {
                        in_control.push((who, dist));
                    } else if rel_speed > cfg.prox_kick_rel_speed {
                        // Kick signature: the ball left the holder at speed.
                        owner_kicked = true;
                    }
                } else if dist < cfg.prox_enter_dist && rel_speed < cfg.prox_acquire_rel_speed {
                    in_control.push((who, dist));
                }
            }
        }
        in_control.sort_by(|a, b| a.1.total_cmp(&b.1));

        let (closest, closest_dist) = match in_control.first() {
            Some(&(who, dist)) => (who, dist),
            None => {
                return (
                    RawClass::Loose {
                        fast: owner_kicked,
                    },
                    contest,
                )
            }
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

    /// Accumulate evidence toward `observed`; commit once it has persisted
    /// `needed` frames. The caller picks `needed` per transition: 1 for
    /// authoritative evidence (breakbeam gain, kick-signature release),
    /// `acquire_frames` for gains, `release_frames` for losing a proximity owner.
    fn accumulate(&mut self, observed: PossessionState, needed: u32) {
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

    /// Ball at an arbitrary pose, `detected` explicit (false = vision dropout).
    fn ball_at(x: f64, y: f64, vx: f64, vy: f64, detected: bool) -> BallData {
        BallData {
            timestamp: 0.0,
            position: Vector3::new(x, y, 0.0),
            raw_position: vec![],
            velocity: Vector3::new(vx, vy, 0.0),
            detected,
        }
    }

    /// An uncontrolled (opponent-style) robot: yaw 0, so its dribbler point sits
    /// at `(x + dribbler_offset, y)`. Positioned at (-offset, 0) it "holds" a
    /// ball at the origin.
    fn opp(id: u32, x: f64, y: f64, vx: f64, vy: f64) -> PlayerData {
        let mut p = PlayerData::new(PlayerId::new(id));
        p.position = Vector2::new(x, y);
        p.velocity = Vector2::new(vx, vy);
        p
    }

    /// Drive a tracker with our (blue, controlled) side empty and one yellow
    /// (uncontrolled) robot — the pure proximity path.
    struct OppRig {
        t: PossessionTracker,
        cfg: PossessionConfig,
        now: f64,
    }

    impl OppRig {
        fn new() -> Self {
            Self {
                t: PossessionTracker::new(),
                cfg: PossessionConfig::default(),
                now: 0.0,
            }
        }

        fn tick(&mut self, ball: &BallData, yellow: &[PlayerData]) -> dies_core::Possession {
            self.now += 0.016;
            self.t
                .update(Some(ball), &[], yellow, true, false, &self.cfg, self.now)
        }

        fn tick_n(&mut self, n: u32, ball: &BallData, yellow: &[PlayerData]) -> dies_core::Possession {
            let mut p = dies_core::Possession::default();
            for _ in 0..n {
                p = self.tick(ball, yellow);
            }
            p
        }
    }

    fn owned_by(poss: &dies_core::Possession, id: u32) -> bool {
        poss.state.owner().map(|o| o.player_id) == Some(PlayerId::new(id))
    }

    /// The failure mode that motivated the redesign: an opponent dribbling at
    /// speed. The ball rides the dribbler point moving *with* the robot at
    /// 1.5 m/s — the old absolute ball-speed gate rejected this outright.
    #[test]
    fn moving_carrier_is_owned() {
        let mut rig = OppRig::new();
        let off = rig.cfg.dribbler_offset;
        let holder = vec![opp(1, -off, 0.0, 1500.0, 0.0)];
        let b = ball_at(0.0, 0.0, 1500.0, 0.0, true);
        let poss = rig.tick_n(rig.cfg.acquire_frames, &b, &holder);
        assert!(owned_by(&poss, 1), "co-moving carrier must own, got {:?}", poss.state);
    }

    /// A held ball jitters out of the retain band constantly on real vision (the
    /// holder pushes it ahead). Brief excursions must not release; a sustained
    /// one must, after `release_frames`.
    #[test]
    fn pushed_ahead_jitter_does_not_release() {
        let mut rig = OppRig::new();
        let off = rig.cfg.dribbler_offset;
        let holder = vec![opp(1, -off, 0.0, 0.0, 0.0)];
        let held = ball_at(0.0, 0.0, 0.0, 0.0, true);
        // Ball ~200mm past the dribbler point: outside prox_stay_dist, slow.
        let pushed = ball_at(200.0, 0.0, 0.0, 0.0, true);

        rig.tick_n(rig.cfg.acquire_frames, &held, &holder);
        let poss = rig.tick_n(rig.cfg.release_frames - 1, &pushed, &holder);
        assert!(owned_by(&poss, 1), "brief excursion released early: {:?}", poss.state);
        // One in-band frame resets the release debounce.
        assert!(owned_by(&rig.tick(&held, &holder), 1));
        let poss = rig.tick_n(rig.cfg.release_frames - 1, &pushed, &holder);
        assert!(owned_by(&poss, 1), "debounce did not reset");
        // A sustained excursion commits the loss.
        let poss = rig.tick(&pushed, &holder);
        assert_eq!(poss.state, PossessionState::Loose, "sustained loss must release");
    }

    /// On real vision the ball is undetected ~55% of the time *while held* (the
    /// holder occludes it, bursts up to ~10s). An Owned belief coasts through
    /// the dropout (marked stale) and only decays after `owned_coast_secs`.
    #[test]
    fn occluded_carry_coasts_then_decays() {
        let mut rig = OppRig::new();
        let off = rig.cfg.dribbler_offset;
        let holder = vec![opp(1, -off, 0.0, 0.0, 0.0)];
        let held = ball_at(0.0, 0.0, 0.0, 0.0, true);
        let occluded = ball_at(0.0, 0.0, 0.0, 0.0, false);

        rig.tick_n(rig.cfg.acquire_frames, &held, &holder);
        // 5 s of occlusion: held, stale.
        let poss = rig.tick_n((5.0 / 0.016) as u32, &occluded, &holder);
        assert!(owned_by(&poss, 1), "must coast through occlusion: {:?}", poss.state);
        assert!(poss.stale, "coasted belief must be marked stale");
        // Past owned_coast_secs: decayed.
        let extra = ((rig.cfg.owned_coast_secs - 5.0) / 0.016) as u32 + 5;
        let poss = rig.tick_n(extra, &occluded, &holder);
        assert_eq!(poss.state, PossessionState::Loose, "coast must be bounded");
    }

    /// A kick has an unmistakable signature — ball outside the retain band at
    /// high relative speed — and releases in one frame, not `release_frames`
    /// (opponents have no efference copy to tell us they kicked).
    #[test]
    fn kick_releases_in_one_frame() {
        let mut rig = OppRig::new();
        let off = rig.cfg.dribbler_offset;
        let holder = vec![opp(1, -off, 0.0, 0.0, 0.0)];
        let held = ball_at(0.0, 0.0, 0.0, 0.0, true);
        let kicked = ball_at(600.0, 0.0, 3000.0, 0.0, true);

        rig.tick_n(rig.cfg.acquire_frames, &held, &holder);
        let poss = rig.tick(&kicked, &holder);
        assert_eq!(poss.state, PossessionState::Loose, "kick must release immediately");
    }

    /// A ball rolling slowly through the dribbler zone for fewer than
    /// `acquire_frames` frames is a pass-by, not a hold.
    #[test]
    fn brief_roll_past_does_not_acquire() {
        let mut rig = OppRig::new();
        let off = rig.cfg.dribbler_offset;
        let robot = vec![opp(1, -off, 0.0, 0.0, 0.0)];
        let in_band = ball_at(0.0, 0.0, 200.0, 0.0, true);
        let gone = ball_at(600.0, 0.0, 200.0, 0.0, true);

        let poss = rig.tick_n(rig.cfg.acquire_frames - 1, &in_band, &robot);
        assert_eq!(poss.state, PossessionState::Loose);
        let poss = rig.tick_n(5, &gone, &robot);
        assert_eq!(poss.state, PossessionState::Loose);
    }

    /// A ball moving *relative to* the robot can't be acquired even sitting in
    /// the dribbler zone — acquisition demands the ball move with its holder.
    #[test]
    fn fast_relative_ball_never_acquires() {
        let mut rig = OppRig::new();
        let off = rig.cfg.dribbler_offset;
        let robot = vec![opp(1, -off, 0.0, 0.0, 0.0)];
        let passing = ball_at(0.0, 0.0, 800.0, 0.0, true); // > prox_acquire_rel_speed

        let poss = rig.tick_n(20, &passing, &robot);
        assert_eq!(poss.state, PossessionState::Loose, "rel-speed gate must hold");
    }

    /// Our own carry through a vision dropout: the breakbeam is
    /// vision-independent evidence, so possession stays *fresh* (not stale) for
    /// as long as the beam is tripped — this is what keeps `has_ball` true
    /// while the robot occludes the ball it is carrying.
    #[test]
    fn breakbeam_keeps_owning_through_occlusion() {
        let cfg = PossessionConfig::default();
        let mut t = PossessionTracker::new();
        let blue = vec![player(1, 0.0, 0.0, true)];
        let seen = ball_at(0.0, 0.0, 0.0, 0.0, true);
        let occluded = ball_at(0.0, 0.0, 0.0, 0.0, false);
        let mut now = 0.0;
        let mut tick = |t: &mut PossessionTracker, b: &BallData| {
            now += 0.016;
            t.update(Some(b), &blue, &[], true, false, &cfg, now)
        };

        assert!(tick(&mut t, &seen).state.owner().is_some(), "breakbeam owns in one frame");
        // 3 s of occlusion — far past hold_secs — with the beam still tripped.
        let mut poss = dies_core::Possession::default();
        for _ in 0..((3.0 / 0.016) as u32) {
            poss = tick(&mut t, &occluded);
        }
        assert!(poss.state.owner().is_some(), "beam evidence must survive occlusion");
        assert!(!poss.stale, "beam-backed ownership is fresh, not coasted");
    }

    /// A ToF-backup-enabled robot at the ball with a given raw ToF reading. Its
    /// hardware breakbeam is left `false` so the test proves the ToF path alone
    /// drives possession.
    fn tof_player(id: u32, conf: Option<u8>, tof_xy: Option<[i32; 2]>) -> PlayerData {
        let mut p = PlayerData::new(PlayerId::new(id));
        p.position = Vector2::new(0.0, 0.0);
        p.tof_backup_enabled = true;
        p.tof_confidence = conf;
        p.tof_xy = tof_xy;
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

    /// A sidelined robot (radio-lost / card-removed) sitting on the ball with its
    /// breakbeam tripped must NOT be credited with possession — we can't act on it.
    #[test]
    fn sidelined_robot_does_not_own_ball() {
        let cfg = PossessionConfig::default();
        let mut t = PossessionTracker::new();
        let mut dead = player(1, 0.0, 0.0, true); // breakbeam on, right on the ball
        dead.sideline = Some(dies_core::SidelineReason::RadioLost);
        let blue = vec![dead];
        let yellow: Vec<PlayerData> = vec![];
        let b = ball(0.0);

        let mut poss = dies_core::Possession::default();
        for i in 0..(cfg.contest_enter_frames + 20) {
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
        assert!(
            !matches!(poss.state, PossessionState::Owned { .. }),
            "a sidelined robot must not own the ball, got {:?}",
            poss.state
        );
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

    /// ToF-backup substitutes for the breakbeam: a toggled robot with the hardware
    /// breakbeam off still gains possession once the raw ToF condition
    /// (`conf > c_th && |x| < x_th && y < y_th`) holds for `enter_frames`.
    #[test]
    fn tof_backup_owns_after_enter_frames_and_releases_after_exit() {
        let cfg = PossessionConfig::default();
        let mut t = PossessionTracker::new();
        let yellow = vec![player(2, 3000.0, 0.0, false)]; // far, uninvolved
        let b = ball(0.0);
        let good = [0i32, (cfg.tof_backup_y_th as i32) - 1]; // |x|<x_th, y<y_th
        let conf = Some((cfg.tof_backup_conf_th as u8) + 1);
        let mut now = 0.0;
        let mut tick = |t: &mut PossessionTracker, blue: &[PlayerData]| {
            now += 0.016;
            t.update(Some(&b), blue, &yellow, true, false, &cfg, now)
        };

        let seen = vec![tof_player(1, conf, Some(good))];
        // Not owned until the raw condition has persisted enter_frames; the latch
        // flips (and breakbeam-commits in one frame) on the enter_frames-th tick.
        for _ in 0..(cfg.tof_backup_enter_frames - 1) {
            assert!(tick(&mut t, &seen).state.owner().is_none());
        }
        let owner = tick(&mut t, &seen).state.owner().expect("latched → owned");
        assert_eq!(owner.player_id, PlayerId::new(1));

        // Signal drops (no detection): retained through exit_frames, then released.
        let lost = vec![tof_player(1, conf, None)];
        for _ in 0..cfg.tof_backup_exit_frames {
            let _ = tick(&mut t, &lost);
        }
        // After the exit debounce the latch is off; possession decays to loose
        // (allow the hold_secs coast to elapse by ticking past it).
        for _ in 0..20 {
            let _ = tick(&mut t, &lost);
        }
        assert!(tick(&mut t, &lost).state.owner().is_none());
    }

    /// Fail-safe: an enabled robot with no confidence byte never latches, even
    /// with a perfect geometric reading.
    #[test]
    fn tof_backup_without_confidence_never_owns() {
        let cfg = PossessionConfig::default();
        let mut t = PossessionTracker::new();
        let yellow = vec![player(2, 3000.0, 0.0, false)];
        let b = ball(0.0);
        let blue = vec![tof_player(1, None, Some([0, 0]))];
        let mut poss = dies_core::Possession::default();
        for i in 0..(cfg.tof_backup_enter_frames + 5) {
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
        assert!(
            poss.state.owner().is_none(),
            "no confidence → no ToF-backup possession"
        );
    }
}
