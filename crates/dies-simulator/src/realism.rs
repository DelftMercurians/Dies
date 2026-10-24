//! Hardware realism profiles and noise from the simulation's shared RNG.
use dies_core::TeamColor;
use rand::{rngs::StdRng, Rng};
use std::f64::consts::PI;

/// Which teams' hardware is degraded by [`RealismConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealismTeams {
    /// No team — realism disabled.
    None,
    /// Both teams degrade (symmetric — the realistic case where both real teams
    /// have imperfect hardware).
    Both,
    /// Only Blue degrades (asymmetric — for A/B'ing one team's robustness against
    /// an ideal-hardware twin).
    Blue,
    /// Only Yellow degrades.
    Yellow,
}

/// Models imperfect hardware execution — the sim2real gap. Adds stochastic noise
/// to the three classes of physically-unreliable events: **shooting** (noisy
/// heading, noisy power, occasional misfire), **ball reception / pickup** (the
/// ball bumping off the dribbler or robot sides instead of being cleanly
/// captured), and **movement** (brief actuator stutters and general
/// sluggishness).
///
/// All noise is drawn from the simulation's seeded `StdRng`, so a given `(seed,
/// realism)` is byte-reproducible — the same property self-play relies on.
/// [`Default`] is a perfect actuator (everything off); realism is opt-in via
/// [`RealismConfig::from_env`] so normal benchmarks are unaffected.
#[derive(Debug, Clone)]
pub struct RealismConfig {
    /// Which teams' hardware is degraded.
    pub teams: RealismTeams,

    // --- Shooting ---
    /// Std dev (radians) of Gaussian noise added to the kick heading.
    pub shot_heading_noise: f64,
    /// Std dev (fraction of nominal) of Gaussian noise on kick power. `0.1` = 10 %.
    pub shot_power_noise: f64,
    /// Probability a kick fails to fire at all (a dud — the ball stays on the
    /// dribbler).
    pub shot_failure_prob: f64,

    // --- Reception / pickup ---
    /// Base probability the ball bumps off instead of being captured when a robot
    /// first gains it (models a slow pickup bump even with magnet mode).
    pub capture_fail_base: f64,
    /// Extra bump probability per m/s of the ball's incoming speed (models a fast
    /// pass bouncing off the dribbler / robot sides).
    pub capture_fail_per_mps: f64,
    /// Hard cap on total capture-failure probability.
    pub capture_fail_max: f64,
    /// Speed (mm/s) the ball is deflected when a capture bumps off.
    pub bump_speed: f64,

    // --- Movement ---
    /// Per-robot, per-step probability of a movement stutter (a brief hitch).
    pub move_stutter_prob: f64,
    /// Velocity multiplier applied during a stutter (`0.0` = frozen that step).
    pub move_stutter_factor: f64,
    /// Steady velocity multiplier (`1.0` = none; `<1` models sluggishness).
    pub move_sluggish: f64,
}

impl Default for RealismConfig {
    fn default() -> Self {
        RealismConfig {
            teams: RealismTeams::None,
            shot_heading_noise: 0.0,
            shot_power_noise: 0.0,
            shot_failure_prob: 0.0,
            capture_fail_base: 0.0,
            capture_fail_per_mps: 0.0,
            capture_fail_max: 0.0,
            bump_speed: 0.0,
            move_stutter_prob: 0.0,
            move_stutter_factor: 1.0,
            move_sluggish: 1.0,
        }
    }
}

impl RealismConfig {
    /// The level-5 ("max") profile. Intermediate levels scale linearly from a
    /// perfect actuator (level 0) toward this, so `DIES_REALISM_LEVEL` defines a
    /// smooth gradient. Magnitudes chosen to be hard-but-playable; calibrated
    /// against the degradation curve, not from real hardware data (none yet).
    pub(crate) fn scaled(frac: f64) -> Self {
        let f = frac.clamp(0.0, 1.0);
        RealismConfig {
            teams: RealismTeams::Both,
            shot_heading_noise: f * 0.09,
            shot_power_noise: f * 0.12,
            shot_failure_prob: f * 0.05,
            // Capture-bump strengthened relative to the first calibration so
            // reception/pickup failures contribute on par with shot noise.
            capture_fail_base: f * 0.10,
            capture_fail_per_mps: f * 0.085,
            capture_fail_max: 0.5,
            bump_speed: 700.0,
            // Movement noise toned down from the first calibration: a 2 %/frame
            // freeze + 8 % steady slow applied to *every* robot swamped the other
            // two factors (it taxes all positioning/race-winning at once). At
            // ~0.9 %/frame + 4 % slow it is a meaningful but no-longer-dominant
            // contributor, so the gradient is a balanced three-factor model.
            move_stutter_prob: f * 0.009,
            move_stutter_factor: 0.0,
            move_sluggish: 1.0 - f * 0.04,
        }
    }

    /// Build a config from the environment, read once at simulator construction
    /// (so a sweep needs no recompile). `DIES_REALISM_LEVEL` (0..=5, may be
    /// fractional) selects a point on the gradient; `DIES_REALISM_TEAMS`
    /// (`both`|`blue`|`yellow`|`none`) scopes it. Individual `DIES_REALISM_*`
    /// knobs override single parameters on top of the level preset, for fine
    /// one-axis sweeps.
    pub fn from_env() -> Self {
        let level = std::env::var("DIES_REALISM_LEVEL")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        let mut cfg = if level > 0.0 {
            Self::scaled(level / 5.0)
        } else {
            Self::default()
        };

        if let Ok(t) = std::env::var("DIES_REALISM_TEAMS") {
            cfg.teams = match t.to_ascii_lowercase().as_str() {
                "both" => RealismTeams::Both,
                "blue" => RealismTeams::Blue,
                "yellow" => RealismTeams::Yellow,
                _ => RealismTeams::None,
            };
        }

        let ov = |name: &str, cur: f64| -> f64 {
            std::env::var(name)
                .ok()
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(cur)
        };
        cfg.shot_heading_noise = ov("DIES_REALISM_SHOT_HEADING", cfg.shot_heading_noise);
        cfg.shot_power_noise = ov("DIES_REALISM_SHOT_POWER", cfg.shot_power_noise);
        cfg.shot_failure_prob = ov("DIES_REALISM_SHOT_FAIL", cfg.shot_failure_prob);
        cfg.capture_fail_base = ov("DIES_REALISM_CAPTURE_BASE", cfg.capture_fail_base);
        cfg.capture_fail_per_mps = ov("DIES_REALISM_CAPTURE_PER_MPS", cfg.capture_fail_per_mps);
        cfg.bump_speed = ov("DIES_REALISM_BUMP_SPEED", cfg.bump_speed);
        cfg.move_stutter_prob = ov("DIES_REALISM_STUTTER_PROB", cfg.move_stutter_prob);
        cfg.move_sluggish = ov("DIES_REALISM_SLUGGISH", cfg.move_sluggish);

        cfg
    }

    pub(crate) fn applies_to(&self, team: TeamColor) -> bool {
        match self.teams {
            RealismTeams::None => false,
            RealismTeams::Both => true,
            RealismTeams::Blue => team == TeamColor::Blue,
            RealismTeams::Yellow => team == TeamColor::Yellow,
        }
    }

    /// Probability that a fresh capture of a ball arriving at `speed_mps` bumps
    /// off instead of sticking. `0` when realism doesn't apply to `team`.
    pub(crate) fn capture_fail_prob(&self, team: TeamColor, speed_mps: f64) -> f64 {
        if !self.applies_to(team) {
            return 0.0;
        }
        (self.capture_fail_base + self.capture_fail_per_mps * speed_mps.max(0.0))
            .clamp(0.0, self.capture_fail_max)
    }
}

/// Sample a zero-mean Gaussian with the given std dev from a uniform RNG
/// (Box–Muller). Returns 0 for a non-positive std dev (the no-noise fast path).
pub(crate) fn gaussian(rng: &mut StdRng, std: f64) -> f64 {
    if std <= 0.0 {
        return 0.0;
    }
    let u1: f64 = rng.gen_range(1e-12..1.0);
    let u2: f64 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos() * std
}
