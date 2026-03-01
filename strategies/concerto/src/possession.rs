use dies_strategy_api::prelude::*;

/// Coarse possession category for plan continuity checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PossessionCategory {
    We,
    Opponent,
    Loose,
}

/// Detailed possession state.
#[derive(Debug, Clone)]
pub enum PossessionState {
    We(PlayerId),
    Opponent(PlayerId),
    Loose,
}

impl From<&PossessionState> for PossessionCategory {
    fn from(state: &PossessionState) -> Self {
        match state {
            PossessionState::We(_) => PossessionCategory::We,
            PossessionState::Opponent(_) => PossessionCategory::Opponent,
            PossessionState::Loose => PossessionCategory::Loose,
        }
    }
}

/// Maximum distance (mm) for an opponent to be considered in possession.
const OPP_POSSESSION_THRESHOLD: f64 = 150.0;

/// Determine which side currently possesses the ball.
///
/// Priority:
/// 1. Any own player whose breakbeam reports `has_ball` → `We(id)`.
/// 2. Closest opponent within [`OPP_POSSESSION_THRESHOLD`] mm → `Opponent(id)`.
/// 3. Otherwise → `Loose`.
pub fn detect_possession(world: &World) -> PossessionState {
    // Check own players first (breakbeam is authoritative).
    for player in world.own_players() {
        if player.has_ball {
            return PossessionState::We(player.id);
        }
    }

    // Need ball position for proximity checks.
    let ball_pos = match world.ball_position() {
        Some(pos) => pos,
        None => return PossessionState::Loose,
    };

    // Find the closest opponent to the ball.
    let closest = world
        .opp_players()
        .iter()
        .map(|p| (p, (p.position - ball_pos).norm()))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    if let Some((player, dist)) = closest {
        if dist < OPP_POSSESSION_THRESHOLD {
            return PossessionState::Opponent(player.id);
        }
    }

    PossessionState::Loose
}
