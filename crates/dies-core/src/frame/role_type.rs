use serde::{Deserialize, Serialize};

/// Role type that determines special rules applied to a player.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RoleType {
    /// No special role.
    #[default]
    None,
    /// A player who is the goalkeeper.
    Goalkeeper,
    /// A player who is the kickoff kicker.
    KickoffKicker,
    /// A player who is the freekick taker.
    FreekickTaker,
    /// A player who is the penalty taker.
    PenaltyTaker,
}
