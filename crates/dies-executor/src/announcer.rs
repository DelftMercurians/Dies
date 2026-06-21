//! Backend-generated announcer feed.
//!
//! Watches every referee message (sim and live) plus the simulator's own
//! internal events, and turns them into human-readable [`Announcement`] lines
//! that the web UI renders as a scrolling commentary log. Keeping the formatting
//! here means sim and live matches narrate identically and the rich game-event
//! detail stays where the protobuf data already lives.

use std::collections::HashMap;

use dies_core::{Announcement, AnnouncementCategory, TeamColor};
use dies_protos::{
    ssl_gc_common::Team,
    ssl_gc_game_event::{game_event::Event as GcEvent, GameEvent},
    ssl_gc_referee_message::{referee::Command, Referee},
};
use dies_simulator::{RefereeMessage, SimRefereeEvent};

/// Suppress an identical line if it was emitted within this many seconds. Tames
/// level-triggered sim events (e.g. "defender too close" raised every frame).
const DEDUPE_SECS: f64 = 3.0;

/// How many recent dedupe keys to retain (bounds memory; far above the number of
/// distinct lines that can occur within `DEDUPE_SECS`).
const MAX_DEDUPE_KEYS: usize = 64;

#[derive(Default)]
pub struct Announcer {
    next_id: u64,
    /// The last referee command we narrated, to detect transitions.
    last_command: Option<Command>,
    /// Ids of game events already narrated (live autoref feed resends the
    /// accumulated array every packet until the next RUNNING state).
    seen_event_ids: std::collections::HashSet<String>,
    /// Last emit time per line text, for short-window dedupe.
    recent: HashMap<String, f64>,
    /// Lines minted since the last `drain`.
    pending: Vec<Announcement>,
}

impl Announcer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Process a referee message (works for both the sim's synthetic messages and
    /// a real game controller's). Narrates command transitions and game events.
    pub fn on_referee(&mut self, referee: &Referee, t: f64) {
        // Command transitions.
        let command = referee.command();
        if self.last_command != Some(command) {
            self.last_command = Some(command);
            if let Some((category, team, text)) = command_announcement(command) {
                self.emit(category, team, text, t);
            }
        }

        // Game events (populated by a real autoref; empty in sim).
        if referee.game_events.is_empty() {
            // The array is cleared when play resumes — reset so the same ids can
            // legitimately recur later in the match.
            self.seen_event_ids.clear();
        } else {
            // Collect first to avoid borrowing `referee` across `self` mutation.
            let mut new_events: Vec<(AnnouncementCategory, Option<TeamColor>, String)> = Vec::new();
            for ev in &referee.game_events {
                let key = event_dedupe_key(ev);
                if self.seen_event_ids.insert(key) {
                    if let Some(item) = game_event_announcement(ev) {
                        new_events.push(item);
                    }
                }
            }
            for (category, team, text) in new_events {
                self.emit(category, team, text, t);
            }
        }
    }

    /// Process a simulator-internal event (foul, violation, goal).
    pub fn on_sim_event(&mut self, ev: &SimRefereeEvent, t: f64) {
        if let Some((category, team, text)) = sim_event_announcement(ev) {
            self.emit(category, team, text, t);
        }
    }

    /// Take the lines minted since the previous call.
    pub fn drain(&mut self) -> Vec<Announcement> {
        std::mem::take(&mut self.pending)
    }

    fn emit(
        &mut self,
        category: AnnouncementCategory,
        team: Option<TeamColor>,
        text: String,
        t: f64,
    ) {
        // Short-window dedupe by text.
        if let Some(&last) = self.recent.get(&text) {
            if (t - last).abs() < DEDUPE_SECS {
                return;
            }
        }
        self.recent.insert(text.clone(), t);
        if self.recent.len() > MAX_DEDUPE_KEYS {
            // Drop the oldest half (cheap, infrequent).
            let mut entries: Vec<_> = self.recent.iter().map(|(k, v)| (k.clone(), *v)).collect();
            entries.sort_by(|a, b| a.1.total_cmp(&b.1));
            for (k, _) in entries.into_iter().take(MAX_DEDUPE_KEYS / 2) {
                self.recent.remove(&k);
            }
        }

        let id = self.next_id;
        self.next_id += 1;
        self.pending.push(Announcement {
            id,
            timestamp: t,
            category,
            team,
            text,
        });
    }
}

fn team_of(team: Team) -> Option<TeamColor> {
    match team {
        Team::BLUE => Some(TeamColor::Blue),
        Team::YELLOW => Some(TeamColor::Yellow),
        Team::UNKNOWN => None,
    }
}

fn team_name(team: Option<TeamColor>) -> &'static str {
    match team {
        Some(TeamColor::Blue) => "Blue",
        Some(TeamColor::Yellow) => "Yellow",
        None => "?",
    }
}

/// A line for a referee command transition. `None` for commands we don't narrate.
fn command_announcement(
    command: Command,
) -> Option<(AnnouncementCategory, Option<TeamColor>, String)> {
    use AnnouncementCategory as C;
    let blue = Some(TeamColor::Blue);
    let yellow = Some(TeamColor::Yellow);
    Some(match command {
        Command::HALT => (C::Info, None, "Halt".to_string()),
        Command::STOP => (C::Info, None, "Stop".to_string()),
        Command::NORMAL_START => (C::Info, None, "Normal start — play on".to_string()),
        Command::FORCE_START => (C::Info, None, "Force start — ball is free".to_string()),
        Command::PREPARE_KICKOFF_BLUE => (C::Kickoff, blue, "Kick-off — Blue".to_string()),
        Command::PREPARE_KICKOFF_YELLOW => (C::Kickoff, yellow, "Kick-off — Yellow".to_string()),
        Command::PREPARE_PENALTY_BLUE => (C::Penalty, blue, "Penalty — Blue".to_string()),
        Command::PREPARE_PENALTY_YELLOW => (C::Penalty, yellow, "Penalty — Yellow".to_string()),
        Command::DIRECT_FREE_BLUE | Command::INDIRECT_FREE_BLUE => {
            (C::FreeKick, blue, "Free kick — Blue".to_string())
        }
        Command::DIRECT_FREE_YELLOW | Command::INDIRECT_FREE_YELLOW => {
            (C::FreeKick, yellow, "Free kick — Yellow".to_string())
        }
        Command::TIMEOUT_BLUE => (C::Info, blue, "Timeout — Blue".to_string()),
        Command::TIMEOUT_YELLOW => (C::Info, yellow, "Timeout — Yellow".to_string()),
        Command::BALL_PLACEMENT_BLUE => {
            (C::Placement, blue, "Ball placement — Blue".to_string())
        }
        Command::BALL_PLACEMENT_YELLOW => {
            (C::Placement, yellow, "Ball placement — Yellow".to_string())
        }
        // Goals are narrated via game events (live) / sim events (sim).
        Command::GOAL_BLUE | Command::GOAL_YELLOW => return None,
    })
}

fn sim_event_announcement(
    ev: &SimRefereeEvent,
) -> Option<(AnnouncementCategory, Option<TeamColor>, String)> {
    use AnnouncementCategory as C;
    let team = ev.team;
    let item = match ev.kind {
        RefereeMessage::Goal => (C::Goal, team, format!("Goal — {}!", team_name(team))),
        RefereeMessage::NoProgress => {
            (C::Stoppage, None, "No progress — force start".to_string())
        }
        RefereeMessage::DoubleTouchViolation => (
            C::Foul,
            team,
            format!("Double touch — {}", team_name(team)),
        ),
        RefereeMessage::DefenderTooCloseToBall => (
            C::Foul,
            team,
            format!("Defender too close to the ball — {}", team_name(team)),
        ),
        RefereeMessage::BoundaryCrossing => (
            C::Foul,
            team,
            format!("Ball kicked out of bounds — {}", team_name(team)),
        ),
        RefereeMessage::FreekickTimeExceeded => {
            (C::Stoppage, None, "Free kick not taken in time".to_string())
        }
        RefereeMessage::PenaltyTimeExceeded => {
            (C::Stoppage, None, "Penalty not taken in time".to_string())
        }
        RefereeMessage::PenaltyKeeperPositionViolation
        | RefereeMessage::PenaltyRobotPositionViolation => {
            (C::Foul, team, "Penalty position violation".to_string())
        }
        RefereeMessage::PenaltyBallDirectionViolation => {
            (C::Foul, team, "Penalty ball moved the wrong way".to_string())
        }
        RefereeMessage::RobotTooCloseToOpponentDefenseArea => (
            C::Foul,
            team,
            "Robot too close to opponent defense area".to_string(),
        ),
        // Lower-value events — surface generically.
        RefereeMessage::BallPlacementInterference => {
            (C::Foul, team, "Ball placement interference".to_string())
        }
        RefereeMessage::RobotOutOfField => (C::Foul, team, "Robot out of field".to_string()),
        RefereeMessage::PlayerTooCloseToBall => {
            (C::Foul, team, "Player too close to the ball".to_string())
        }
        RefereeMessage::KickoffPositionViolation => {
            (C::Foul, team, "Kick-off position violation".to_string())
        }
    };
    Some(item)
}

/// A stable dedupe key for a game event: its id if present, else a synthesized
/// key from the type + origin so the same event isn't narrated twice.
fn event_dedupe_key(ev: &GameEvent) -> String {
    if ev.has_id() && !ev.id().is_empty() {
        ev.id().to_string()
    } else {
        format!("{:?}:{}", ev.type_(), ev.created_timestamp())
    }
}

/// Format a live autoref game event. Explicit arms for the high-value events,
/// with a graceful humanized fallback so every event still produces a line.
fn game_event_announcement(
    ev: &GameEvent,
) -> Option<(AnnouncementCategory, Option<TeamColor>, String)> {
    use AnnouncementCategory as C;
    let item = match ev.event.as_ref() {
        Some(GcEvent::BallLeftFieldTouchLine(e)) => (
            C::Stoppage,
            team_of(e.by_team()),
            "Ball left the field — throw-in".to_string(),
        ),
        Some(GcEvent::BallLeftFieldGoalLine(e)) => (
            C::Stoppage,
            team_of(e.by_team()),
            "Ball left the field — goal/corner kick".to_string(),
        ),
        Some(GcEvent::AimlessKick(e)) => {
            (C::Stoppage, team_of(e.by_team()), "Aimless kick".to_string())
        }
        Some(GcEvent::Goal(e)) | Some(GcEvent::PossibleGoal(e)) => {
            let team = team_of(e.by_team());
            (C::Goal, team, format!("Goal — {}!", team_name(team)))
        }
        Some(GcEvent::InvalidGoal(e)) => {
            (C::Goal, team_of(e.by_team()), "Goal disallowed".to_string())
        }
        Some(GcEvent::NoProgressInGame(_)) => {
            (C::Stoppage, None, "No progress — force start".to_string())
        }
        Some(GcEvent::DefenderInDefenseArea(e)) => (
            C::Foul,
            team_of(e.by_team()),
            "Defender in defense area — penalty".to_string(),
        ),
        Some(GcEvent::BotCrashUnique(e)) => {
            (C::Foul, team_of(e.by_team()), "Robot crash".to_string())
        }
        Some(GcEvent::BotCrashDrawn(_)) => {
            (C::Foul, None, "Robots crashed (both at fault)".to_string())
        }
        Some(GcEvent::BotPushedBot(e)) => {
            (C::Foul, team_of(e.by_team()), "Pushing".to_string())
        }
        Some(GcEvent::BotHeldBallDeliberately(e)) => {
            (C::Foul, team_of(e.by_team()), "Ball holding".to_string())
        }
        Some(GcEvent::BotTippedOver(e)) => {
            (C::Foul, team_of(e.by_team()), "Robot tipped over".to_string())
        }
        Some(GcEvent::KeeperHeldBall(e)) => (
            C::Foul,
            team_of(e.by_team()),
            "Keeper held the ball".to_string(),
        ),
        Some(GcEvent::BotDribbledBallTooFar(e)) => (
            C::Foul,
            team_of(e.by_team()),
            "Excessive dribbling".to_string(),
        ),
        Some(GcEvent::BotKickedBallTooFast(e)) => (
            C::Foul,
            team_of(e.by_team()),
            "Ball kicked too fast".to_string(),
        ),
        Some(GcEvent::AttackerTooCloseToDefenseArea(e)) => (
            C::Foul,
            team_of(e.by_team()),
            "Attacker too close to defense area".to_string(),
        ),
        Some(GcEvent::AttackerTouchedBallInDefenseArea(e)) => (
            C::Foul,
            team_of(e.by_team()),
            "Attacker touched ball in defense area".to_string(),
        ),
        Some(GcEvent::BoundaryCrossing(e)) => (
            C::Foul,
            team_of(e.by_team()),
            "Ball kicked out of the field".to_string(),
        ),
        Some(GcEvent::AttackerDoubleTouchedBall(e)) => {
            (C::Foul, team_of(e.by_team()), "Double touch".to_string())
        }
        Some(GcEvent::PlacementSucceeded(e)) => (
            C::Placement,
            team_of(e.by_team()),
            "Ball placement succeeded".to_string(),
        ),
        Some(GcEvent::PlacementFailed(e)) => (
            C::Placement,
            team_of(e.by_team()),
            "Ball placement failed".to_string(),
        ),
        Some(GcEvent::MultipleFouls(e)) => (
            C::Card,
            team_of(e.by_team()),
            "Multiple fouls — yellow card".to_string(),
        ),
        Some(GcEvent::MultipleCards(e)) => (
            C::Card,
            team_of(e.by_team()),
            "Multiple cards — red card".to_string(),
        ),
        Some(GcEvent::TooManyRobots(e)) => {
            (C::Foul, team_of(e.by_team()), "Too many robots".to_string())
        }
        Some(GcEvent::UnsportingBehaviorMinor(e)) => (
            C::Card,
            team_of(e.by_team()),
            "Unsporting behavior — yellow card".to_string(),
        ),
        Some(GcEvent::UnsportingBehaviorMajor(e)) => (
            C::Card,
            team_of(e.by_team()),
            "Unsporting behavior — red card".to_string(),
        ),
        // Anything else: humanize the type name so the line still appears.
        _ => (category_from_type_name(ev), None, humanize_type(ev)),
    };
    Some(item)
}

fn type_name(ev: &GameEvent) -> String {
    format!("{:?}", ev.type_())
}

fn humanize_type(ev: &GameEvent) -> String {
    let raw = type_name(ev);
    let lowered = raw.replace('_', " ").to_lowercase();
    let mut chars = lowered.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => lowered,
    }
}

fn category_from_type_name(ev: &GameEvent) -> AnnouncementCategory {
    let name = type_name(ev);
    if name.contains("GOAL") {
        AnnouncementCategory::Goal
    } else if name.contains("CARD") {
        AnnouncementCategory::Card
    } else if name.contains("PLACEMENT") {
        AnnouncementCategory::Placement
    } else if name.contains("BALL_LEFT") || name.contains("NO_PROGRESS") || name.contains("AIMLESS")
    {
        AnnouncementCategory::Stoppage
    } else {
        AnnouncementCategory::Foul
    }
}
