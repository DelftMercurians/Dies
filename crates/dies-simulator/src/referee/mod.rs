//! Auto-referee decisions, match state, scoring, and referee event queues.
use crate::{config::SimulationConfig, physics::PhysicsWorld, robot::RobotFleet};
pub(crate) use cards::card_team_info;
use dies_core::{Angle, GameState, GcSimCommand, PlayerId, SideAssignment, TeamColor, Vector2};
use dies_protos::ssl_gc_referee_message::{referee, Referee};
#[cfg(test)]
pub(crate) use placement::valid_free_kick_position;
#[cfg(test)]
pub(crate) use placement::FREE_KICK_DEFENSE_AREA_DISTANCE;
pub(crate) use placement::FREE_KICK_PLACEMENT_MARGIN;
use rapier3d_f64::prelude::*;
use serde::Serialize;
pub use state::SimulationGameState;
use std::{
    collections::{HashMap, VecDeque},
    fmt,
};
mod cards;
mod placement;
mod rules;
mod state;
mod transitions;
/// Goalkeeper id the sim's referee designates for both teams (broadcast in the
/// `TeamInfo` messages) — the single source of truth its own penalty-position
/// check must use too.
pub(crate) const SIM_GOALKEEPER_ID: u32 = 1;
/// SSL rule 8.4.3 grace period (seconds): a recurring ball-out-of-play foul is
/// re-raised for the same team at most once per this interval.
const FOUL_GRACE_SECS: f64 = 2.0;

pub(crate) struct RefereeContext<'a> {
    pub(crate) config: &'a SimulationConfig,
    pub(crate) now: f64,
    pub(crate) physics: &'a PhysicsWorld,
    pub(crate) robots: &'a RobotFleet,
    pub(crate) actions: Vec<RefereeAction>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub enum RefereeMessage {
    RobotTooCloseToOpponentDefenseArea, //8.4.1
    BoundaryCrossing,                   //8.4.1
    DefenderTooCloseToBall,             //8.4.3
    BallPlacementInterference,          //8.4.3
    RobotOutOfField,
    FreekickTimeExceeded,
    PlayerTooCloseToBall,
    KickoffPositionViolation,
    DoubleTouchViolation,
    PenaltyKeeperPositionViolation,
    PenaltyRobotPositionViolation,
    PenaltyBallDirectionViolation,
    PenaltyTimeExceeded,
    NoProgress,
    Goal,
    /// A yellow card was shown to a team (SSL §5.5.1). Lowers `max_allowed_bots`.
    YellowCard,
    /// A team has more robots on the field than its current `max_allowed_bots`
    /// allows (e.g. it has not yet removed a carded robot).
    TooManyRobots,
}

/// A simulator-internal referee event together with the team it concerns (if
/// any). Surfaced to the executor so sim matches narrate the same class of
/// events (fouls, violations, goals) that a real autoref would report.
#[derive(Debug, Clone)]
pub struct SimRefereeEvent {
    pub kind: RefereeMessage,
    pub team: Option<TeamColor>,
}

impl fmt::Display for RefereeMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RefereeMessage::RobotTooCloseToOpponentDefenseArea => {
                write!(f, "Robot too close to opponent defense area")
            }
            RefereeMessage::BoundaryCrossing => write!(f, "Boundary crossing"),
            RefereeMessage::DefenderTooCloseToBall => write!(f, "Defender too close to ball"),
            RefereeMessage::FreekickTimeExceeded => write!(f, "Free kick time exceeded"),
            RefereeMessage::BallPlacementInterference => write!(f, "Ball placement interference"),
            RefereeMessage::RobotOutOfField => write!(f, "Robot out of field"),
            RefereeMessage::PlayerTooCloseToBall => write!(f, "Player too close to ball"),
            RefereeMessage::KickoffPositionViolation => write!(f, "Kickoff position violation"),
            RefereeMessage::DoubleTouchViolation => write!(f, "Double touch violation"),
            RefereeMessage::PenaltyKeeperPositionViolation => {
                write!(f, "Penalty keeper position violation")
            }
            RefereeMessage::PenaltyRobotPositionViolation => {
                write!(f, "Penalty robot position violation")
            }
            RefereeMessage::PenaltyBallDirectionViolation => {
                write!(f, "Penalty ball direction violation")
            }
            RefereeMessage::PenaltyTimeExceeded => write!(f, "Penalty time exceeded"),
            RefereeMessage::NoProgress => write!(f, "No progress"),
            RefereeMessage::Goal => write!(f, "Goal"),
            RefereeMessage::YellowCard => write!(f, "Yellow card"),
            RefereeMessage::TooManyRobots => write!(f, "Too many robots"),
        }
    }
}

pub(crate) struct AutoReferee {
    pub(crate) referee_message: VecDeque<Referee>,
    /// Monotonic counter advertised in the referee `command_counter` field so
    /// consumers can tell distinct commands apart.
    pub(crate) command_counter: u32,
    /// Simulator-internal referee events (fouls/violations/goals) awaiting drain
    /// into the announcer feed.
    pub(crate) referee_events: VecDeque<SimRefereeEvent>,
    /// Last sim time (seconds) each recurring §8.4.3 foul was raised, keyed by
    /// (foul, team). Enforces the rule's 2-second per-team grace period so a
    /// sustained violation isn't re-emitted every physics frame.
    pub(crate) foul_last_raised: HashMap<(RefereeMessage, Option<TeamColor>), f64>,
    /// Running match score, incremented on each goal and reported via `TeamInfo`.
    pub(crate) blue_score: u32,
    pub(crate) yellow_score: u32,
    pub(crate) game_state: SimulationGameState,
    pub(crate) designated_ball_position: Vector<f64>,
    pub(crate) last_touch_info: Option<(PlayerId, TeamColor)>,
    pub(crate) side_assignment: SideAssignment,
    // New fields for rule enforcement
    pub(crate) kick_start_time: f64,
    pub(crate) kick_ball_position: Option<Vector2>,
    pub(crate) last_kicker_info: Option<(PlayerId, TeamColor)>,
    pub(crate) kick_in_progress: bool,
    /// Active yellow-card expiry sim-times (seconds) per team. A card is active
    /// while `current_time < expiry`; each active card lowers that team's
    /// `max_allowed_bots` by one. Cards expire 120 s after they are shown
    /// (SSL §5.5.1), restoring the bot allowance.
    pub(crate) blue_card_expiries: Vec<f64>,
    pub(crate) yellow_card_expiries: Vec<f64>,
    /// Robots physically removed from the field to serve an active card, by team.
    /// Modelled like a real substitution: once the executor has parked a carded
    /// robot beyond the touchline, the sim removes it (so the strategy sees the
    /// true, reduced roster); on card expiry it is re-added at the halfway line.
    pub(crate) blue_carded_out: Vec<PlayerId>,
    pub(crate) yellow_carded_out: Vec<PlayerId>,
    /// Last referee command emitted, so a card change can be re-broadcast with
    /// updated `TeamInfo` without altering the game state — re-emitting an
    /// unchanged command is a no-op in the world game-state tracker (which
    /// early-returns on an unchanged command) but still refreshes the card /
    /// max-bots fields, which are read on every referee message.
    pub(crate) last_command: referee::Command,
}

impl AutoReferee {
    pub(crate) fn new(side_assignment: SideAssignment) -> Self {
        Self {
            referee_message: VecDeque::new(),
            command_counter: 0,
            referee_events: VecDeque::new(),
            foul_last_raised: HashMap::new(),
            blue_score: 0,
            yellow_score: 0,
            game_state: SimulationGameState::default(),
            designated_ball_position: Vector::new(0.0, 0.0, 20.0),
            last_touch_info: None,
            side_assignment: side_assignment,
            kick_start_time: 0.0,
            kick_ball_position: None,
            last_kicker_info: None,
            kick_in_progress: false,
            blue_card_expiries: Vec::new(),
            yellow_card_expiries: Vec::new(),
            blue_carded_out: Vec::new(),
            yellow_carded_out: Vec::new(),
            last_command: referee::Command::HALT,
        }
    }

    pub(crate) fn handle_gc_command(
        &mut self,
        ctx: &mut RefereeContext<'_>,
        command: GcSimCommand,
    ) {
        match command {
            GcSimCommand::KickOff { team_color } => {
                match team_color {
                    TeamColor::Blue => {
                        self.update_referee_command(ctx, referee::Command::PREPARE_KICKOFF_BLUE);
                    }
                    TeamColor::Yellow => {
                        self.update_referee_command(ctx, referee::Command::PREPARE_KICKOFF_YELLOW);
                    }
                }
                self.game_state = SimulationGameState::prepare_kickoff(team_color);
            }
            GcSimCommand::Penalty { team_color } => {
                match team_color {
                    TeamColor::Blue => {
                        self.update_referee_command(ctx, referee::Command::PREPARE_PENALTY_BLUE);
                    }
                    TeamColor::Yellow => {
                        self.update_referee_command(ctx, referee::Command::PREPARE_PENALTY_YELLOW);
                    }
                }
                // The referee places the ball on the penalty mark (rules §5.3.5):
                // on the goal-to-goal line, `goal_line_to_penalty_mark` from the
                // center of the goal the attacking team shoots at.
                let attack_sign = self.side_assignment.attacking_direction_sign(team_color);
                let mark_x = attack_sign
                    * (ctx.config.field_geometry.field_length / 2.0
                        - ctx.config.field_geometry.goal_line_to_penalty_mark);
                self.place_ball(ctx, mark_x, 0.0);
                // A fresh penalty starts a fresh kick procedure — a stale
                // in-progress flag (command issued mid-kick) would skip the
                // position check and NORMAL_START entirely.
                self.kick_in_progress = false;
                self.game_state = SimulationGameState::penalty(team_color);
            }
            GcSimCommand::Stop => {
                self.update_referee_command(ctx, referee::Command::STOP);
                self.game_state = SimulationGameState::stop();
            }
            GcSimCommand::Halt => {
                // Make Halt authoritative over the auto-ref, not just a broadcast,
                // so a seeded/forced Halt actually freezes the state machine.
                self.update_referee_command(ctx, referee::Command::HALT);
                self.game_state = SimulationGameState::Halt;
            }
            GcSimCommand::NormalStart => {
                self.update_referee_command(ctx, referee::Command::NORMAL_START)
            }
            GcSimCommand::ForceStart => {
                // Drop straight into free play and make the auto-ref agree, so a
                // forced/seeded start isn't immediately overridden by a pending
                // kickoff in the state machine.
                self.update_referee_command(ctx, referee::Command::FORCE_START);
                self.game_state = SimulationGameState::run();
            }
            GcSimCommand::DirectFree { team_color } => {
                // Relocate the ball to a rule 5.3.3-valid free-kick position. A real
                // referee/auto-ref always places the ball ≥1 m from either defense
                // area before a free kick; without this a manual `DirectFree` can
                // leave the ball jammed against a defense area where the kicker
                // cannot legally get behind it, and the restart just times out.
                if let Some(ball) = ctx.physics.ball.as_ref() {
                    let current = ctx
                        .physics
                        .rigid_body_set
                        .get(ball._rigid_body_handle)
                        .unwrap()
                        .position()
                        .translation
                        .vector
                        .xy();
                    let placement = self.valid_free_kick_position(ctx, current);
                    self.place_ball(ctx, placement.x, placement.y);
                }
                match team_color {
                    TeamColor::Blue => {
                        self.update_referee_command(ctx, referee::Command::DIRECT_FREE_BLUE);
                    }
                    TeamColor::Yellow => {
                        self.update_referee_command(ctx, referee::Command::DIRECT_FREE_YELLOW);
                    }
                }
                self.game_state = SimulationGameState::free_kick(team_color);
            }
            GcSimCommand::BallPlacement {
                team_color,
                position,
            } => {
                match team_color {
                    TeamColor::Blue => {
                        self.update_referee_command(ctx, referee::Command::BALL_PLACEMENT_BLUE);
                    }
                    TeamColor::Yellow => {
                        self.update_referee_command(ctx, referee::Command::BALL_PLACEMENT_YELLOW);
                    }
                }
                self.game_state = SimulationGameState::ball_placement(team_color, position);
            }
            GcSimCommand::YellowCard { team_color } => {
                self.add_yellow_card(ctx, team_color);
            }
        }
    }

    /// Seed the auto-referee game state directly, e.g. when starting a headless
    /// rollout from a snapshot. Maps the strategy-facing [`GameState`] onto the
    /// equivalent GC command; `Run`/`Kickoff`/`PenaltyRun` drop the match straight
    /// into free play via `ForceStart` (skipping the kickoff cycle). `operating_team`
    /// selects the team for asymmetric states (kickoff/free-kick/penalty/placement)
    /// and is ignored for symmetric ones; it defaults to Blue when unknown.
    pub(crate) fn seed_game_state(
        &mut self,
        ctx: &mut RefereeContext<'_>,
        state: GameState,
        operating_team: Option<TeamColor>,
    ) {
        let team = operating_team.unwrap_or(TeamColor::Blue);
        match state {
            GameState::Run | GameState::Kickoff | GameState::PenaltyRun => {
                self.handle_gc_command(ctx, GcSimCommand::ForceStart)
            }
            GameState::Stop => self.handle_gc_command(ctx, GcSimCommand::Stop),
            GameState::Halt | GameState::Timeout => self.handle_gc_command(ctx, GcSimCommand::Halt),
            GameState::PrepareKickoff => {
                self.handle_gc_command(ctx, GcSimCommand::KickOff { team_color: team })
            }
            GameState::FreeKick => {
                self.handle_gc_command(ctx, GcSimCommand::DirectFree { team_color: team })
            }
            GameState::Penalty | GameState::PreparePenalty => {
                self.handle_gc_command(ctx, GcSimCommand::Penalty { team_color: team })
            }
            GameState::BallReplacement(position) => self.handle_gc_command(
                ctx,
                GcSimCommand::BallPlacement {
                    team_color: team,
                    position,
                },
            ),
            GameState::Unknown => {}
        }
    }

    pub(crate) fn update_referee_command(
        &mut self,
        ctx: &mut RefereeContext<'_>,
        command: referee::Command,
    ) {
        self.update_referee_command_full(ctx, command, None, None);
    }

    /// Broadcast a referee command with its optional next command and countdown.
    /// Packet construction still observes the stored game_state, including the
    /// original temporary default state during transition dispatch.
    pub(crate) fn update_referee_command_full(
        &mut self,
        ctx: &mut RefereeContext<'_>,
        command: referee::Command,
        next_command: Option<referee::Command>,
        action_time_s: Option<f64>,
    ) {
        self.last_command = command;
        self.command_counter = self.command_counter.wrapping_add(1);
        let msg =
            crate::packets::referee_packet(self, ctx.now, command, next_command, action_time_s);
        self.referee_message.push_back(msg);
    }

    /// Record a simulator-internal referee event (foul, violation, goal, …) so the
    /// executor can surface it in the announcer feed. Drained via
    /// [`take_referee_events`].
    pub(crate) fn push_referee_event(&mut self, kind: RefereeMessage, team: Option<TeamColor>) {
        dies_core::debug_string("RefereeMessage", kind.to_string());
        self.referee_events
            .push_back(SimRefereeEvent { kind, team });
    }

    /// Raise a recurring "ball out of play" foul (§8.4.3 — e.g.
    /// [`RefereeMessage::DefenderTooCloseToBall`]) subject to the rule's
    /// per-team 2-second grace period: "Each foul has a grace period of 2
    /// seconds per team until it is raised again. … If a robot keeps committing
    /// a foul, it will be punished again after the grace period." The position
    /// checks run every physics frame, so without this gate one sustained
    /// violation (e.g. a keeper pinned by a ball parked behind its own goal line
    /// during an opponent free kick) emits hundreds of identical events. The
    /// grace collapses that to one event per 2 s, matching what a real autoref
    /// reports — and what downstream counters/digests should see.
    pub(crate) fn raise_foul(
        &mut self,
        ctx: &mut RefereeContext<'_>,
        kind: RefereeMessage,
        team: Option<TeamColor>,
    ) {
        let key = (kind, team);
        let recently_raised = self
            .foul_last_raised
            .get(&key)
            .is_some_and(|&last| ctx.now - last < FOUL_GRACE_SECS);
        if !recently_raised {
            self.foul_last_raised.insert(key.clone(), ctx.now);
            self.push_referee_event(key.0, key.1);
        }
    }

    /// Drain the simulator-internal referee events accumulated since the last call.
    pub(crate) fn take_referee_events(&mut self) -> Vec<SimRefereeEvent> {
        self.referee_events.drain(..).collect()
    }

    pub(crate) fn gc_message(&mut self) -> Option<Referee> {
        self.referee_message.pop_front()
    }
}

/// Physical effects requested by referee decisions, applied by Simulation.
pub(crate) enum RefereeAction {
    PlaceBall {
        x: f64,
        y: f64,
    },
    RemoveRobot {
        team: TeamColor,
        id: PlayerId,
    },
    AddRobot {
        team: TeamColor,
        id: PlayerId,
        position: Vector2,
        yaw: Angle,
    },
}

impl<'a> RefereeContext<'a> {
    pub(crate) fn new(
        config: &'a SimulationConfig,
        now: f64,
        physics: &'a PhysicsWorld,
        robots: &'a RobotFleet,
    ) -> Self {
        Self {
            config,
            now,
            physics,
            robots,
            actions: Vec::new(),
        }
    }
}
