//! Notebook host: runs the embedded JS engine (`dies-notebook`) once per frame
//! as a command source next to the strategy host.
//!
//! The boundary is data only. In: a snapshot of the world (absolute field
//! coordinates) plus per-robot executor info and raw feedback. Out: per-robot
//! commands (`PlayerControlInput` or `SkillCommand`, deserialized here), debug
//! values, log events and cell outputs. Commands are transformed into each
//! team's frame and handed to the team controllers as a [`NotebookInput`]
//! overlay, so they pass through compliance, avoidance and the player
//! controller like any strategy output.

use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, Result};
use dies_core::{
    Angle, NotebookCmd, NotebookEvent, PlayerFeedbackMsg, PlayerId, SideAssignment, TeamColor,
    Vector2, WorldData,
};
use dies_logger::worker;
use dies_notebook::NotebookEngine;
use dies_strategy_protocol::{AcquirePosition, BallAction, PassResult, SkillCommand, SkillStatus};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::{NotebookCommand, NotebookInput, PlayerControlInput};

pub use dies_notebook::PRELUDE_TYPES;

/// Per-robot executor info for the snapshot (absolute coordinates).
#[derive(Serialize)]
pub(crate) struct RobotInfo<'a> {
    pub team: TeamColor,
    pub id: PlayerId,
    /// Status of the skill executed last frame.
    pub skill: Option<SkillStatus>,
    /// Velocity setpoint the controller produced last frame (mm/s, global).
    pub target_vel: Option<Vector2>,
    pub pass: Option<PassResult>,
    pub feedback: Option<&'a PlayerFeedbackMsg>,
}

#[derive(Serialize)]
pub(crate) struct Snapshot<'a> {
    pub frame: u64,
    pub t: f64,
    pub dt: f64,
    /// Controlled team colors.
    pub teams: Vec<TeamColor>,
    pub world: &'a WorldData,
    pub robots: Vec<RobotInfo<'a>>,
}

#[derive(Deserialize)]
struct RobotRef {
    team: TeamColor,
    id: PlayerId,
}

#[derive(Deserialize)]
struct CommandOut {
    team: TeamColor,
    id: PlayerId,
    cmd: serde_json::Value,
    /// Frame the command was first sent in; the newest one wins a robot.
    #[serde(default)]
    since: Option<u64>,
}

#[derive(Deserialize)]
enum CommandJson {
    Input(PlayerControlInput),
    Skill(SkillCommand),
}

/// Firmware config-variable write (`HG_Variable` name → raw u32 bits).
#[derive(Deserialize)]
pub(crate) struct FwWrite {
    pub team: TeamColor,
    pub id: PlayerId,
    pub vars: Vec<(String, u32)>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum DebugValue {
    Num(f64),
    Str(String),
}

#[derive(Deserialize)]
struct DebugOut {
    key: String,
    value: DebugValue,
}

#[derive(Deserialize)]
struct EventOut {
    #[serde(rename = "type")]
    kind: String,
    payload: serde_json::Value,
}

#[derive(Deserialize)]
struct OutputOut {
    cell: Option<String>,
    kind: String,
    #[serde(default)]
    data: Option<serde_json::Value>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct EngineOutput {
    commands: Vec<CommandOut>,
    release: Vec<RobotRef>,
    debug: Vec<DebugOut>,
    events: Vec<EventOut>,
    outputs: Vec<OutputOut>,
    fw: Vec<FwWrite>,
}

pub(crate) struct NotebookHost {
    engine: Option<NotebookEngine>,
    /// Robots the notebook commanded last frame, so a reset/failure releases them.
    held: HashSet<(TeamColor, PlayerId)>,
    /// Releases to apply on the next frame even if the engine is gone.
    pending_release: HashSet<(TeamColor, PlayerId)>,
    events_tx: broadcast::Sender<NotebookEvent>,
    /// Firmware variable writes awaiting the radio (drained by the live loop).
    fw_writes: Vec<FwWrite>,
}

impl NotebookHost {
    pub fn new(events_tx: broadcast::Sender<NotebookEvent>) -> Self {
        Self {
            engine: None,
            held: HashSet::new(),
            pending_release: HashSet::new(),
            events_tx,
            fw_writes: Vec::new(),
        }
    }

    /// Whether the host needs a frame call (and therefore a snapshot) this frame.
    pub fn is_active(&self) -> bool {
        self.engine.is_some() || !self.pending_release.is_empty()
    }

    pub fn has_engine(&self) -> bool {
        self.engine.is_some()
    }

    /// Take the firmware variable writes issued since the last call.
    pub fn take_fw_writes(&mut self) -> Vec<FwWrite> {
        std::mem::take(&mut self.fw_writes)
    }

    /// Handle a UI command. `snapshot` builds the current snapshot JSON, used to
    /// prime a freshly created engine.
    pub fn handle_cmd(
        &mut self,
        cmd: NotebookCmd,
        frame_id: u64,
        t: f64,
        snapshot: impl FnOnce() -> String,
    ) {
        match cmd {
            NotebookCmd::Eval {
                notebook,
                cell,
                source,
                original,
                robot,
            } => {
                if worker::is_active() {
                    let payload = serde_json::json!({
                        "notebook": notebook, "cell": cell, "source": original, "robot": robot,
                    });
                    worker::log_event(frame_id, t, "notebook.eval", payload.to_string());
                }
                if self.engine.is_none() {
                    match NotebookEngine::new().and_then(|mut e| e.prime(&snapshot()).map(|_| e)) {
                        Ok(engine) => self.engine = Some(engine),
                        Err(err) => {
                            self.send(Some(&notebook), Some(cell), "error", Some(format!("engine: {err}")));
                            return;
                        }
                    }
                }
                let engine = self.engine.as_mut().expect("engine just created");
                let robot_json = serde_json::to_string(&robot).unwrap_or_else(|_| "null".into());
                let res = engine
                    .eval(&notebook, &source, &robot_json)
                    .and_then(|_| engine.flush(&notebook));
                match res {
                    Ok(out) => self.apply_between_frames(&notebook, &out, frame_id, t),
                    Err(err) => {
                        self.send(Some(&notebook), Some(cell), "error", Some(err.to_string()));
                        self.abort("cell failed to start", frame_id, t);
                    }
                }
            }
            NotebookCmd::Abort => self.abort("stopped", frame_id, t),
            NotebookCmd::Reset { notebook } => {
                let res = match self.engine.as_mut() {
                    Some(engine) => engine.reset(&notebook),
                    None => Ok(None),
                };
                match res {
                    Ok(Some(out)) => self.apply_between_frames(&notebook, &out, frame_id, t),
                    Ok(None) => {}
                    Err(err) => {
                        log::warn!("notebook reset failed, dropping engine: {err}");
                        self.engine = None;
                        self.pending_release.extend(self.held.drain());
                    }
                }
                // Its held robots disappear from the next frame's commands and are
                // released there.
                self.send(Some(&notebook), None, "reset", None);
            }
        }
    }

    /// Abort every running and queued cell and release all robots.
    pub fn abort(&mut self, reason: &str, frame_id: u64, t: f64) {
        if let Some(engine) = self.engine.as_mut() {
            match engine.abort_all(reason) {
                Ok(outs) => {
                    for (nb, out) in outs {
                        self.apply_between_frames(&nb, &out, frame_id, t);
                    }
                }
                Err(err) => {
                    // The engine is wedged; drop it. Held robots are released below.
                    log::warn!("notebook abort failed, dropping engine: {err}");
                    self.engine = None;
                    self.send(None, None, "error", Some(format!("engine reset: {err}")));
                }
            }
        }
        self.pending_release.extend(self.held.drain());
    }

    /// Run one frame. Returns the per-team overlay for the team controllers.
    pub fn frame(
        &mut self,
        snapshot_json: &str,
        side_assignment: SideAssignment,
        frame_id: u64,
        t: f64,
    ) -> HashMap<TeamColor, NotebookInput> {
        let mut inputs: HashMap<TeamColor, NotebookInput> = HashMap::new();

        let started = std::time::Instant::now();
        let result = match self.engine.as_mut() {
            Some(engine) => engine.frame(snapshot_json).and_then(|outs| {
                outs.into_iter()
                    .map(|(nb, out)| {
                        serde_json::from_str::<EngineOutput>(&out)
                            .map(|o| (nb, o))
                            .map_err(|e| anyhow!("bad engine output: {e}"))
                    })
                    .collect::<Result<Vec<_>>>()
            }),
            None => Ok(Vec::new()),
        };
        if self.engine.is_some() {
            // Wall-clock cost of the JS side of this frame (diagnostic only).
            dies_core::debug_value("notebook.frame_ms", started.elapsed().as_secs_f64() * 1e3);
        }
        let outs = match result {
            Ok(outs) => outs,
            Err(err) => {
                self.send(None, None, "error", Some(format!("{err}")));
                self.abort("engine error", frame_id, t);
                Vec::new()
            }
        };

        // Merge notebooks. When several hold the same robot, the most recently
        // issued command wins (ties: later notebook id); the losers forget their
        // hold so it can't take the robot back later.
        let mut out = EngineOutput::default();
        let mut cmd_origin = Vec::new();
        let mut tagged_outputs = Vec::new();
        let mut winner: HashMap<(TeamColor, PlayerId), (u64, usize)> = HashMap::new();
        for (nb, o) in outs {
            for c in o.commands {
                let since = c.since.unwrap_or(u64::MAX);
                let idx = out.commands.len();
                match winner.get(&(c.team, c.id)) {
                    Some(&(best, _)) if best > since => {}
                    _ => {
                        winner.insert((c.team, c.id), (since, idx));
                    }
                }
                cmd_origin.push(nb.clone());
                out.commands.push(c);
            }
            out.release.extend(o.release);
            out.debug.extend(o.debug);
            out.events.extend(o.events);
            self.fw_writes.extend(o.fw);
            tagged_outputs.extend(o.outputs.into_iter().map(|x| (Some(nb.clone()), x)));
        }
        let mut losers: HashMap<String, Vec<serde_json::Value>> = HashMap::new();
        let mut kept = Vec::new();
        let mut kept_origin = Vec::new();
        for (idx, (c, nb)) in out.commands.into_iter().zip(cmd_origin).enumerate() {
            if winner.get(&(c.team, c.id)).map(|w| w.1) == Some(idx) {
                kept.push(c);
                kept_origin.push(nb);
            } else {
                losers
                    .entry(nb)
                    .or_default()
                    .push(serde_json::json!({ "team": c.team, "id": c.id }));
            }
        }
        out.commands = kept;
        let cmd_origin = kept_origin;
        if let Some(engine) = self.engine.as_mut() {
            for (nb, robots) in losers {
                let json = serde_json::Value::Array(robots).to_string();
                if let Err(e) = engine.drop_held(&nb, &json) {
                    log::warn!("notebook {nb}: drop_held failed: {e}");
                }
            }
        }

        // Parse commands first: one bad command (e.g. a typo'd field) aborts the
        // notebook instead of silently dropping the robot's input.
        let mut parsed = Vec::with_capacity(out.commands.len());
        let mut parse_error = None;
        for (c, nb) in out.commands.into_iter().zip(cmd_origin) {
            match serde_json::from_value::<CommandJson>(c.cmd) {
                Ok(cmd) => parsed.push((c.team, c.id, cmd)),
                Err(e) => {
                    parse_error = Some((nb, format!(
                        "robot {} ({:?}): invalid command: {e}",
                        c.id, c.team
                    )));
                    break;
                }
            }
        }
        if let Some((nb, msg)) = parse_error {
            self.send(Some(&nb), None, "error", Some(msg));
            self.abort("invalid command", frame_id, t);
            parsed.clear();
        }

        let mut held = HashSet::new();
        for (team, id, cmd) in parsed {
            let cmd = match cmd {
                CommandJson::Input(input) => {
                    NotebookCommand::Input(input.to_team_frame(side_assignment, team))
                }
                CommandJson::Skill(skill) => {
                    NotebookCommand::Skill(skill_to_team_frame(skill, side_assignment, team))
                }
            };
            inputs.entry(team).or_default().commands.insert(id, cmd);
            held.insert((team, id));
        }
        for r in out.release {
            self.pending_release.insert((r.team, r.id));
        }
        // Robots held last frame but not this one were released.
        for key in self.held.difference(&held) {
            self.pending_release.insert(*key);
        }
        for (team, id) in self.pending_release.drain() {
            if !held.contains(&(team, id)) {
                inputs.entry(team).or_default().released.insert(id);
            }
        }
        self.held = held;

        for d in out.debug {
            match d.value {
                DebugValue::Num(v) => dies_core::debug_value(d.key, v),
                DebugValue::Str(s) => dies_core::debug_string(d.key, s),
            }
        }
        self.log_events(out.events, frame_id, t);
        self.forward_outputs(tagged_outputs);
        inputs
    }

    fn apply_between_frames(&mut self, notebook: &str, out: &str, frame_id: u64, t: f64) {
        match serde_json::from_str::<EngineOutput>(out) {
            Ok(out) => {
                self.log_events(out.events, frame_id, t);
                let outputs = out.outputs.into_iter().map(|o| (Some(notebook.to_string()), o));
                self.forward_outputs(outputs.collect());
            }
            Err(e) => self.send(
                Some(notebook),
                None,
                "error",
                Some(format!("bad engine output: {e}")),
            ),
        }
    }

    fn log_events(&self, events: Vec<EventOut>, frame_id: u64, t: f64) {
        if !worker::is_active() {
            return;
        }
        for ev in events {
            worker::log_event(frame_id, t, ev.kind, ev.payload.to_string());
        }
    }

    fn forward_outputs(&self, outputs: Vec<(Option<String>, OutputOut)>) {
        for (nb, o) in outputs {
            let data = o.data.map(|d| d.to_string());
            self.send(nb.as_deref(), o.cell, &o.kind, data);
        }
    }

    fn send(&self, notebook: Option<&str>, cell: Option<String>, kind: &str, data: Option<String>) {
        let _ = self.events_tx.send(NotebookEvent {
            notebook: notebook.map(str::to_string),
            cell,
            kind: kind.to_string(),
            data,
        });
    }
}

/// Transform a field-frame (absolute) skill command into `color`'s team frame.
/// Exhaustive on purpose: a new `SkillCommand` variant fails to compile here
/// until its geometric fields are handled.
pub(crate) fn skill_to_team_frame(
    cmd: SkillCommand,
    side: SideAssignment,
    color: TeamColor,
) -> SkillCommand {
    if side.attacking_direction_sign(color) > 0.0 {
        return cmd;
    }
    let v = |p: Vector2| side.transform_vec2(color, &p);
    let a = |h: Angle| side.transform_angle(color, h);
    match cmd {
        SkillCommand::GoToPos {
            position,
            heading,
            hold_ground,
            avoid_robots,
        } => SkillCommand::GoToPos {
            position: v(position),
            heading: heading.map(a),
            hold_ground,
            avoid_robots,
        },
        SkillCommand::GoToBounded {
            position,
            heading,
            bounds,
        } => SkillCommand::GoToBounded {
            position: v(position),
            heading: heading.map(a),
            bounds: bounds.mirror_x(),
        },
        SkillCommand::Dribble {
            target_pos,
            target_heading,
        } => SkillCommand::Dribble {
            target_pos: v(target_pos),
            target_heading: a(target_heading),
        },
        SkillCommand::Shoot { target } => SkillCommand::Shoot { target: v(target) },
        SkillCommand::ReflexReceive {
            from_pos,
            intercept_pos,
            target,
            capture_limit,
        } => SkillCommand::ReflexReceive {
            from_pos: v(from_pos),
            intercept_pos: v(intercept_pos),
            target: v(target),
            capture_limit,
        },
        SkillCommand::Pass {
            partner,
            role,
            target_hint,
            forward_to,
            seq,
        } => SkillCommand::Pass {
            partner,
            role,
            target_hint: target_hint.map(v),
            forward_to: forward_to.map(v),
            seq,
        },
        SkillCommand::Snatch { release_hint } => SkillCommand::Snatch {
            release_hint: release_hint.map(v),
        },
        SkillCommand::HandleBall {
            action,
            acquire,
            magnet,
        } => SkillCommand::HandleBall {
            action: match action {
                BallAction::Shoot { target } => BallAction::Shoot { target: v(target) },
                BallAction::Strike {
                    target,
                    acquire_first,
                } => BallAction::Strike {
                    target: v(target),
                    acquire_first,
                },
                BallAction::Carry { to, heading } => BallAction::Carry {
                    to: v(to),
                    heading: a(heading),
                },
                BallAction::Hold { heading } => BallAction::Hold {
                    heading: a(heading),
                },
            },
            acquire: match acquire {
                AcquirePosition::Heading(h) => AcquirePosition::Heading(a(h)),
                other => other,
            },
            magnet,
        },
        SkillCommand::Stop => SkillCommand::Stop,
    }
}

/// Error helper for building snapshots.
pub(crate) fn to_json<T: Serialize>(value: &T) -> Result<String> {
    serde_json::to_string(value).map_err(|e| anyhow!("snapshot: {e}"))
}
