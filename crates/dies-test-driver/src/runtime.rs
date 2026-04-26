//! TestDriver — the outer shell. Owns the Boa context, loads scenarios, drives
//! the async event loop one tick per executor frame.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use boa_engine::builtins::promise::PromiseState;
use boa_engine::object::builtins::JsPromise;
use boa_engine::{js_string, Context, JsError, JsNativeError, JsObject, JsString, JsValue, Source};
use dies_core::{Angle, PlayerId, TeamColor, TeamData, Vector2};
use dies_strategy_protocol::SkillStatus;
use serde::{Deserialize, Serialize};
use typeshare::typeshare;

use crate::bridge::{register_api, InnerState, StateRef, TestFrameOutput, Waker};
use crate::capture::CapturedSample;
use crate::log_bus::{LogBus, TestLogEntry, TestLogLevel};
use crate::primitives::ExcitationSample;

/// The environment the driver is running against. Scenarios gate on this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestEnv {
    Sim,
    Real,
    Either,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScenarioEnv {
    Sim,
    Real,
    Either,
}

impl ScenarioEnv {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "sim" => Some(Self::Sim),
            "real" | "live" => Some(Self::Real),
            "either" | "any" | "both" => Some(Self::Either),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioMeta {
    pub name: String,
    pub team: TeamColor,
    pub robots: Vec<u32>,
    pub env: ScenarioEnv,
}

impl ScenarioMeta {
    pub fn env_compatible(&self, env: TestEnv) -> bool {
        matches!(
            (self.env, env),
            (ScenarioEnv::Sim, TestEnv::Sim)
                | (ScenarioEnv::Real, TestEnv::Real)
                | (ScenarioEnv::Either, _)
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[typeshare]
pub struct ScenarioArtifact {
    pub tag: String,
    pub value_json: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "state", content = "data")]
#[typeshare]
pub enum TestStatus {
    Idle,
    Starting,
    Running { name: String },
    Completed { artifacts: Vec<ScenarioArtifact> },
    Failed { error: String },
    Aborted,
}

#[derive(Debug, Clone, Default)]
pub struct PlayerControlSlot {
    pub position: Option<Vector2>,
    pub yaw: Option<Angle>,
    pub vel_global: Option<Vector2>,
    pub vel_local: Option<Vector2>,
    pub angular_velocity: Option<f64>,
    pub dribble: f64,
    pub fan: Option<f64>,
    pub kick_force: Option<f64>,
    pub kick_speed: Option<f64>,
    pub disarm_kicker: bool,
}

pub struct TestDriver {
    ctx: Context,
    state: StateRef,
    entry_promise: Option<JsPromise>,
    pending_entry: Option<(JsObject, JsObject)>,
    status: TestStatus,
    scenario_path: Option<PathBuf>,
}

impl TestDriver {
    pub fn new(team: TeamColor, env: TestEnv, log: LogBus) -> anyhow::Result<Self> {
        let mut ctx = Context::default();
        let state = Rc::new(RefCell::new(InnerState::new(team, env, log.clone())));
        register_api(&mut ctx, state.clone()).map_err(js_err)?;
        Ok(Self {
            ctx,
            state,
            entry_promise: None,
            pending_entry: None,
            status: TestStatus::Idle,
            scenario_path: None,
        })
    }

    pub fn load_and_start(&mut self, path: &Path) -> anyhow::Result<ScenarioMeta> {
        self.scenario_path = Some(path.to_path_buf());
        let source = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("failed to read scenario {}: {}", path.display(), e))?;
        let stdlib = include_str!("../js_stdlib.js");
        let combined = format!("{}\n// === scenario ===\n{}", stdlib, source);
        self.ctx
            .eval(Source::from_bytes(&combined))
            .map_err(js_err)?;

        let meta_val = self
            .ctx
            .global_object()
            .get(js_string!("scenario"), &mut self.ctx)
            .map_err(js_err)?;
        let meta = parse_scenario_meta(&meta_val, &mut self.ctx).map_err(js_err)?;
        self.state.borrow_mut().scenario = Some(meta.clone());

        let env = self.state.borrow().env;
        if !meta.env_compatible(env) {
            return Err(anyhow::anyhow!(
                "scenario env `{:?}` not compatible with driver env `{:?}`",
                meta.env,
                env
            ));
        }
        if meta.team != self.state.borrow().team_color {
            self.state.borrow_mut().team_color = meta.team;
        }

        let run_val = self
            .ctx
            .global_object()
            .get(js_string!("run"), &mut self.ctx)
            .map_err(js_err)?;
        let run_obj = run_val
            .as_object()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("scenario must export an async `run` function"))?;
        if !run_obj.is_callable() {
            return Err(anyhow::anyhow!("`run` is not callable"));
        }

        let args_obj = boa_engine::JsObject::with_object_proto(self.ctx.intrinsics());
        for name in ["team", "world", "log", "sysid"] {
            let v = self
                .ctx
                .global_object()
                .get(JsString::from(name), &mut self.ctx)
                .map_err(js_err)?;
            args_obj
                .set(JsString::from(name), v, false, &mut self.ctx)
                .map_err(js_err)?;
        }

        // Defer the entry call until the first `tick()` so that `world.t` is
        // initialized from real team data before any JS runs. Otherwise the JS
        // captures `world.t = 0.0` for things like `moveTo` deadlines, and the
        // deadline check fires immediately when the world jumps to the
        // executor's monotonic clock on the first tick.
        self.pending_entry = Some((run_obj, args_obj));
        self.status = TestStatus::Running {
            name: meta.name.clone(),
        };
        self.state.borrow().log.emit(TestLogEntry {
            level: TestLogLevel::Info,
            tag: None,
            message: format!("scenario `{}` started", meta.name),
            value_json: None,
            ts_ms: crate::log_bus::now_ms(),
        });
        Ok(meta)
    }

    pub fn abort(&mut self) {
        self.state.borrow_mut().aborted = true;
    }

    pub fn status(&self) -> TestStatus {
        self.status.clone()
    }

    pub fn scenario(&self) -> Option<ScenarioMeta> {
        self.state.borrow().scenario.clone()
    }

    pub fn team_color(&self) -> TeamColor {
        self.state.borrow().team_color
    }

    pub fn set_skill_statuses(
        &mut self,
        statuses: std::collections::HashMap<PlayerId, SkillStatus>,
    ) {
        self.state.borrow_mut().skill_statuses = statuses;
    }

    /// Push the executor's per-player computed velocity setpoints (global frame,
    /// mm/s) into the driver. Called once per tick from the host so recordings
    /// can capture the controller's actual cmd during position-controlled motion.
    pub fn set_actual_cmds_global(
        &mut self,
        cmds: std::collections::HashMap<PlayerId, Vector2>,
    ) {
        self.state.borrow_mut().actual_cmds_global = cmds;
    }

    pub fn tick(&mut self, team_data: &TeamData) -> TestFrameOutput {
        self.state.borrow_mut().world.update_from(team_data);

        self.start_pending_entry();

        let aborted_now = self.state.borrow().aborted
            && !matches!(self.status, TestStatus::Aborted | TestStatus::Failed { .. });
        if aborted_now {
            self.reject_all_wakers("AbortError: scenario aborted");
            self.status = TestStatus::Aborted;
        }

        self.poll_wakers();
        self.state.borrow_mut().tick_recordings();
        self.ctx.run_jobs();
        self.check_entry_promise();

        let mut st = self.state.borrow_mut();
        let sim_cmds = std::mem::take(&mut st.pending_sim_cmds);
        TestFrameOutput {
            skill_commands: st.slots.skill.clone(),
            player_roles: st.slots.roles.clone(),
            direct_inputs: st.slots.direct.clone(),
            sim_commands: sim_cmds,
            cleared_direct: Vec::new(),
        }
    }

    fn start_pending_entry(&mut self) {
        let Some((run_obj, args_obj)) = self.pending_entry.take() else {
            return;
        };
        match run_obj.call(&JsValue::undefined(), &[args_obj.into()], &mut self.ctx) {
            Ok(ret) => {
                let promise = match ret.as_object() {
                    Some(obj) => JsPromise::from_object(obj.clone())
                        .unwrap_or_else(|_| JsPromise::resolve(ret.clone(), &mut self.ctx)),
                    None => JsPromise::resolve(ret.clone(), &mut self.ctx),
                };
                self.entry_promise = Some(promise);
            }
            Err(e) => {
                let err_text = js_value_message(&e.to_opaque(&mut self.ctx), &mut self.ctx);
                self.state.borrow().log.emit(TestLogEntry {
                    level: TestLogLevel::Error,
                    tag: None,
                    message: format!("scenario failed: {}", err_text),
                    value_json: None,
                    ts_ms: crate::log_bus::now_ms(),
                });
                self.status = TestStatus::Failed { error: err_text };
            }
        }
    }

    fn check_entry_promise(&mut self) {
        let Some(promise) = self.entry_promise.as_ref() else {
            return;
        };
        match promise.state() {
            PromiseState::Pending => {}
            PromiseState::Fulfilled(_) => {
                if matches!(self.status, TestStatus::Running { .. }) {
                    let artifacts = std::mem::take(&mut self.state.borrow_mut().record_artifacts)
                        .into_iter()
                        .map(|(tag, value_json)| ScenarioArtifact { tag, value_json })
                        .collect();
                    let name = match &self.status {
                        TestStatus::Running { name } => name.clone(),
                        _ => "scenario".into(),
                    };
                    self.state.borrow().log.emit(TestLogEntry {
                        level: TestLogLevel::Info,
                        tag: None,
                        message: format!("scenario `{}` completed", name),
                        value_json: None,
                        ts_ms: crate::log_bus::now_ms(),
                    });
                    self.status = TestStatus::Completed { artifacts };
                }
            }
            PromiseState::Rejected(err) => {
                let err_text = js_value_message(&err, &mut self.ctx);
                self.state.borrow().log.emit(TestLogEntry {
                    level: TestLogLevel::Error,
                    tag: None,
                    message: format!("scenario failed: {}", err_text),
                    value_json: None,
                    ts_ms: crate::log_bus::now_ms(),
                });
                if matches!(self.status, TestStatus::Aborted) {
                } else {
                    self.status = TestStatus::Failed { error: err_text };
                }
            }
        }
    }

    fn reject_all_wakers(&mut self, msg: &str) {
        let wakers: Vec<(u64, Waker)> = self.state.borrow_mut().wakers.drain().collect();
        for (_id, w) in wakers {
            let err_val: JsValue = JsString::from(msg).into();
            let _ = match w {
                Waker::Sleep { resolve, .. } => {
                    resolve.call(&JsValue::undefined(), &[err_val], &mut self.ctx)
                }
                Waker::MoveTo { reject, .. }
                | Waker::WaitStopped { reject, .. }
                | Waker::SkillDone { reject, .. }
                | Waker::WaitUntil { reject, .. } => {
                    reject.call(&JsValue::undefined(), &[err_val], &mut self.ctx)
                }
                Waker::Excite { resolve, .. } => {
                    resolve.call(&JsValue::undefined(), &[err_val], &mut self.ctx)
                }
                Waker::CaptureActive { resolve, .. } => {
                    resolve.call(&JsValue::undefined(), &[err_val], &mut self.ctx)
                }
            };
        }
    }

    fn poll_wakers(&mut self) {
        let drained: Vec<(u64, Waker)> = {
            let mut st = self.state.borrow_mut();
            st.wakers.drain().collect()
        };
        let mut to_keep: Vec<(u64, Waker)> = Vec::with_capacity(drained.len());

        for (id, waker) in drained.into_iter() {
            match self.evaluate_waker(waker) {
                WakerAction::Keep(w) => to_keep.push((id, w)),
                WakerAction::Done => {}
            }
        }
        let mut st = self.state.borrow_mut();
        for (id, w) in to_keep {
            st.wakers.insert(id, w);
        }
    }

    fn evaluate_waker(&mut self, waker: Waker) -> WakerAction {
        let now = self.state.borrow().world.t;
        match waker {
            Waker::Sleep {
                deadline_s,
                resolve,
            } => {
                if now >= deadline_s {
                    let _ = resolve.call(
                        &JsValue::undefined(),
                        &[JsValue::undefined()],
                        &mut self.ctx,
                    );
                    WakerAction::Done
                } else {
                    WakerAction::Keep(Waker::Sleep {
                        deadline_s,
                        resolve,
                    })
                }
            }
            Waker::MoveTo {
                player,
                target,
                tol,
                vel_thresh,
                resolve,
                reject,
                deadline_s,
            } => {
                let snap = self.state.borrow().snapshot_player(player);
                if let Some(s) = snap {
                    let dist = (s.position - target).norm();
                    let speed = s.velocity.norm();
                    if dist < tol && speed < vel_thresh {
                        let _ = resolve.call(
                            &JsValue::undefined(),
                            &[JsValue::undefined()],
                            &mut self.ctx,
                        );
                        return WakerAction::Done;
                    }
                }
                if let Some(d) = deadline_s {
                    if now > d {
                        let msg: JsValue =
                            JsString::from(format!("moveTo timeout (player {})", player.as_u32()))
                                .into();
                        let _ = reject.call(&JsValue::undefined(), &[msg], &mut self.ctx);
                        return WakerAction::Done;
                    }
                }
                WakerAction::Keep(Waker::MoveTo {
                    player,
                    target,
                    tol,
                    vel_thresh,
                    resolve,
                    reject,
                    deadline_s,
                })
            }
            Waker::WaitStopped {
                player,
                thresh,
                resolve,
                reject,
                deadline_s,
            } => {
                let snap = self.state.borrow().snapshot_player(player);
                if let Some(s) = snap {
                    if s.velocity.norm() < thresh {
                        let _ = resolve.call(
                            &JsValue::undefined(),
                            &[JsValue::undefined()],
                            &mut self.ctx,
                        );
                        return WakerAction::Done;
                    }
                }
                if let Some(d) = deadline_s {
                    if now > d {
                        let msg: JsValue = JsString::from("waitStopped timeout".to_string()).into();
                        let _ = reject.call(&JsValue::undefined(), &[msg], &mut self.ctx);
                        return WakerAction::Done;
                    }
                }
                WakerAction::Keep(Waker::WaitStopped {
                    player,
                    thresh,
                    resolve,
                    reject,
                    deadline_s,
                })
            }
            Waker::SkillDone {
                player,
                resolve,
                reject,
            } => {
                let status = self
                    .state
                    .borrow()
                    .skill_statuses
                    .get(&player)
                    .copied()
                    .unwrap_or(SkillStatus::Running);
                match status {
                    SkillStatus::Succeeded => {
                        let _ = resolve.call(
                            &JsValue::undefined(),
                            &[JsValue::undefined()],
                            &mut self.ctx,
                        );
                        WakerAction::Done
                    }
                    SkillStatus::Failed => {
                        let msg: JsValue =
                            JsString::from(format!("skill failed (player {})", player.as_u32()))
                                .into();
                        let _ = reject.call(&JsValue::undefined(), &[msg], &mut self.ctx);
                        WakerAction::Done
                    }
                    SkillStatus::Running | SkillStatus::Idle => {
                        WakerAction::Keep(Waker::SkillDone {
                            player,
                            resolve,
                            reject,
                        })
                    }
                }
            }
            Waker::Excite {
                player,
                profile,
                start_s,
                hold_yaw,
                resolve,
            } => {
                let t_rel = now - start_s;
                let dur = profile.duration();
                if t_rel >= dur {
                    {
                        let mut st = self.state.borrow_mut();
                        st.slots
                            .route_direct(player, PlayerControlSlot::default(), "exciteDone");
                    }
                    let _ = resolve.call(
                        &JsValue::undefined(),
                        &[JsValue::undefined()],
                        &mut self.ctx,
                    );
                    WakerAction::Done
                } else {
                    let sample = profile.sample(t_rel);
                    self.state
                        .borrow_mut()
                        .apply_excitation_tick(player, sample, hold_yaw);
                    WakerAction::Keep(Waker::Excite {
                        player,
                        profile,
                        start_s,
                        hold_yaw,
                        resolve,
                    })
                }
            }
            Waker::CaptureActive {
                player,
                excitation,
                rate_hz,
                duration,
                start_s,
                last_sample_s,
                mut buffer,
                hold_yaw,
                resolve,
            } => {
                let t_rel = now - start_s;
                if let Some(prof) = &excitation {
                    let sample = prof.sample(t_rel.max(0.0));
                    self.state
                        .borrow_mut()
                        .apply_excitation_tick(player, sample, hold_yaw);
                }
                let interval = if rate_hz > 0.0 { 1.0 / rate_hz } else { 0.05 };
                let snap = self.state.borrow().snapshot_player(player);
                let mut new_last = last_sample_s;
                if now - last_sample_s >= interval {
                    if let Some(s) = snap {
                        let _ = s.yaw.inv().rotate_vector(&s.velocity);
                        let cmd_sample = excitation
                            .as_ref()
                            .map(|p| p.sample(t_rel.max(0.0)))
                            .unwrap_or_else(ExcitationSample::default);
                        buffer.push(CapturedSample {
                            t: now,
                            cmd_x: cmd_sample.vel.x,
                            cmd_y: cmd_sample.vel.y,
                            heading: s.yaw.radians(),
                            pos_x: s.position.x,
                            pos_y: s.position.y,
                            vel_x: s.velocity.x,
                            vel_y: s.velocity.y,
                            tags: Vec::new(),
                        });
                    }
                    new_last = now;
                }
                if t_rel >= duration {
                    {
                        let mut st = self.state.borrow_mut();
                        st.slots
                            .route_direct(player, PlayerControlSlot::default(), "captureDone");
                    }
                    let arr = match crate::bridge::samples_to_js_array(
                        &buffer.samples,
                        &[],
                        &mut self.ctx,
                    ) {
                        Ok(a) => JsValue::from(a),
                        Err(_) => JsValue::undefined(),
                    };
                    let _ = resolve.call(&JsValue::undefined(), &[arr], &mut self.ctx);
                    WakerAction::Done
                } else {
                    WakerAction::Keep(Waker::CaptureActive {
                        player,
                        excitation,
                        rate_hz,
                        duration,
                        start_s,
                        last_sample_s: new_last,
                        buffer,
                        hold_yaw,
                        resolve,
                    })
                }
            }
            Waker::WaitUntil {
                predicate,
                poll_s,
                next_poll_s,
                resolve,
                reject,
                deadline_s,
            } => {
                if let Some(d) = deadline_s {
                    if now > d {
                        let msg: JsValue = JsString::from("waitUntil timeout".to_string()).into();
                        let _ = reject.call(&JsValue::undefined(), &[msg], &mut self.ctx);
                        return WakerAction::Done;
                    }
                }
                if now < next_poll_s {
                    return WakerAction::Keep(Waker::WaitUntil {
                        predicate,
                        poll_s,
                        next_poll_s,
                        resolve,
                        reject,
                        deadline_s,
                    });
                }
                let r = predicate.call(&JsValue::undefined(), &[], &mut self.ctx);
                let truthy = match r {
                    Ok(v) => v.to_boolean(),
                    Err(_) => false,
                };
                if truthy {
                    let _ = resolve.call(
                        &JsValue::undefined(),
                        &[JsValue::undefined()],
                        &mut self.ctx,
                    );
                    WakerAction::Done
                } else {
                    WakerAction::Keep(Waker::WaitUntil {
                        predicate,
                        poll_s,
                        next_poll_s: now + poll_s,
                        resolve,
                        reject,
                        deadline_s,
                    })
                }
            }
        }
    }
}

enum WakerAction {
    Keep(Waker),
    Done,
}

fn parse_scenario_meta(v: &JsValue, ctx: &mut Context) -> Result<ScenarioMeta, JsError> {
    let obj = v.as_object().cloned().ok_or_else(|| {
        JsNativeError::typ()
            .with_message("scenario metadata missing (expected globalThis.scenario)")
    })?;
    let name = obj
        .get(js_string!("name"), ctx)?
        .to_string(ctx)?
        .to_std_string_escaped();
    let team_str = obj
        .get(js_string!("team"), ctx)?
        .to_string(ctx)?
        .to_std_string_escaped();
    let team = match team_str.to_lowercase().as_str() {
        "blue" => TeamColor::Blue,
        "yellow" => TeamColor::Yellow,
        _ => {
            return Err(JsNativeError::typ()
                .with_message(format!("unknown team: {}", team_str))
                .into())
        }
    };
    let env_str = obj
        .get(js_string!("env"), ctx)?
        .to_string(ctx)?
        .to_std_string_escaped();
    let env = ScenarioEnv::parse(&env_str)
        .ok_or_else(|| JsNativeError::typ().with_message(format!("unknown env: {}", env_str)))?;
    let robots_val = obj.get(js_string!("robots"), ctx)?;
    let mut robots = Vec::new();
    if let Some(robots_obj) = robots_val.as_object().cloned() {
        if let Ok(arr) = boa_engine::object::builtins::JsArray::from_object(robots_obj) {
            let len = arr.length(ctx)?;
            for i in 0..len {
                let n = arr.get(i, ctx)?.to_number(ctx)? as u32;
                robots.push(n);
            }
        }
    }
    Ok(ScenarioMeta {
        name,
        team,
        robots,
        env,
    })
}

fn js_err(e: JsError) -> anyhow::Error {
    anyhow::anyhow!("JS error: {}", e)
}

fn js_value_message(v: &JsValue, ctx: &mut Context) -> String {
    if let Some(obj) = v.as_object() {
        if let Ok(msg) = obj.get(js_string!("message"), ctx) {
            if !msg.is_undefined() {
                return msg
                    .to_string(ctx)
                    .map(|s| s.to_std_string_escaped())
                    .unwrap_or_default();
            }
        }
    }
    v.to_string(ctx)
        .map(|s| s.to_std_string_escaped())
        .unwrap_or_else(|_| "<unprintable>".into())
}

// Unused avoid vector2 warning
#[allow(dead_code)]
fn _touch(_v: Vector2) {}
