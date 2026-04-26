//! Rust↔JS bridge state + shared types.
//!
//! `InnerState` is shared by cloning an `Rc<RefCell<InnerState>>` into every
//! host function closure. Boa's GC won't trace inside these closures (we use
//! `unsafe from_closure`), which is fine because nothing in `InnerState`
//! participates in a GC cycle back to itself. Rust-side Rc refcounting keeps
//! the `JsFunction` resolvers alive for as long as we need them.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use boa_engine::object::builtins::{JsArray, JsFunction, JsPromise};
use boa_engine::object::ObjectInitializer;
use boa_engine::property::{Attribute, PropertyKey};
use boa_engine::value::Type;
use boa_engine::{
    js_string, Context, JsError, JsNativeError, JsObject, JsResult, JsString, JsValue,
    NativeFunction,
};
use dies_core::{
    Angle, DebugSubscriber, DebugValue, FieldGeometry, PlayerId, SimulatorCmd, SysStatus,
    TeamColor, TeamData, Vector2,
};
use dies_strategy_protocol::{SkillCommand, SkillStatus};

use crate::capture::{self, samples_to_csv, CaptureBuffer, CapturedSample, Recording};
use crate::log_bus::{now_ms, LogBus, TestLogEntry, TestLogLevel};
use crate::primitives::{ExcitationProfile, ExcitationSample};
use crate::runtime::{PlayerControlSlot, ScenarioMeta, TestEnv};

/// Output of a single driver tick — deltas the executor should apply this frame.
#[derive(Debug, Default)]
pub struct TestFrameOutput {
    /// Skill commands to feed to `StrategyInput.skill_commands` for the active team.
    pub skill_commands: HashMap<PlayerId, Option<SkillCommand>>,
    /// Per-player roles (string label) for debug display.
    pub player_roles: HashMap<PlayerId, String>,
    /// Direct PlayerControlInput slots to feed into `manual_override`.
    pub direct_inputs: HashMap<PlayerId, PlayerControlSlot>,
    /// Simulator commands queued by the script (teleport/addRobot/etc.).
    pub sim_commands: Vec<SimulatorCmd>,
    /// Players whose direct slot was cleared this tick; remove from `manual_override`.
    pub cleared_direct: Vec<PlayerId>,
}

/// Per-player input slot maintained by the driver. A player's current entry is
/// re-emitted every tick until the script replaces or clears it.
#[derive(Debug, Clone, Default)]
pub struct SlotStore {
    pub direct: HashMap<PlayerId, PlayerControlSlot>,
    pub skill: HashMap<PlayerId, Option<SkillCommand>>,
    pub roles: HashMap<PlayerId, String>,
}

impl SlotStore {
    /// Mark the player as using the skill path — clear the direct slot if present.
    pub fn route_skill(&mut self, player: PlayerId, cmd: SkillCommand, role_tag: &str) {
        self.direct.remove(&player);
        self.skill.insert(player, Some(cmd));
        self.roles.insert(player, role_tag.to_string());
    }

    /// Mark the player as using the direct path — clear the skill slot.
    pub fn route_direct(&mut self, player: PlayerId, slot: PlayerControlSlot, role_tag: &str) {
        self.skill.remove(&player);
        self.direct.insert(player, slot);
        self.roles.insert(player, role_tag.to_string());
    }

    pub fn clear_player(&mut self, player: PlayerId) {
        self.direct.remove(&player);
        self.skill.remove(&player);
        self.roles.remove(&player);
    }
}

/// World snapshot exposed to JS as an object `world`. Updated in place each tick.
#[derive(Debug, Clone, Default)]
pub struct WorldSnap {
    pub t: f64,
    pub dt: f64,
    pub ball: Option<(Vector2, Vector2)>,
    pub players: HashMap<PlayerId, PlayerSnap>,
    pub field: Option<FieldGeometry>,
}

#[derive(Debug, Clone)]
pub struct PlayerSnap {
    pub position: Vector2,
    pub velocity: Vector2,
    pub yaw: Angle,
    pub angular_speed: f64,
    /// Latest basestation feedback status. `None` means we have never received
    /// telemetry for this robot — used by `team.autoRobot()` in real mode to
    /// only pick robots that are actually responsive.
    pub primary_status: Option<SysStatus>,
}

impl PlayerSnap {
    /// True if the robot has both vision (it's in the snapshot at all) and a
    /// usable basestation reply. Used as the "auto-pickable" filter.
    pub fn is_basestation_responsive(&self) -> bool {
        match self.primary_status {
            None | Some(SysStatus::NoReply) | Some(SysStatus::NotInstalled) => false,
            Some(_) => true,
        }
    }
}

impl WorldSnap {
    pub fn update_from(&mut self, data: &TeamData) {
        self.t = data.t_received;
        self.dt = data.dt;
        self.ball = data.ball.as_ref().map(|b| {
            (
                Vector2::new(b.position.x, b.position.y),
                Vector2::new(b.velocity.x, b.velocity.y),
            )
        });
        self.field = data.field_geom.clone();
        self.players.clear();
        for p in &data.own_players {
            self.players.insert(
                p.id,
                PlayerSnap {
                    position: p.position,
                    velocity: p.velocity,
                    yaw: p.raw_yaw,
                    angular_speed: p.angular_speed,
                    primary_status: p.primary_status,
                },
            );
        }
    }
}

/// Wake condition for a pending JS promise. Checked each tick by the driver.
#[derive(Debug)]
pub enum Waker {
    Sleep {
        deadline_s: f64,
        resolve: JsFunction,
    },
    MoveTo {
        player: PlayerId,
        target: Vector2,
        tol: f64,
        vel_thresh: f64,
        resolve: JsFunction,
        reject: JsFunction,
        deadline_s: Option<f64>,
    },
    WaitStopped {
        player: PlayerId,
        thresh: f64,
        resolve: JsFunction,
        reject: JsFunction,
        deadline_s: Option<f64>,
    },
    SkillDone {
        player: PlayerId,
        resolve: JsFunction,
        reject: JsFunction,
    },
    Excite {
        player: PlayerId,
        profile: ExcitationProfile,
        start_s: f64,
        /// Yaw setpoint to hold during translational excitations. `None` when
        /// the profile drives yaw itself (yaw-axis chirp/step/etc.).
        hold_yaw: Option<Angle>,
        resolve: JsFunction,
    },
    CaptureActive {
        player: PlayerId,
        excitation: Option<ExcitationProfile>,
        rate_hz: f64,
        duration: f64,
        start_s: f64,
        last_sample_s: f64,
        buffer: CaptureBuffer,
        /// See `Waker::Excite::hold_yaw`. `None` for capture-only (no excitation).
        hold_yaw: Option<Angle>,
        resolve: JsFunction,
    },
    WaitUntil {
        predicate: JsFunction,
        poll_s: f64,
        next_poll_s: f64,
        resolve: JsFunction,
        reject: JsFunction,
        deadline_s: Option<f64>,
    },
}

/// The per-scenario inner state shared with host function closures.
pub struct InnerState {
    pub scenario: Option<ScenarioMeta>,
    pub team_color: TeamColor,
    pub env: TestEnv,
    pub wakers: HashMap<u64, Waker>,
    pub next_waker_id: u64,
    pub world: WorldSnap,
    pub skill_statuses: HashMap<PlayerId, SkillStatus>,
    pub slots: SlotStore,
    pub pending_sim_cmds: Vec<SimulatorCmd>,
    pub log: LogBus,
    pub aborted: bool,
    pub record_artifacts: Vec<(String, String)>,
    /// Active free-form recordings (one per player), driven each tick from
    /// `tick_recordings`.
    pub recordings: HashMap<PlayerId, Recording>,
    /// Per-player **actual** velocity command (global frame, mm/s) computed by
    /// the executor's PlayerController on the previous tick. Pushed in by the
    /// host before `tick`. Used as the recorded `cmd` so position-controlled
    /// motion (moveTo / goToPos) records the MTP/iLQR output instead of zero.
    pub actual_cmds_global: HashMap<PlayerId, Vector2>,
    /// Process-wide debug-value subscriber. Used by `tick_recordings` to read
    /// tag values written via `dies_core::debug_value(...)`.
    pub debug_sub: DebugSubscriber,
    /// Next id handed out by `team.autoRobot()` in sim mode. Bumped on every
    /// call so consecutive autos get distinct ids regardless of when the
    /// queued `SimulatorCmd::AddRobot` lands in the next world snapshot.
    pub next_auto_id: u32,
    /// Player ids that `team.autoRobot()` has handed out (sim or real). Lets
    /// real-mode picks avoid handing the same robot out twice within a run.
    pub auto_claimed_ids: HashSet<PlayerId>,
}

pub type StateRef = Rc<RefCell<InnerState>>;

impl InnerState {
    pub fn new(team_color: TeamColor, env: TestEnv, log: LogBus) -> Self {
        Self {
            scenario: None,
            team_color,
            env,
            wakers: HashMap::new(),
            next_waker_id: 1,
            world: WorldSnap::default(),
            skill_statuses: HashMap::new(),
            slots: SlotStore::default(),
            pending_sim_cmds: Vec::new(),
            log,
            aborted: false,
            record_artifacts: Vec::new(),
            recordings: HashMap::new(),
            actual_cmds_global: HashMap::new(),
            debug_sub: DebugSubscriber::instance(),
            next_auto_id: 0,
            auto_claimed_ids: HashSet::new(),
        }
    }

    pub fn alloc_waker_id(&mut self) -> u64 {
        let id = self.next_waker_id;
        self.next_waker_id = self.next_waker_id.wrapping_add(1);
        id
    }

    pub fn emit_log(&self, level: TestLogLevel, tag: Option<&str>, msg: &str, value: Option<&str>) {
        self.log.emit(TestLogEntry {
            level,
            tag: tag.map(|s| s.to_string()),
            message: msg.to_string(),
            value_json: value.map(|s| s.to_string()),
            ts_ms: now_ms(),
        });
    }

    pub fn apply_excitation_tick(
        &mut self,
        player: PlayerId,
        sample: ExcitationSample,
        hold_yaw: Option<Angle>,
    ) {
        let mut slot = self.slots.direct.get(&player).cloned().unwrap_or_default();
        slot.position = None;
        slot.vel_local = Some(sample.vel);
        if sample.angular.abs() > 1e-9 {
            // Yaw-axis excitation: drive angular velocity directly, leave the
            // yaw setpoint unset so the PD controller doesn't fight us.
            slot.angular_velocity = Some(sample.angular);
            slot.yaw = None;
        } else {
            // Translational excitation: hold the captured heading. Without this
            // the robot drifts in yaw (asymmetric wheel response) and the
            // body→world rotation used by sysid is wrong.
            slot.angular_velocity = None;
            slot.yaw = hold_yaw;
        }
        self.slots.direct.insert(player, slot);
    }

    pub fn snapshot_player(&self, player: PlayerId) -> Option<PlayerSnap> {
        self.world.players.get(&player).cloned()
    }

    /// Sample any active recordings using the current world snapshot. Should be
    /// called once per tick after wakers (which may write to slots) have run.
    pub fn tick_recordings(&mut self) {
        let now = self.world.t;
        let player_ids: Vec<PlayerId> = self.recordings.keys().copied().collect();
        // Snapshot the debug map once per tick (only if any recording has tags)
        // so we don't clone it per player.
        let debug_snapshot = if self.recordings.values().any(|r| !r.tags.is_empty()) {
            Some(self.debug_sub.get_copy())
        } else {
            None
        };
        for player in player_ids {
            let interval = {
                let r = self.recordings.get(&player).expect("just iterated");
                if r.rate_hz > 0.0 {
                    1.0 / r.rate_hz
                } else {
                    0.05
                }
            };
            let last = self
                .recordings
                .get(&player)
                .expect("just iterated")
                .last_sample_s;
            if now - last < interval {
                continue;
            }
            let Some(snap) = self.world.players.get(&player).cloned() else {
                continue;
            };
            // Cmd resolution priority:
            //   1. Controller's actual computed velocity (global, rotated to
            //      body) — captures MTP / iLQR output during moveTo/goToPos.
            //   2. Slot vel_local (already body-frame) — set by setLocalVelocity
            //      and excitation profiles.
            //   3. Slot vel_global rotated to body — setGlobalVelocity.
            //   4. Zero.
            let (cmd_x, cmd_y) = if let Some(v_global) = self.actual_cmds_global.get(&player) {
                let body = snap.yaw.inv().rotate_vector(v_global);
                (body.x, body.y)
            } else {
                match self.slots.direct.get(&player) {
                    Some(slot) => {
                        if let Some(v) = slot.vel_local {
                            (v.x, v.y)
                        } else if let Some(v) = slot.vel_global {
                            let body = snap.yaw.inv().rotate_vector(&v);
                            (body.x, body.y)
                        } else {
                            (0.0, 0.0)
                        }
                    }
                    None => (0.0, 0.0),
                }
            };
            let tag_values: Vec<f64> = {
                let r = self.recordings.get(&player).expect("just iterated");
                if r.tags.is_empty() {
                    Vec::new()
                } else {
                    let snap_map = debug_snapshot
                        .as_ref()
                        .expect("snapshot exists when tags present");
                    r.tags
                        .iter()
                        .map(|key| match snap_map.get(key) {
                            Some(DebugValue::Number(n)) => *n,
                            _ => f64::NAN,
                        })
                        .collect()
                }
            };
            let sample = CapturedSample {
                t: now,
                cmd_x,
                cmd_y,
                heading: snap.yaw.radians(),
                pos_x: snap.position.x,
                pos_y: snap.position.y,
                vel_x: snap.velocity.x,
                vel_y: snap.velocity.y,
                tags: tag_values,
            };
            let r = self.recordings.get_mut(&player).expect("just iterated");
            r.buffer.push(sample);
            r.last_sample_s = now;
        }
    }
}

/// Write samples to `./.dies/recordings/<label>_<unix_ms>.csv` and return the
/// absolute path. Creates the directory if needed.
pub fn dump_samples_csv(
    label: &str,
    samples: &[CapturedSample],
    tag_names: &[String],
) -> std::io::Result<std::path::PathBuf> {
    let safe_label: String = label
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let dir = std::path::Path::new(".dies").join("recordings");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{}_{}.csv", safe_label, now_ms()));
    std::fs::write(&path, samples_to_csv(samples, tag_names))?;
    Ok(path)
}

// ---------------------------------------------------------------------------
// Native function helpers
// ---------------------------------------------------------------------------

fn expect_object(v: &JsValue, what: &str) -> JsResult<JsObject> {
    v.as_object().cloned().ok_or_else(|| {
        JsNativeError::typ()
            .with_message(format!("{}: expected object", what))
            .into()
    })
}

/// Extract a required number property from a JS options object.
fn get_num(obj_val: &JsValue, key: &str, ctx: &mut Context) -> JsResult<f64> {
    let obj = expect_object(obj_val, "get_num")?;
    let v = obj.get(JsString::from(key), ctx)?;
    v.to_number(ctx)
}

/// Extract an optional number property.
fn get_num_opt(obj_val: &JsValue, key: &str, ctx: &mut Context) -> JsResult<Option<f64>> {
    let obj = expect_object(obj_val, "get_num_opt")?;
    let v = obj.get(JsString::from(key), ctx)?;
    if v.is_undefined() || v.is_null() {
        return Ok(None);
    }
    Ok(Some(v.to_number(ctx)?))
}

fn get_string_opt(obj_val: &JsValue, key: &str, ctx: &mut Context) -> JsResult<Option<String>> {
    let obj = expect_object(obj_val, "get_string_opt")?;
    let v = obj.get(JsString::from(key), ctx)?;
    if v.is_undefined() || v.is_null() {
        return Ok(None);
    }
    Ok(Some(v.to_string(ctx)?.to_std_string_escaped()))
}

fn get_obj_opt(obj_val: &JsValue, key: &str, ctx: &mut Context) -> JsResult<Option<JsValue>> {
    let obj = expect_object(obj_val, "get_obj_opt")?;
    let v = obj.get(JsString::from(key), ctx)?;
    if v.is_undefined() || v.is_null() {
        Ok(None)
    } else {
        Ok(Some(v))
    }
}

/// Extract an optional array-of-strings property.
fn get_string_array_opt(
    obj_val: &JsValue,
    key: &str,
    ctx: &mut Context,
) -> JsResult<Option<Vec<String>>> {
    let obj = expect_object(obj_val, "get_string_array_opt")?;
    let v = obj.get(JsString::from(key), ctx)?;
    if v.is_undefined() || v.is_null() {
        return Ok(None);
    }
    let arr_obj = v
        .as_object()
        .cloned()
        .ok_or_else(|| JsNativeError::typ().with_message(format!("{}: expected array", key)))?;
    let arr = JsArray::from_object(arr_obj)
        .map_err(|_| JsNativeError::typ().with_message(format!("{}: expected Array", key)))?;
    let len = arr.length(ctx)?;
    let mut out = Vec::with_capacity(len as usize);
    for i in 0..len {
        let item = arr.get(i, ctx)?;
        out.push(item.to_string(ctx)?.to_std_string_escaped());
    }
    Ok(Some(out))
}

/// Expand the `p.{tag}` shorthand to `p{id}.{tag}`. All other keys pass
/// through unchanged so callers can record global tags or another player's.
fn resolve_tag(tag: &str, player: PlayerId) -> String {
    if let Some(rest) = tag.strip_prefix("p.") {
        format!("p{}.{}", player.as_u32(), rest)
    } else {
        tag.to_string()
    }
}

// ---------------------------------------------------------------------------
// Global API registration
// ---------------------------------------------------------------------------

/// Register the full scenario API on the context: `team`, `world`, `log`,
/// `sleep`, `waitUntil`, `chirp`, `step`, `prbs`, `ramp`.
pub fn register_api(ctx: &mut Context, state: StateRef) -> JsResult<()> {
    register_globals(ctx, state.clone())?;
    register_world(ctx, state.clone())?;
    register_team(ctx, state.clone())?;
    register_log(ctx, state.clone())?;
    register_excitation_builders(ctx)?;
    Ok(())
}

fn register_globals(ctx: &mut Context, state: StateRef) -> JsResult<()> {
    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let ms = args
                    .first()
                    .cloned()
                    .unwrap_or(JsValue::undefined())
                    .to_number(ctx)?;
                let (promise, resolvers) = JsPromise::new_pending(ctx);
                let mut st = state.borrow_mut();
                let id = st.alloc_waker_id();
                let deadline_s = st.world.t + (ms * 1e-3);
                st.wakers.insert(
                    id,
                    Waker::Sleep {
                        deadline_s,
                        resolve: resolvers.resolve,
                    },
                );
                Ok(promise.into())
            })
        };
        ctx.register_global_callable(js_string!("sleep"), 1, f)?;
    }

    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let pred_val = args.first().cloned().unwrap_or(JsValue::undefined());
                let pred_obj = pred_val.as_object().cloned().ok_or_else(|| {
                    JsNativeError::typ().with_message("waitUntil: predicate must be a function")
                })?;
                let pred = JsFunction::from_object(pred_obj).ok_or_else(|| {
                    JsNativeError::typ().with_message("waitUntil: not a function")
                })?;
                let opts = args.get(1).cloned().unwrap_or(JsValue::undefined());
                let (poll_ms, timeout_ms) = if opts.is_object() {
                    (
                        get_num_opt(&opts, "pollMs", ctx)?.unwrap_or(50.0),
                        get_num_opt(&opts, "timeoutMs", ctx)?,
                    )
                } else {
                    (50.0, None)
                };
                let (promise, resolvers) = JsPromise::new_pending(ctx);
                let mut st = state.borrow_mut();
                let id = st.alloc_waker_id();
                let now = st.world.t;
                st.wakers.insert(
                    id,
                    Waker::WaitUntil {
                        predicate: pred,
                        poll_s: poll_ms * 1e-3,
                        next_poll_s: now,
                        resolve: resolvers.resolve,
                        reject: resolvers.reject,
                        deadline_s: timeout_ms.map(|ms| now + ms * 1e-3),
                    },
                );
                Ok(promise.into())
            })
        };
        ctx.register_global_callable(js_string!("waitUntil"), 2, f)?;
    }

    Ok(())
}

fn register_world(ctx: &mut Context, state: StateRef) -> JsResult<()> {
    let mut init = ObjectInitializer::new(ctx);

    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, _args, _ctx| {
                Ok(JsValue::new(state.borrow().world.t))
            })
        };
        init.function(f, js_string!("t"), 0);
    }
    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, _args, _ctx| {
                Ok(JsValue::new(state.borrow().world.dt))
            })
        };
        init.function(f, js_string!("dt"), 0);
    }

    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, _args, ctx| {
                let b = state.borrow().world.ball;
                match b {
                    Some((p, v)) => Ok(ball_to_js(p, v, ctx)?.into()),
                    None => Ok(JsValue::null()),
                }
            })
        };
        init.function(f, js_string!("ball"), 0);
    }

    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let id_num = args
                    .first()
                    .cloned()
                    .unwrap_or(JsValue::undefined())
                    .to_number(ctx)? as u32;
                let pid = PlayerId::new(id_num);
                let snap = state.borrow().world.players.get(&pid).cloned();
                match snap {
                    Some(p) => Ok(player_to_js(&p, ctx)?.into()),
                    None => Ok(JsValue::null()),
                }
            })
        };
        init.function(f, js_string!("robot"), 1);
    }

    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let opts = args.first().cloned().unwrap_or(JsValue::undefined());
                let opts = expect_object(&opts, "world.addRobot")?;
                let team = parse_team(&opts.clone().into(), ctx)?;
                let id = PlayerId::new(get_num(&opts.clone().into(), "id", ctx)? as u32);
                let x = get_num(&opts.clone().into(), "x", ctx)?;
                let y = get_num(&opts.clone().into(), "y", ctx)?;
                let yaw = get_num_opt(&opts.clone().into(), "yaw", ctx)?.unwrap_or(0.0);
                let mut st = state.borrow_mut();
                if !env_allows_sim_mutation(&st.env) {
                    return Err(JsNativeError::typ()
                        .with_message("world.addRobot: not permitted outside sim")
                        .into());
                }
                st.pending_sim_cmds.push(SimulatorCmd::AddRobot {
                    team_color: team,
                    player_id: id,
                    position: Vector2::new(x, y),
                    yaw: Angle::from_radians(yaw),
                });
                Ok(resolved_undefined(ctx).into())
            })
        };
        init.function(f, js_string!("addRobot"), 1);
    }

    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let opts = args.first().cloned().unwrap_or(JsValue::undefined());
                let team = parse_team(&opts, ctx)?;
                let id = PlayerId::new(get_num(&opts, "id", ctx)? as u32);
                let mut st = state.borrow_mut();
                if !env_allows_sim_mutation(&st.env) {
                    return Err(JsNativeError::typ()
                        .with_message("world.removeRobot: not permitted outside sim")
                        .into());
                }
                st.pending_sim_cmds.push(SimulatorCmd::RemoveRobot {
                    team_color: team,
                    player_id: id,
                });
                Ok(resolved_undefined(ctx).into())
            })
        };
        init.function(f, js_string!("removeRobot"), 1);
    }

    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let opts = args.first().cloned().unwrap_or(JsValue::undefined());
                let x = get_num(&opts, "x", ctx)?;
                let y = get_num(&opts, "y", ctx)?;
                let mut st = state.borrow_mut();
                if !env_allows_sim_mutation(&st.env) {
                    return Err(JsNativeError::typ()
                        .with_message("world.setBallForce: sim only")
                        .into());
                }
                st.pending_sim_cmds.push(SimulatorCmd::ApplyBallForce {
                    force: Vector2::new(x, y),
                });
                Ok(JsValue::undefined())
            })
        };
        init.function(f, js_string!("setBallForce"), 1);
    }

    let world_obj = init.build();
    ctx.register_global_property(
        js_string!("world"),
        world_obj,
        Attribute::WRITABLE | Attribute::CONFIGURABLE,
    )?;
    Ok(())
}

fn register_team(ctx: &mut Context, state: StateRef) -> JsResult<()> {
    let mut init = ObjectInitializer::new(ctx);
    {
        let state = state.clone();
        let robot_fn = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let id_num = args
                    .first()
                    .cloned()
                    .unwrap_or(JsValue::undefined())
                    .to_number(ctx)? as u32;
                let id = PlayerId::new(id_num);
                let handle = build_robot_handle(ctx, state.clone(), id)?;
                Ok(handle.into())
            })
        };
        init.function(robot_fn, js_string!("robot"), 1);
    }

    {
        let state = state.clone();
        let auto_fn = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let opts = args.first().cloned().unwrap_or(JsValue::undefined());
                let id = pick_auto_robot(state.clone(), &opts, ctx)?;
                let handle = build_robot_handle(ctx, state.clone(), id)?;
                Ok(handle.into())
            })
        };
        init.function(auto_fn, js_string!("autoRobot"), 1);
    }
    let team_obj = init.build();
    ctx.register_global_property(
        js_string!("team"),
        team_obj,
        Attribute::WRITABLE | Attribute::CONFIGURABLE,
    )?;
    Ok(())
}

/// Pick (or spawn, in sim) a robot id for `team.autoRobot`.
///
/// **Sim / Either**: uses the next id from `next_auto_id`, queues an
/// `AddRobot` simulator command at the (optionally provided) pose, and
/// returns the id. The counter is bumped unconditionally so consecutive calls
/// never collide even if the queued spawns haven't appeared in the world
/// snapshot yet.
///
/// **Real**: scans the current world snapshot for a player that has both
/// vision (it's in `world.players`) and a usable basestation reply
/// (`primary_status` set and not `NoReply` / `NotInstalled`), and that hasn't
/// already been claimed by an earlier `autoRobot` call. Returns the smallest
/// such id, or throws if none exists.
fn pick_auto_robot(state: StateRef, opts: &JsValue, ctx: &mut Context) -> JsResult<PlayerId> {
    let (x, y, yaw) = if opts.is_object() {
        (
            get_num_opt(opts, "x", ctx)?.unwrap_or(0.0),
            get_num_opt(opts, "y", ctx)?.unwrap_or(0.0),
            get_num_opt(opts, "yaw", ctx)?.unwrap_or(0.0),
        )
    } else {
        (0.0, 0.0, 0.0)
    };
    let mut st = state.borrow_mut();
    if env_allows_sim_mutation(&st.env) {
        let id_num = st.next_auto_id;
        st.next_auto_id = id_num.wrapping_add(1);
        let id = PlayerId::new(id_num);
        let team = st.team_color;
        st.pending_sim_cmds.push(SimulatorCmd::AddRobot {
            team_color: team,
            player_id: id,
            position: Vector2::new(x, y),
            yaw: Angle::from_radians(yaw),
        });
        st.auto_claimed_ids.insert(id);
        Ok(id)
    } else {
        // Real mode: pick the smallest visible+responsive id we haven't
        // already handed out. Sort to make the choice deterministic across
        // runs since HashMap iteration order isn't stable.
        let mut candidates: Vec<PlayerId> = st
            .world
            .players
            .iter()
            .filter(|(id, snap)| {
                !st.auto_claimed_ids.contains(id) && snap.is_basestation_responsive()
            })
            .map(|(id, _)| *id)
            .collect();
        candidates.sort_by_key(|p| p.as_u32());
        let Some(id) = candidates.first().copied() else {
            return Err(JsNativeError::typ()
                .with_message("team.autoRobot: no robot has both vision and basestation status")
                .into());
        };
        st.auto_claimed_ids.insert(id);
        Ok(id)
    }
}

fn register_log(ctx: &mut Context, state: StateRef) -> JsResult<()> {
    let mut init = ObjectInitializer::new(ctx);
    for (name, level) in [
        ("info", TestLogLevel::Info),
        ("warn", TestLogLevel::Warn),
        ("error", TestLogLevel::Error),
    ] {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let msg = args
                    .first()
                    .cloned()
                    .unwrap_or(JsValue::undefined())
                    .to_string(ctx)?
                    .to_std_string_escaped();
                match level {
                    TestLogLevel::Info => tracing::info!(target: "test_driver.script", "{}", msg),
                    TestLogLevel::Warn => tracing::warn!(target: "test_driver.script", "{}", msg),
                    TestLogLevel::Error => tracing::error!(target: "test_driver.script", "{}", msg),
                    _ => {}
                }
                state.borrow().emit_log(level, None, &msg, None);
                Ok(JsValue::undefined())
            })
        };
        init.function(f, JsString::from(name), 1);
    }

    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let tag = args
                    .first()
                    .cloned()
                    .unwrap_or(JsValue::undefined())
                    .to_string(ctx)?
                    .to_std_string_escaped();
                let val = args.get(1).cloned().unwrap_or(JsValue::undefined());
                let json = js_value_to_json(&val, ctx);
                {
                    let mut st = state.borrow_mut();
                    st.record_artifacts.push((tag.clone(), json.clone()));
                    st.emit_log(TestLogLevel::Record, Some(&tag), "record", Some(&json));
                }
                Ok(JsValue::undefined())
            })
        };
        init.function(f, js_string!("record"), 2);
    }

    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let label = args
                    .first()
                    .cloned()
                    .unwrap_or(JsValue::undefined())
                    .to_string(ctx)?
                    .to_std_string_escaped();
                let arr_val = args.get(1).cloned().unwrap_or(JsValue::undefined());
                let (samples, tag_names) = js_array_to_captured(&arr_val, ctx)?;
                let path = dump_samples_csv(&label, &samples, &tag_names).map_err(|e| {
                    JsNativeError::typ().with_message(format!("log.dumpCsv: {}", e))
                })?;
                let path_str = path.to_string_lossy().to_string();
                state.borrow().emit_log(
                    TestLogLevel::Info,
                    Some(&label),
                    &format!("dumped {} samples to {}", samples.len(), path_str),
                    None,
                );
                Ok(JsValue::from(JsString::from(path_str)))
            })
        };
        init.function(f, js_string!("dumpCsv"), 2);
    }

    let obj = init.build();
    ctx.register_global_property(
        js_string!("log"),
        obj,
        Attribute::WRITABLE | Attribute::CONFIGURABLE,
    )?;
    Ok(())
}

fn register_excitation_builders(ctx: &mut Context) -> JsResult<()> {
    for name in ["chirp", "step", "prbs", "ramp", "zero"] {
        let name_owned = name.to_string();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let v = args.first().cloned().unwrap_or(JsValue::undefined());
                let obj = JsObject::with_object_proto(ctx.intrinsics());
                obj.set(
                    js_string!("__kind"),
                    JsValue::from(JsString::from(name_owned.clone())),
                    false,
                    ctx,
                )?;
                if v.is_object() {
                    let src = v.as_object().cloned().unwrap();
                    let keys = src.own_property_keys(ctx)?;
                    for k in keys {
                        let pv = src.get(k.clone(), ctx)?;
                        obj.set(k, pv, false, ctx)?;
                    }
                }
                Ok(obj.into())
            })
        };
        ctx.register_global_callable(JsString::from(name), 1, f)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Robot handle — object returned by `team.robot(id)`
// ---------------------------------------------------------------------------

fn build_robot_handle(ctx: &mut Context, state: StateRef, id: PlayerId) -> JsResult<JsObject> {
    let mut init = ObjectInitializer::new(ctx);

    init.property(
        js_string!("id"),
        JsValue::new(id.as_u32() as f64),
        Attribute::READONLY,
    );

    // --- direct path ---
    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let opts = args.first().cloned().unwrap_or(JsValue::undefined());
                let x = get_num(&opts, "x", ctx)?;
                let y = get_num(&opts, "y", ctx)?;
                let yaw = get_num_opt(&opts, "yaw", ctx)?.unwrap_or(0.0);
                let mut st = state.borrow_mut();
                if !env_allows_sim_mutation(&st.env) {
                    tracing::warn!(
                        "robot.teleport called on non-sim env — no-op (env=either required)"
                    );
                    return Ok(resolved_undefined(ctx).into());
                }
                let team = st.team_color;
                st.pending_sim_cmds.push(SimulatorCmd::TeleportRobot {
                    team_color: team,
                    player_id: id,
                    position: Vector2::new(x, y),
                    yaw: Angle::from_radians(yaw),
                });
                Ok(resolved_undefined(ctx).into())
            })
        };
        init.function(f, js_string!("teleport"), 1);
    }

    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let opts = args.first().cloned().unwrap_or(JsValue::undefined());
                let x = get_num(&opts, "x", ctx)?;
                let y = get_num(&opts, "y", ctx)?;
                let yaw = get_num_opt(&opts, "yaw", ctx)?;
                let opts2 = args.get(1).cloned().unwrap_or(JsValue::undefined());
                let (tol, timeout_ms, vel_thresh) = if opts2.is_object() {
                    (
                        get_num_opt(&opts2, "tolMm", ctx)?.unwrap_or(50.0),
                        get_num_opt(&opts2, "timeoutMs", ctx)?,
                        get_num_opt(&opts2, "velThreshMmPerSec", ctx)?.unwrap_or(40.0),
                    )
                } else {
                    (50.0, None, 40.0)
                };
                let (promise, resolvers) = JsPromise::new_pending(ctx);
                let mut slot = PlayerControlSlot::default();
                slot.position = Some(Vector2::new(x, y));
                if let Some(yv) = yaw {
                    slot.yaw = Some(Angle::from_radians(yv));
                }
                let mut st = state.borrow_mut();
                st.slots.route_direct(id, slot, "moveTo");
                let id_w = st.alloc_waker_id();
                let now = st.world.t;
                let deadline = timeout_ms.map(|ms| now + ms * 1e-3);
                st.wakers.insert(
                    id_w,
                    Waker::MoveTo {
                        player: id,
                        target: Vector2::new(x, y),
                        tol,
                        vel_thresh,
                        resolve: resolvers.resolve,
                        reject: resolvers.reject,
                        deadline_s: deadline,
                    },
                );
                Ok(promise.into())
            })
        };
        init.function(f, js_string!("moveTo"), 2);
    }

    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let opts = args.first().cloned().unwrap_or(JsValue::undefined());
                let x = get_num_opt(&opts, "x", ctx)?.unwrap_or(0.0);
                let y = get_num_opt(&opts, "y", ctx)?.unwrap_or(0.0);
                let yaw = get_num_opt(&opts, "yaw", ctx)?;
                let w = get_num_opt(&opts, "w", ctx)?;
                let mut slot = PlayerControlSlot::default();
                slot.vel_global = Some(Vector2::new(x, y));
                slot.yaw = yaw.map(Angle::from_radians);
                slot.angular_velocity = w;
                state
                    .borrow_mut()
                    .slots
                    .route_direct(id, slot, "setGlobalVelocity");
                Ok(JsValue::undefined())
            })
        };
        init.function(f, js_string!("setGlobalVelocity"), 1);
    }

    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let opts = args.first().cloned().unwrap_or(JsValue::undefined());
                let x = get_num_opt(&opts, "x", ctx)?.unwrap_or(0.0);
                let y = get_num_opt(&opts, "y", ctx)?.unwrap_or(0.0);
                let yaw = get_num_opt(&opts, "yaw", ctx)?;
                let w = get_num_opt(&opts, "w", ctx)?;
                let mut slot = PlayerControlSlot::default();
                slot.vel_local = Some(Vector2::new(x, y));
                slot.yaw = yaw.map(Angle::from_radians);
                slot.angular_velocity = w;
                state
                    .borrow_mut()
                    .slots
                    .route_direct(id, slot, "setLocalVelocity");
                Ok(JsValue::undefined())
            })
        };
        init.function(f, js_string!("setLocalVelocity"), 1);
    }

    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, _args, _ctx| {
                let mut st = state.borrow_mut();
                st.slots.clear_player(id);
                st.slots
                    .route_direct(id, PlayerControlSlot::default(), "stop");
                Ok(JsValue::undefined())
            })
        };
        init.function(f, js_string!("stop"), 0);
    }

    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let opts = args.first().cloned().unwrap_or(JsValue::undefined());
                let (thresh, timeout_ms) = if opts.is_object() {
                    (
                        get_num_opt(&opts, "thresholdMmPerSec", ctx)?.unwrap_or(30.0),
                        get_num_opt(&opts, "timeoutMs", ctx)?,
                    )
                } else {
                    (30.0, None)
                };
                let (promise, resolvers) = JsPromise::new_pending(ctx);
                let mut st = state.borrow_mut();
                let id_w = st.alloc_waker_id();
                let now = st.world.t;
                let deadline = timeout_ms.map(|ms| now + ms * 1e-3);
                st.wakers.insert(
                    id_w,
                    Waker::WaitStopped {
                        player: id,
                        thresh,
                        resolve: resolvers.resolve,
                        reject: resolvers.reject,
                        deadline_s: deadline,
                    },
                );
                Ok(promise.into())
            })
        };
        init.function(f, js_string!("waitStopped"), 1);
    }

    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let speed = args
                    .first()
                    .cloned()
                    .unwrap_or(JsValue::new(0.0))
                    .to_number(ctx)?;
                let mut st = state.borrow_mut();
                let mut existing = st.slots.direct.get(&id).cloned().unwrap_or_default();
                existing.dribble = speed.clamp(0.0, 1.0);
                st.slots.route_direct(id, existing, "setDribble");
                Ok(JsValue::undefined())
            })
        };
        init.function(f, js_string!("setDribble"), 1);
    }

    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let speed = args
                    .first()
                    .cloned()
                    .unwrap_or(JsValue::new(0.0))
                    .to_number(ctx)?;
                let mut st = state.borrow_mut();
                let mut existing = st.slots.direct.get(&id).cloned().unwrap_or_default();
                existing.fan = Some(speed.clamp(0.0, 1.0));
                st.slots.route_direct(id, existing, "setFan");
                Ok(JsValue::undefined())
            })
        };
        init.function(f, js_string!("setFan"), 1);
    }

    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let opts = args.first().cloned().unwrap_or(JsValue::undefined());
                let speed = if opts.is_object() {
                    get_num_opt(&opts, "speed", ctx)?.unwrap_or(3.0)
                } else {
                    3.0
                };
                let mut st = state.borrow_mut();
                let mut existing = st.slots.direct.get(&id).cloned().unwrap_or_default();
                existing.kick_speed = Some(speed);
                st.slots.route_direct(id, existing, "kick");
                Ok(resolved_undefined(ctx).into())
            })
        };
        init.function(f, js_string!("kick"), 1);
    }

    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, _args, _ctx| {
                let mut st = state.borrow_mut();
                let mut existing = st.slots.direct.get(&id).cloned().unwrap_or_default();
                existing.disarm_kicker = true;
                st.slots.route_direct(id, existing, "dischargeKicker");
                Ok(JsValue::undefined())
            })
        };
        init.function(f, js_string!("dischargeKicker"), 0);
    }

    // --- skill path ---
    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let opts = args.first().cloned().unwrap_or(JsValue::undefined());
                let x = get_num(&opts, "x", ctx)?;
                let y = get_num(&opts, "y", ctx)?;
                let heading = get_num_opt(&opts, "heading", ctx)?;
                let cmd = SkillCommand::GoToPos {
                    position: Vector2::new(x, y),
                    heading: heading.map(Angle::from_radians),
                };
                register_skill_and_await(state.clone(), id, cmd, "goToPos", ctx)
            })
        };
        init.function(f, js_string!("goToPos"), 1);
    }

    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let opts = args.first().cloned().unwrap_or(JsValue::undefined());
                let heading = get_num(&opts, "heading", ctx)?;
                let cmd = SkillCommand::PickupBall {
                    target_heading: Angle::from_radians(heading),
                };
                register_skill_and_await(state.clone(), id, cmd, "pickupBall", ctx)
            })
        };
        init.function(f, js_string!("pickupBall"), 1);
    }

    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let opts = args.first().cloned().unwrap_or(JsValue::undefined());
                let target_val = get_obj_opt(&opts, "target", ctx)?
                    .ok_or_else(|| JsNativeError::typ().with_message("dribble: target required"))?;
                let tx = get_num(&target_val, "x", ctx)?;
                let ty = get_num(&target_val, "y", ctx)?;
                let heading = get_num(&opts, "heading", ctx)?;
                let cmd = SkillCommand::Dribble {
                    target_pos: Vector2::new(tx, ty),
                    target_heading: Angle::from_radians(heading),
                };
                register_skill_and_await(state.clone(), id, cmd, "dribble", ctx)
            })
        };
        init.function(f, js_string!("dribble"), 1);
    }

    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let opts = args.first().cloned().unwrap_or(JsValue::undefined());
                let target_val = get_obj_opt(&opts, "target", ctx)?
                    .ok_or_else(|| JsNativeError::typ().with_message("shoot: target required"))?;
                let tx = get_num(&target_val, "x", ctx)?;
                let ty = get_num(&target_val, "y", ctx)?;
                let cmd = SkillCommand::Shoot {
                    target: Vector2::new(tx, ty),
                };
                register_skill_and_await(state.clone(), id, cmd, "shoot", ctx)
            })
        };
        init.function(f, js_string!("shoot"), 1);
    }

    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let opts = args.first().cloned().unwrap_or(JsValue::undefined());
                let from_val = get_obj_opt(&opts, "from", ctx)?
                    .ok_or_else(|| JsNativeError::typ().with_message("receive: from required"))?;
                let fx = get_num(&from_val, "x", ctx)?;
                let fy = get_num(&from_val, "y", ctx)?;
                let target_val = get_obj_opt(&opts, "target", ctx)?
                    .ok_or_else(|| JsNativeError::typ().with_message("receive: target required"))?;
                let tx = get_num(&target_val, "x", ctx)?;
                let ty = get_num(&target_val, "y", ctx)?;

                let capture_limit = get_num_opt(&opts, "captureLimit", ctx)?.unwrap_or(5.0);

                let cmd = SkillCommand::Receive {
                    from_pos: Vector2::new(fx, fy),
                    target_pos: Vector2::new(tx, ty),
                    capture_limit: capture_limit,
                    cushion: false,
                };
                register_skill_and_await(state.clone(), id, cmd, "receive", ctx)
            })
        };
        init.function(f, js_string!("receive"), 1);
    }

    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, _args, _ctx| {
                let mut st = state.borrow_mut();
                st.slots.route_skill(id, SkillCommand::Stop, "skillStop");
                Ok(JsValue::undefined())
            })
        };
        init.function(f, js_string!("skillStop"), 0);
    }

    // --- sysid primitives ---
    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let prof_val = args.first().cloned().unwrap_or(JsValue::undefined());
                let profile = parse_excitation_profile(&prof_val, ctx)?;
                let (promise, resolvers) = JsPromise::new_pending(ctx);
                let mut st = state.borrow_mut();
                let id_w = st.alloc_waker_id();
                let start_s = st.world.t;
                let hold_yaw = st.world.players.get(&id).map(|p| p.yaw);
                st.slots
                    .route_direct(id, PlayerControlSlot::default(), "excite");
                st.wakers.insert(
                    id_w,
                    Waker::Excite {
                        player: id,
                        profile,
                        start_s,
                        hold_yaw,
                        resolve: resolvers.resolve,
                    },
                );
                Ok(promise.into())
            })
        };
        init.function(f, js_string!("excite"), 2);
    }

    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let opts = args.first().cloned().unwrap_or(JsValue::undefined());
                let opts_obj = expect_object(&opts, "robot.captureWhileExciting")?;
                let exc_val = opts_obj.get(js_string!("excitation"), ctx)?;
                let profile = parse_excitation_profile(&exc_val, ctx)?;
                let rate = get_num_opt(&opts, "rateHz", ctx)?.unwrap_or(20.0);
                let duration =
                    get_num_opt(&opts, "durationSec", ctx)?.unwrap_or_else(|| profile.duration());
                let (promise, resolvers) = JsPromise::new_pending(ctx);
                let mut st = state.borrow_mut();
                let id_w = st.alloc_waker_id();
                let start_s = st.world.t;
                let hold_yaw = st.world.players.get(&id).map(|p| p.yaw);
                st.slots
                    .route_direct(id, PlayerControlSlot::default(), "capture");
                st.wakers.insert(
                    id_w,
                    Waker::CaptureActive {
                        player: id,
                        excitation: Some(profile),
                        rate_hz: rate,
                        duration,
                        start_s,
                        last_sample_s: start_s - 1.0 / rate,
                        buffer: CaptureBuffer::default(),
                        hold_yaw,
                        resolve: resolvers.resolve,
                    },
                );
                Ok(promise.into())
            })
        };
        init.function(f, js_string!("captureWhileExciting"), 1);
    }

    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let opts = args.first().cloned().unwrap_or(JsValue::undefined());
                let rate = get_num_opt(&opts, "rateHz", ctx)?.unwrap_or(20.0);
                let duration = get_num(&opts, "durationSec", ctx)?;
                let (promise, resolvers) = JsPromise::new_pending(ctx);
                let mut st = state.borrow_mut();
                let id_w = st.alloc_waker_id();
                let start_s = st.world.t;
                st.wakers.insert(
                    id_w,
                    Waker::CaptureActive {
                        player: id,
                        excitation: None,
                        rate_hz: rate,
                        duration,
                        start_s,
                        last_sample_s: start_s - 1.0 / rate,
                        buffer: CaptureBuffer::default(),
                        hold_yaw: None,
                        resolve: resolvers.resolve,
                    },
                );
                Ok(promise.into())
            })
        };
        init.function(f, js_string!("capture"), 1);
    }

    // --- free-form recording ---
    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, args, ctx| {
                let opts = args.first().cloned().unwrap_or(JsValue::undefined());
                let (rate_hz, tags) = if opts.is_object() {
                    let rate = get_num_opt(&opts, "rateHz", ctx)?.unwrap_or(50.0);
                    let tags = get_string_array_opt(&opts, "tags", ctx)?
                        .unwrap_or_default()
                        .into_iter()
                        .map(|t| resolve_tag(&t, id))
                        .collect();
                    (rate, tags)
                } else {
                    (50.0, Vec::new())
                };
                let mut st = state.borrow_mut();
                let now = st.world.t;
                if st.recordings.contains_key(&id) {
                    return Err(JsNativeError::typ()
                        .with_message(format!(
                            "startRecording: player {} already recording (call stopRecording first)",
                            id.as_u32()
                        ))
                        .into());
                }
                st.recordings.insert(id, Recording::new(rate_hz, now, tags));
                Ok(JsValue::undefined())
            })
        };
        init.function(f, js_string!("startRecording"), 1);
    }

    {
        let state = state.clone();
        let f = unsafe {
            NativeFunction::from_closure(move |_this, _args, ctx| {
                let recording = state.borrow_mut().recordings.remove(&id);
                let Some(rec) = recording else {
                    return Err(JsNativeError::typ()
                        .with_message(format!(
                            "stopRecording: player {} has no active recording",
                            id.as_u32()
                        ))
                        .into());
                };
                let arr = samples_to_js_array(&rec.buffer.samples, &rec.tags, ctx)?;
                Ok(arr.into())
            })
        };
        init.function(f, js_string!("stopRecording"), 0);
    }

    Ok(init.build())
}

fn register_skill_and_await(
    state: StateRef,
    id: PlayerId,
    cmd: SkillCommand,
    role_tag: &str,
    ctx: &mut Context,
) -> JsResult<JsValue> {
    let (promise, resolvers) = JsPromise::new_pending(ctx);
    let mut st = state.borrow_mut();
    st.skill_statuses.insert(id, SkillStatus::Running);
    st.slots.route_skill(id, cmd, role_tag);
    let wid = st.alloc_waker_id();
    st.wakers.insert(
        wid,
        Waker::SkillDone {
            player: id,
            resolve: resolvers.resolve,
            reject: resolvers.reject,
        },
    );
    Ok(promise.into())
}

// ---------------------------------------------------------------------------
// JS helpers
// ---------------------------------------------------------------------------

pub(crate) fn resolved_undefined(ctx: &mut Context) -> JsPromise {
    JsPromise::resolve(JsValue::undefined(), ctx)
}

fn ball_to_js(pos: Vector2, vel: Vector2, ctx: &mut Context) -> JsResult<JsObject> {
    let obj = JsObject::with_object_proto(ctx.intrinsics());
    obj.set(js_string!("position"), vec2_to_js(pos, ctx)?, false, ctx)?;
    obj.set(js_string!("velocity"), vec2_to_js(vel, ctx)?, false, ctx)?;
    Ok(obj)
}

fn player_to_js(p: &PlayerSnap, ctx: &mut Context) -> JsResult<JsObject> {
    let obj = JsObject::with_object_proto(ctx.intrinsics());
    obj.set(
        js_string!("position"),
        vec2_to_js(p.position, ctx)?,
        false,
        ctx,
    )?;
    obj.set(
        js_string!("velocity"),
        vec2_to_js(p.velocity, ctx)?,
        false,
        ctx,
    )?;
    obj.set(js_string!("yaw"), JsValue::new(p.yaw.radians()), false, ctx)?;
    obj.set(
        js_string!("angularSpeed"),
        JsValue::new(p.angular_speed),
        false,
        ctx,
    )?;
    Ok(obj)
}

pub(crate) fn vec2_to_js(v: Vector2, ctx: &mut Context) -> JsResult<JsValue> {
    let obj = JsObject::with_object_proto(ctx.intrinsics());
    obj.set(js_string!("x"), JsValue::new(v.x), false, ctx)?;
    obj.set(js_string!("y"), JsValue::new(v.y), false, ctx)?;
    Ok(obj.into())
}

fn parse_team(opts: &JsValue, ctx: &mut Context) -> JsResult<TeamColor> {
    let s = get_string_opt(opts, "team", ctx)?.unwrap_or_else(|| "blue".into());
    match s.to_lowercase().as_str() {
        "blue" => Ok(TeamColor::Blue),
        "yellow" => Ok(TeamColor::Yellow),
        other => Err(JsNativeError::typ()
            .with_message(format!("unknown team color: {}", other))
            .into()),
    }
}

fn env_allows_sim_mutation(env: &TestEnv) -> bool {
    matches!(env, TestEnv::Sim | TestEnv::Either)
}

pub fn parse_excitation_profile(v: &JsValue, ctx: &mut Context) -> JsResult<ExcitationProfile> {
    if !v.is_object() {
        return Err(JsNativeError::typ()
            .with_message("excitation profile must be an object")
            .into());
    }
    let kind = get_string_opt(v, "__kind", ctx)?
        .ok_or_else(|| JsNativeError::typ().with_message("profile missing __kind"))?;
    match kind.as_str() {
        "chirp" => Ok(ExcitationProfile::Chirp {
            axis: excitation_axis(v, ctx)?,
            f0: get_num(v, "f0", ctx)?,
            f1: get_num(v, "f1", ctx)?,
            amp: get_num(v, "amp", ctx)?,
            duration: get_num(v, "duration", ctx)?,
        }),
        "step" => {
            let duration = get_num_opt(v, "duration", ctx)?
                .or(get_num_opt(v, "holdSec", ctx)?.map(|h| h * 2.0))
                .unwrap_or(1.0);
            Ok(ExcitationProfile::Step {
                axis: excitation_axis(v, ctx)?,
                magnitude: get_num(v, "magnitude", ctx)?,
                hold_sec: get_num_opt(v, "holdSec", ctx)?.unwrap_or(duration / 2.0),
                duration,
            })
        }
        "prbs" => Ok(ExcitationProfile::Prbs {
            axis: excitation_axis(v, ctx)?,
            amp: get_num(v, "amp", ctx)?,
            bandwidth_hz: get_num_opt(v, "bandwidthHz", ctx)?.unwrap_or(2.0),
            duration: get_num(v, "duration", ctx)?,
            seed: get_num_opt(v, "seed", ctx)?.unwrap_or(0xACE1u64 as f64) as u64,
        }),
        "ramp" => Ok(ExcitationProfile::Ramp {
            axis: excitation_axis(v, ctx)?,
            start: get_num(v, "start", ctx)?,
            end: get_num(v, "end", ctx)?,
            duration: get_num(v, "duration", ctx)?,
        }),
        "zero" => Ok(ExcitationProfile::Zero {
            duration: get_num_opt(v, "duration", ctx)?.unwrap_or(1.0),
        }),
        other => Err(JsNativeError::typ()
            .with_message(format!("unknown excitation profile kind: {}", other))
            .into()),
    }
}

fn excitation_axis(v: &JsValue, ctx: &mut Context) -> JsResult<crate::primitives::ExcitationAxis> {
    let s = get_string_opt(v, "axis", ctx)?.unwrap_or_else(|| "forward".into());
    crate::primitives::ExcitationAxis::parse(&s).ok_or_else(|| {
        JsNativeError::typ()
            .with_message(format!("unknown axis: {}", s))
            .into()
    })
}

pub fn samples_to_js_array(
    samples: &[CapturedSample],
    tag_names: &[String],
    ctx: &mut Context,
) -> JsResult<JsArray> {
    let arr = JsArray::new(ctx);
    for s in samples {
        let obj = JsObject::with_object_proto(ctx.intrinsics());
        obj.set(js_string!("t"), JsValue::new(s.t), false, ctx)?;
        let cmd = JsObject::with_object_proto(ctx.intrinsics());
        cmd.set(js_string!("x"), JsValue::new(s.cmd_x), false, ctx)?;
        cmd.set(js_string!("y"), JsValue::new(s.cmd_y), false, ctx)?;
        obj.set(js_string!("cmd"), cmd, false, ctx)?;
        obj.set(js_string!("heading"), JsValue::new(s.heading), false, ctx)?;
        let state = JsObject::with_object_proto(ctx.intrinsics());
        let pos = JsObject::with_object_proto(ctx.intrinsics());
        pos.set(js_string!("x"), JsValue::new(s.pos_x), false, ctx)?;
        pos.set(js_string!("y"), JsValue::new(s.pos_y), false, ctx)?;
        state.set(js_string!("pos"), pos, false, ctx)?;
        let vel = JsObject::with_object_proto(ctx.intrinsics());
        vel.set(js_string!("x"), JsValue::new(s.vel_x), false, ctx)?;
        vel.set(js_string!("y"), JsValue::new(s.vel_y), false, ctx)?;
        state.set(js_string!("vel"), vel, false, ctx)?;
        obj.set(js_string!("state"), state, false, ctx)?;
        if !tag_names.is_empty() {
            let tags_obj = JsObject::with_object_proto(ctx.intrinsics());
            for (name, &v) in tag_names.iter().zip(&s.tags) {
                tags_obj.set(JsString::from(name.as_str()), JsValue::new(v), false, ctx)?;
            }
            obj.set(js_string!("tags"), tags_obj, false, ctx)?;
        }
        arr.push(obj, ctx)?;
    }
    Ok(arr)
}

/// Convert a JS samples array back into Rust. Tag column order is taken from
/// the first sample's `tags` object key order — this matches the order in which
/// tags were declared at `startRecording`.
fn js_array_to_captured(
    arr: &JsValue,
    ctx: &mut Context,
) -> JsResult<(Vec<CapturedSample>, Vec<String>)> {
    let obj = arr
        .as_object()
        .cloned()
        .ok_or_else(|| JsNativeError::typ().with_message("dumpCsv: expected samples array"))?;
    let arr = JsArray::from_object(obj)
        .map_err(|_| JsNativeError::typ().with_message("dumpCsv: expected Array"))?;
    let len = arr.length(ctx)?;
    let mut tag_names: Vec<String> = Vec::new();
    if len > 0 {
        let first = arr.get(0, ctx)?;
        if let Some(tags_val) = get_obj_opt(&first, "tags", ctx)? {
            if let Some(tags_obj) = tags_val.as_object().cloned() {
                let keys = tags_obj.own_property_keys(ctx)?;
                for k in keys {
                    if let PropertyKey::String(s) = &k {
                        tag_names.push(s.to_std_string_escaped());
                    }
                }
            }
        }
    }
    let mut out = Vec::with_capacity(len as usize);
    for i in 0..len {
        let item = arr.get(i, ctx)?;
        let cmd = get_obj_opt(&item, "cmd", ctx)?
            .ok_or_else(|| JsNativeError::typ().with_message("sample missing cmd"))?;
        let state = get_obj_opt(&item, "state", ctx)?
            .ok_or_else(|| JsNativeError::typ().with_message("sample missing state"))?;
        let pos = get_obj_opt(&state, "pos", ctx)?
            .ok_or_else(|| JsNativeError::typ().with_message("sample.state missing pos"))?;
        let vel = get_obj_opt(&state, "vel", ctx)?
            .ok_or_else(|| JsNativeError::typ().with_message("sample.state missing vel"))?;
        let tag_values: Vec<f64> = if tag_names.is_empty() {
            Vec::new()
        } else {
            let tags_val = get_obj_opt(&item, "tags", ctx)?;
            let tags_obj = tags_val.as_ref().and_then(|v| v.as_object().cloned());
            tag_names
                .iter()
                .map(|name| -> JsResult<f64> {
                    let Some(ref t) = tags_obj else {
                        return Ok(f64::NAN);
                    };
                    let v = t.get(JsString::from(name.as_str()), ctx)?;
                    if v.is_undefined() || v.is_null() {
                        Ok(f64::NAN)
                    } else {
                        v.to_number(ctx)
                    }
                })
                .collect::<JsResult<Vec<f64>>>()?
        };
        out.push(CapturedSample {
            t: get_num(&item, "t", ctx)?,
            cmd_x: get_num(&cmd, "x", ctx)?,
            cmd_y: get_num(&cmd, "y", ctx)?,
            heading: get_num(&item, "heading", ctx)?,
            pos_x: get_num(&pos, "x", ctx)?,
            pos_y: get_num(&pos, "y", ctx)?,
            vel_x: get_num(&vel, "x", ctx)?,
            vel_y: get_num(&vel, "y", ctx)?,
            tags: tag_values,
        });
    }
    Ok((out, tag_names))
}

fn js_value_to_json(v: &JsValue, ctx: &mut Context) -> String {
    match v.get_type() {
        Type::Null | Type::Undefined => "null".into(),
        Type::Boolean => v.as_boolean().map(|b| b.to_string()).unwrap_or_default(),
        Type::Number => {
            let n = v.as_number().unwrap_or(0.0);
            if n.is_finite() {
                n.to_string()
            } else {
                "null".into()
            }
        }
        Type::String => v
            .as_string()
            .cloned()
            .map(|js| serde_json::to_string(&js.to_std_string_escaped()).unwrap_or_default())
            .unwrap_or_else(|| "\"\"".into()),
        Type::Object => {
            let Some(obj) = v.as_object().cloned() else {
                return "null".into();
            };
            if let Ok(arr) = JsArray::from_object(obj.clone()) {
                let Ok(len) = arr.length(ctx) else {
                    return "[]".into();
                };
                let mut parts = Vec::new();
                for i in 0..len {
                    let Ok(el) = arr.get(i, ctx) else {
                        parts.push("null".into());
                        continue;
                    };
                    parts.push(js_value_to_json(&el, ctx));
                }
                format!("[{}]", parts.join(","))
            } else {
                let keys = match obj.own_property_keys(ctx) {
                    Ok(k) => k,
                    Err(_) => return "{}".into(),
                };
                let mut parts = Vec::new();
                for k in keys {
                    let key_s = match &k {
                        PropertyKey::String(s) => s.to_std_string_escaped(),
                        PropertyKey::Index(i) => i.get().to_string(),
                        PropertyKey::Symbol(_) => continue,
                    };
                    let Ok(val) = obj.get(k, ctx) else { continue };
                    parts.push(format!(
                        "{}:{}",
                        serde_json::to_string(&key_s).unwrap_or_default(),
                        js_value_to_json(&val, ctx)
                    ));
                }
                format!("{{{}}}", parts.join(","))
            }
        }
        _ => "null".into(),
    }
}

#[allow(dead_code)]
pub fn anyhow_to_js(err: anyhow::Error) -> JsError {
    JsNativeError::typ().with_message(err.to_string()).into()
}

#[allow(dead_code)]
pub(crate) fn build_excitation_sample(profile: &ExcitationProfile, t_rel: f64) -> ExcitationSample {
    profile.sample(t_rel)
}
