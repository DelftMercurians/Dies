//! # scenarios
//!
//! A tiny, pure-Rust harness for **invoking and testing skills through the exact
//! interface strategies use**. There is no dedicated scenario API: a scenario is
//! just a [`Strategy`] that scripts one or more robots through a sequence of
//! skills, launched like any other strategy (pick it in the strategy picker, or
//! `--strategy <name>`).
//!
//! ## Authoring
//!
//! Each scenario is a binary under `src/bin/<name>.rs` that hands a [`Scenario`]
//! to [`run_scenario`]. A `Scenario` is an ordered list of [`Step`]s; the harness
//! runs the current step every frame and advances when it reports success.
//!
//! ```ignore
//! use scenarios::prelude::*;
//!
//! fn main() {
//!     run_scenario(|| {
//!         let p = PlayerId::new(0);
//!         Scenario::looping(move || {
//!             vec![
//!                 Step::skill("acquire", p, |h| { h.handle_ball(BallAction::Hold { heading: Angle::from_radians(0.0) }, None); })
//!                     .timeout(15.0),
//!                 Step::skill("shoot", p, |h| { h.reflex_shoot(OPP_GOAL); })
//!                     .timeout(10.0),
//!                 Step::wait(0.5),
//!             ]
//!         })
//!     });
//! }
//! ```
//!
//! ## Setup
//!
//! Scenarios do **not** place robots or the ball — they assume a seeded field.
//! Seed it with the snapshot loader in the Web UI, or `--snapshot <name>` on the
//! CLI, before the scenario takes over.

use dies_strategy_api::prelude::*;
use dies_strategy_protocol::PassResult;

pub use dies_strategy_runner::run_strategy;

/// A convenient stand-in for the opponent goal centre in team-relative
/// coordinates (+x toward the opponent). Scenarios don't have live field
/// geometry at build time, so skill targets are expressed as plain points; this
/// is "roughly the far goal" on a standard field.
pub const OPP_GOAL: Vector2 = Vector2::new(4500.0, 0.0);

/// Number of frames a step tolerates a *stale* terminal status before honoring
/// it. A discrete skill the same robot just finished still reports `Succeeded`
/// for a frame or two after a new command is issued; without this guard the next
/// step would complete instantly off the previous step's result.
const ARM_GRACE_FRAMES: u32 = 5;

/// The outcome a [`Step`] reports each frame.
pub enum StepOutcome {
    /// Still working — keep running this step next frame.
    Running,
    /// Finished successfully — advance to the next step.
    Succeeded,
    /// Finished unsuccessfully — log and advance to the next step.
    Failed,
}

/// One step of a [`Scenario`]: a per-frame closure plus optional timeout.
///
/// Construct steps with [`Step::skill`], [`Step::pass`], [`Step::wait`], or
/// [`Step::custom`].
pub struct Step {
    name: String,
    /// Category for the UI plan panel (drives its per-step color), e.g.
    /// `"Skill"`, `"Pass"`, `"Wait"`. Mirrors concerto's `PlanStep::kind`.
    kind: String,
    /// The robots this step drives. Used to reset their skills when the step
    /// ends (so a re-run starts from a fresh skill instance, not stale internal
    /// state) and to surface the first as the plan's active robot.
    players: Vec<PlayerId>,
    run: Box<dyn FnMut(&mut TeamContext) -> StepOutcome + Send>,
    timeout: Option<f64>,
    elapsed: f64,
}

impl Step {
    fn new(
        name: impl Into<String>,
        kind: impl Into<String>,
        run: impl FnMut(&mut TeamContext) -> StepOutcome + Send + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            kind: kind.into(),
            players: Vec::new(),
            run: Box::new(run),
            timeout: None,
            elapsed: 0.0,
        }
    }

    /// Fail (and advance) the step if it hasn't completed within `seconds` of
    /// simulated/world time. Skills can stall, so most steps want a timeout.
    pub fn timeout(mut self, seconds: f64) -> Self {
        self.timeout = Some(seconds);
        self
    }

    /// Drive a single skill on one player and complete when the executor reports
    /// the skill `Succeeded` (continuous skills like `go_to` complete on arrival;
    /// discrete skills like `handle_ball`/`reflex_shoot` complete when done).
    ///
    /// `issue` is called every frame with the player's [`PlayerHandle`] — issue
    /// the skill there exactly as a strategy would. The closure should command
    /// the *same* skill each frame so continuous skills update live.
    pub fn skill(
        name: impl Into<String>,
        id: PlayerId,
        mut issue: impl FnMut(&mut PlayerHandle) + Send + 'static,
    ) -> Self {
        let mut armed = false;
        let mut frames: u32 = 0;
        let mut step = Step::new(name, "Skill", move |ctx| {
            frames += 1;
            let status = match ctx.player(id) {
                Some(player) => {
                    issue(player);
                    player.skill_status()
                }
                // Robot not in the world yet (vision warming up) — keep waiting.
                None => return StepOutcome::Running,
            };
            match status {
                SkillStatus::Running => {
                    armed = true;
                    StepOutcome::Running
                }
                SkillStatus::Succeeded if armed || frames > ARM_GRACE_FRAMES => {
                    StepOutcome::Succeeded
                }
                SkillStatus::Failed if armed || frames > ARM_GRACE_FRAMES => StepOutcome::Failed,
                _ => StepOutcome::Running,
            }
        });
        step.players = vec![id];
        step
    }

    /// Run the joint pass coordinator from `passer` to `receiver` and complete on
    /// the typed [`PassResult`]: `Success` → succeeded, `Failure` → failed (the
    /// reason is logged).
    pub fn pass(passer: PlayerId, receiver: PlayerId, target_hint: Option<Vector2>) -> Self {
        let mut armed = false;
        let mut frames: u32 = 0;
        let mut step = Step::new(format!("pass {passer}->{receiver}"), "Pass", move |ctx| {
            frames += 1;
            // Issue the pass into both slots (committed on the builder's drop).
            match target_hint {
                Some(hint) => {
                    ctx.pass(passer, receiver).target_hint(hint);
                }
                None => {
                    ctx.pass(passer, receiver);
                }
            }
            if ctx
                .player_ref(passer)
                .map(|p| p.skill_status() == SkillStatus::Running)
                .unwrap_or(false)
            {
                armed = true;
            }
            if armed || frames > ARM_GRACE_FRAMES {
                if let Some(result) = ctx.pass_result(passer) {
                    return match result {
                        PassResult::Success { .. } => StepOutcome::Succeeded,
                        PassResult::Failure { reason, .. } => {
                            tracing::warn!("pass failed: {reason:?}");
                            StepOutcome::Failed
                        }
                    };
                }
            }
            StepOutcome::Running
        });
        step.players = vec![passer, receiver];
        step
    }

    /// Idle for `seconds` of world time, then succeed. Useful for letting the
    /// field settle between skills or pacing a loop.
    pub fn wait(seconds: f64) -> Self {
        let mut elapsed = 0.0;
        Step::new(format!("wait {seconds}s"), "Wait", move |ctx| {
            elapsed += ctx.world().dt();
            if elapsed >= seconds {
                StepOutcome::Succeeded
            } else {
                StepOutcome::Running
            }
        })
    }

    /// Escape hatch: a fully custom per-frame step. Use when a step needs world
    /// state or coordinates more than one robot.
    pub fn custom(
        name: impl Into<String>,
        run: impl FnMut(&mut TeamContext) -> StepOutcome + Send + 'static,
    ) -> Self {
        Step::new(name, "Custom", run)
    }
}

/// An ordered list of [`Step`]s run through the strategy IPC.
///
/// Built from a *factory* closure so a looping scenario can rebuild fresh steps
/// (resetting their internal latch state) on each pass.
pub struct Scenario {
    factory: Box<dyn Fn() -> Vec<Step> + Send>,
    steps: Vec<Step>,
    idx: usize,
    looping: bool,
    finished: bool,
}

impl Scenario {
    fn build(factory: impl Fn() -> Vec<Step> + Send + 'static, looping: bool) -> Self {
        let steps = factory();
        Self {
            factory: Box::new(factory),
            steps,
            idx: 0,
            looping,
            finished: false,
        }
    }

    /// Run the steps once, then stop all robots and idle.
    pub fn once(factory: impl Fn() -> Vec<Step> + Send + 'static) -> Self {
        Self::build(factory, false)
    }

    /// Run the steps forever, rebuilding them on each pass. Great for eyeballing a
    /// skill repeatedly.
    pub fn looping(factory: impl Fn() -> Vec<Step> + Send + 'static) -> Self {
        Self::build(factory, true)
    }
}

impl Strategy for Scenario {
    fn update(&mut self, ctx: &mut TeamContext) {
        if self.finished {
            return;
        }

        if self.idx >= self.steps.len() {
            if self.looping {
                self.steps = (self.factory)();
                self.idx = 0;
                if self.steps.is_empty() {
                    self.finished = true;
                    return;
                }
            } else {
                tracing::info!("scenario complete");
                self.finished = true;
                for player in ctx.players() {
                    player.stop();
                }
                return;
            }
        }

        let idx = self.idx;
        let total = self.steps.len();
        let dt = ctx.world().dt();

        // Mirror the step list into the UI's plan panel via the same
        // `DebugValue::Plan` primitive concerto uses. Re-emitted every frame
        // (debug entries are TTL-evicted) so the inspector tracks progress: the
        // current step is the active one. The key is prefixed to
        // `team_{Color}.strategy.plan` by the strategy host — exactly what the
        // PlanSection panel reads.
        let plan_steps: Vec<debug::PlanStep> = self
            .steps
            .iter()
            .enumerate()
            .map(|(i, step)| debug::PlanStep {
                kind: step.kind.clone(),
                label: step.name.clone(),
                detail: None,
                active: i == idx,
            })
            .collect();
        let active_robot = self.steps[idx].players.first().map(|p| p.as_u32());
        debug::plan("plan", active_robot, plan_steps);

        let (name, players, timed_out, outcome) = {
            let step = &mut self.steps[idx];
            step.elapsed += dt;
            let timed_out = step.timeout.is_some_and(|t| step.elapsed >= t);
            let outcome = if timed_out {
                StepOutcome::Failed
            } else {
                (step.run)(ctx)
            };
            (step.name.clone(), step.players.clone(), timed_out, outcome)
        };

        let advanced = match outcome {
            StepOutcome::Running => false,
            StepOutcome::Succeeded => {
                tracing::info!("✓ step {}/{} '{}' succeeded", idx + 1, total, name);
                self.idx += 1;
                true
            }
            StepOutcome::Failed => {
                if timed_out {
                    tracing::warn!("✗ step {}/{} '{}' timed out", idx + 1, total, name);
                } else {
                    tracing::warn!("✗ step {}/{} '{}' failed", idx + 1, total, name);
                }
                self.idx += 1;
                true
            }
        };

        // Reset the skills this step drove so the next step — or, in a looping
        // scenario, the next pass — rebuilds each skill from a fresh instance
        // instead of inheriting the finished skill's terminal status or internal
        // state. `stop` drops the executor's `current_skill` for the player; the
        // next frame's command recreates it. (Custom steps that drive untracked
        // robots are responsible for their own cleanup.)
        if advanced {
            for pid in players {
                if let Some(player) = ctx.player(pid) {
                    player.stop();
                }
            }
        }
    }
}

/// Run a scenario binary: connect over the strategy IPC and drive the built
/// [`Scenario`] every frame. Call this from `main()`.
pub fn run_scenario(factory: impl FnOnce() -> Scenario + Send + 'static) {
    run_strategy(factory);
}

/// Prelude for scenario binaries.
pub mod prelude {
    pub use crate::{run_scenario, Scenario, Step, StepOutcome, OPP_GOAL};
    pub use dies_strategy_api::prelude::*;
}
