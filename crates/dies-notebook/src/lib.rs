//! Embedded JavaScript engine for the interactive notebooks (motion testing).
//!
//! The engine knows nothing about robots. The boundary to the host is data only:
//! the host evaluates cell sources, and once per executor frame hands in a JSON
//! snapshot and takes back a JSON output per notebook (commands, debug values,
//! events, cell outputs). All user-facing API — awaiting frames, holding
//! commands, recording trials, plotting — lives in the JS prelude
//! (`prelude.js`), never here.
//!
//! Each notebook gets its own JS context (separate globals, own prelude state)
//! inside one shared runtime, so promise jobs of all notebooks drain together
//! once per frame.

use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use anyhow::{anyhow, Result};
use rquickjs::{CatchResultExt, CaughtError, Context, Ctx, Function, Object, Runtime};

const PRELUDE: &str = include_str!("prelude.js");

/// Hand-written type declarations of the prelude API, for editor autocompletion.
pub const PRELUDE_TYPES: &str = include_str!("prelude.d.ts");

/// Global names the prelude defines; cells may not redefine them.
pub const RESERVED_NAMES: &[&str] = &[
    "frame",
    "now",
    "robot",
    "emit",
    "show",
    "wait",
    "until",
    "trial",
    "Plot",
    "v2",
    "deg",
    "toDeg",
    "clamp",
    "wrapAngle",
    "stats",
    "__nb",
];

/// CPU budget for a single host → JS call (eval, frame, abort). Exceeding it
/// interrupts the script; the host then aborts everything. This is a wall-clock
/// guard against runaway scripts (e.g. a loop that never awaits), not in-match
/// timing. It is generous because post-run analysis in a cell legitimately takes
/// a while; the host reports the actual per-frame cost as a debug value.
const CALL_BUDGET: Duration = Duration::from_millis(500);

/// Upper bound on queued promise jobs drained per call, as a guard against
/// promise loops that never yield to the frame boundary.
const MAX_JOBS_PER_CALL: usize = 1_000_000;

pub struct NotebookEngine {
    runtime: Runtime,
    /// One context per notebook id. BTreeMap: frames visit notebooks in a fixed
    /// order, so "last notebook wins" on a shared robot is deterministic.
    contexts: BTreeMap<String, Context>,
    /// Latest snapshot, used to prime new contexts.
    last_snapshot: Option<String>,
    /// Deadline for the current call, in nanoseconds since `epoch` (0 = none).
    deadline_ns: Arc<AtomicU64>,
    epoch: Instant,
}

impl NotebookEngine {
    pub fn new() -> Result<Self> {
        let runtime = Runtime::new()?;
        runtime.set_memory_limit(512 * 1024 * 1024);
        runtime.set_max_stack_size(1024 * 1024);

        let epoch = Instant::now();
        let deadline_ns = Arc::new(AtomicU64::new(0));
        {
            let deadline_ns = deadline_ns.clone();
            runtime.set_interrupt_handler(Some(Box::new(move || {
                let d = deadline_ns.load(Ordering::Relaxed);
                d != 0 && epoch.elapsed().as_nanos() as u64 > d
            })));
        }

        Ok(Self {
            runtime,
            contexts: BTreeMap::new(),
            last_snapshot: None,
            deadline_ns,
            epoch,
        })
    }

    pub fn has_notebook(&self, notebook: &str) -> bool {
        self.contexts.contains_key(notebook)
    }

    /// Hand the engine a snapshot without resuming anything, so cells evaluated
    /// before the next frame see the world. Also primes notebooks created later.
    pub fn prime(&mut self, snapshot_json: &str) -> Result<()> {
        self.last_snapshot = Some(snapshot_json.to_string());
        let ids: Vec<String> = self.contexts.keys().cloned().collect();
        for id in ids {
            self.call(&id, "frameBegin", Some(snapshot_json))?;
        }
        self.drain_jobs()
    }

    /// Evaluate a (pre-transformed) cell source in a notebook's scope, creating
    /// the scope on first use. `default_robot` is the robot `robot()` resolves
    /// to (JSON `{team, id}` or `null`). The source is expected to start its
    /// body via `__nb.run(...)`; errors inside the body are reported by the
    /// prelude as cell outputs.
    pub fn eval(&mut self, notebook: &str, source: &str, default_robot: &str) -> Result<()> {
        self.ensure_context(notebook)?;
        self.call(notebook, "setDefaultRobot", Some(default_robot))?;
        let ctx = self.contexts.get(notebook).expect("context just ensured");
        self.with_budget(ctx, |ctx| {
            ctx.eval::<(), _>(source)
                .catch(&ctx)
                .map_err(|e| anyhow!(fmt_caught(e)))
        })?;
        self.drain_jobs()
    }

    /// Run one executor frame in every notebook: resolve frame waiters with
    /// `snapshot_json`, let every resumed task run to its next await, and
    /// collect each notebook's output JSON (in notebook-id order).
    pub fn frame(&mut self, snapshot_json: &str) -> Result<Vec<(String, String)>> {
        self.last_snapshot = Some(snapshot_json.to_string());
        let ids: Vec<String> = self.contexts.keys().cloned().collect();
        for id in &ids {
            self.call(id, "frameBegin", Some(snapshot_json))?;
        }
        self.drain_jobs()?;
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            let o = self.call(&id, "frameEnd", None)?;
            out.push((id, o));
        }
        Ok(out)
    }

    /// Abort every running and queued cell in every notebook and release all
    /// robots (the releases go out with the next frame). Returns the outputs
    /// produced by the abort, per notebook.
    pub fn abort_all(&mut self, reason: &str) -> Result<Vec<(String, String)>> {
        let ids: Vec<String> = self.contexts.keys().cloned().collect();
        for id in &ids {
            self.call(id, "abort", Some(reason))?;
        }
        self.drain_jobs()?;
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            let o = self.call(&id, "flush", None)?;
            out.push((id, o));
        }
        Ok(out)
    }

    /// Abort a notebook and drop its scope. Its held robots vanish from the next
    /// frame's commands (the host releases them).
    pub fn reset(&mut self, notebook: &str) -> Result<Option<String>> {
        if !self.contexts.contains_key(notebook) {
            return Ok(None);
        }
        self.call(notebook, "abort", Some("notebook restarted"))?;
        self.drain_jobs()?;
        let out = self.call(notebook, "flush", None)?;
        self.contexts.remove(notebook);
        Ok(Some(out))
    }

    /// Tell a notebook another notebook took over these robots (JSON list of
    /// `{team, id}`); it forgets its hold so it can't spring back later.
    pub fn drop_held(&mut self, notebook: &str, robots_json: &str) -> Result<()> {
        self.call(notebook, "dropHeld", Some(robots_json))
            .map(|_| ())
    }

    /// Outputs produced between frames (e.g. by `eval`), without advancing time.
    pub fn flush(&mut self, notebook: &str) -> Result<String> {
        self.call(notebook, "flush", None)
    }

    fn ensure_context(&mut self, notebook: &str) -> Result<()> {
        if self.contexts.contains_key(notebook) {
            return Ok(());
        }
        let ctx = Context::full(&self.runtime)?;
        self.with_budget(&ctx, |ctx| {
            ctx.eval::<(), _>(PRELUDE)
                .catch(&ctx)
                .map_err(|e| anyhow!("prelude: {}", fmt_caught(e)))
        })?;
        self.contexts.insert(notebook.to_string(), ctx);
        if let Some(snap) = self.last_snapshot.clone() {
            self.call(notebook, "frameBegin", Some(&snap))?;
            self.drain_jobs()?;
        }
        Ok(())
    }

    fn call(&self, notebook: &str, name: &str, arg: Option<&str>) -> Result<String> {
        let ctx = self
            .contexts
            .get(notebook)
            .ok_or_else(|| anyhow!("no notebook {notebook}"))?;
        self.with_budget(ctx, |ctx| {
            let res = (|| -> rquickjs::Result<Option<String>> {
                let nb: Object = ctx.globals().get("__nb")?;
                let f: Function = nb.get(name)?;
                match arg {
                    Some(a) => f.call((a,)),
                    None => f.call(()),
                }
            })();
            res.catch(&ctx)
                .map(|s| s.unwrap_or_default())
                .map_err(|e| anyhow!("{name}: {}", fmt_caught(e)))
        })
    }

    fn drain_jobs(&mut self) -> Result<()> {
        self.set_deadline();
        let mut n = 0;
        let res = loop {
            match self.runtime.execute_pending_job() {
                Ok(true) => {
                    n += 1;
                    if n > MAX_JOBS_PER_CALL {
                        break Err(anyhow!("too many queued jobs in one frame"));
                    }
                }
                Ok(false) => break Ok(()),
                Err(e) => {
                    // User errors are caught by the prelude, so a throwing job is
                    // an interrupt (budget exceeded) or a prelude bug.
                    let msg = e.0.with(|ctx| {
                        fmt_caught(CaughtError::from_error(&ctx, rquickjs::Error::Exception))
                    });
                    break Err(anyhow!("{msg}"));
                }
            }
        };
        self.clear_deadline();
        res
    }

    fn with_budget<R>(&self, ctx: &Context, f: impl FnOnce(Ctx<'_>) -> Result<R>) -> Result<R> {
        self.set_deadline();
        let res = ctx.with(f);
        self.clear_deadline();
        res
    }

    fn set_deadline(&self) {
        let d = self.epoch.elapsed() + CALL_BUDGET;
        self.deadline_ns
            .store(d.as_nanos() as u64, Ordering::Relaxed);
    }

    fn clear_deadline(&self) {
        self.deadline_ns.store(0, Ordering::Relaxed);
    }
}

fn fmt_caught(e: CaughtError<'_>) -> String {
    match e {
        CaughtError::Exception(ex) => {
            let msg = ex.message().unwrap_or_default();
            match ex.stack() {
                Some(stack) if !stack.trim().is_empty() => format!("{msg}\n{stack}"),
                _ => msg,
            }
        }
        CaughtError::Value(v) => format!("{v:?}"),
        CaughtError::Error(e) => e.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(frame: u64) -> String {
        format!(
            r#"{{"frame":{frame},"t":{t},"dt":0.016,"world":{{"blue_team":[{{"id":4,"position":[{x},0],"velocity":[0,0],"yaw":0}}],"yellow_team":[]}},"robots":[{{"team":"Blue","id":4,"skill":null}}],"teams":["Blue"]}}"#,
            t = frame as f64 * 0.016,
            x = frame * 10,
        )
    }

    fn engine() -> NotebookEngine {
        let mut e = NotebookEngine::new().unwrap();
        e.prime(&snapshot(0)).unwrap();
        e
    }

    fn all(outs: &[(String, String)]) -> String {
        outs.iter()
            .map(|(_, o)| o.as_str())
            .collect::<Vec<_>>()
            .join("")
    }

    #[test]
    fn control_loop_runs_once_per_frame() {
        let mut e = engine();
        e.eval(
            "nb",
            r#"__nb.run("c1", async (show) => {
                const R = robot(4);
                for (let i = 0; i < 3; i++) {
                    const s = await frame();
                    R.set({ velocity: { Global: [s.frame, 0] } });
                    R.log({ i });
                }
                show("done");
            });"#,
            "null",
        )
        .unwrap();
        let mut outs = vec![];
        for f in 1..=4 {
            outs.push(all(&e.frame(&snapshot(f)).unwrap()));
        }
        assert!(outs[0].contains(r#""Global":[1,0]"#), "{}", outs[0]);
        assert!(outs[2].contains(r#""Global":[3,0]"#), "{}", outs[2]);
        assert!(outs[2].contains("done"), "{}", outs[2]);
        // Normal completion keeps the held command.
        assert!(outs[3].contains(r#""Global":[3,0]"#), "{}", outs[3]);
    }

    #[test]
    fn write_vars_emits_raw_bits() {
        let mut e = engine();
        e.eval(
            "nb",
            r#"__nb.run("c1", async () => {
                robot(4).writeVars({ SAS_RESERVED_1: 1.5 });
                robot(4).writeVars({ SAS_RESERVED_6: 3 }, "u32");
            });"#,
            "null",
        )
        .unwrap();
        let out = all(&e.frame(&snapshot(1)).unwrap());
        // 1.5f32 = 0x3FC00000 = 1069547520
        assert!(out.contains(r#"["SAS_RESERVED_1",1069547520]"#), "{out}");
        assert!(out.contains(r#"["SAS_RESERVED_6",3]"#), "{out}");
        assert!(out.contains("fw.write"), "{out}");
    }

    #[test]
    fn error_releases_and_reports() {
        let mut e = engine();
        e.eval(
            "nb",
            r#"__nb.run("c1", async () => {
                robot(4).set({ velocity: { Global: [100, 0] } });
                await frame();
                throw new Error("boom");
            });"#,
            "null",
        )
        .unwrap();
        let out = all(&e.frame(&snapshot(1)).unwrap());
        assert!(out.contains("boom"), "{out}");
        assert!(!out.contains(r#""Global":[100,0]"#), "{out}");
    }

    #[test]
    fn runaway_loop_is_interrupted() {
        let mut e = engine();
        let err = e.eval(
            "nb",
            r#"__nb.run("c1", async () => { for(;;){} });"#,
            "null",
        );
        assert!(err.is_err());
        // The engine stays usable afterwards.
        e.abort_all("interrupted").unwrap();
        e.frame(&snapshot(1)).unwrap();
    }

    #[test]
    fn trial_records_rows_with_params() {
        let mut e = engine();
        e.eval(
            "nb",
            r#"__nb.run("c1", async (show) => {
                const R = robot(4);
                const t = await trial("leg", { h: 45 }, async () => {
                    for (let i = 0; i < 3; i++) { await frame(); R.log({ lat: i }); }
                });
                const rows = t.rows(R);
                show({ n: rows.length, h: rows[0].h, lat: t.last(R, "lat"), x: rows[2].position_x, status: t.status });
            });"#,
            "null",
        )
        .unwrap();
        let mut out = String::new();
        for f in 1..=5 {
            out.push_str(&all(&e.frame(&snapshot(f)).unwrap()));
        }
        assert!(out.contains("trial.start"), "{out}");
        assert!(out.contains(r#""n":3"#), "{out}");
        assert!(out.contains(r#""h":45"#), "{out}");
        assert!(out.contains(r#""lat":2"#), "{out}");
        assert!(out.contains(r#""status":"ok""#), "{out}");
    }

    #[test]
    fn notebooks_have_separate_scopes() {
        let mut e = engine();
        e.eval(
            "a",
            r#"var x = 1; __nb.run("c1", async (show) => {});"#,
            "null",
        )
        .unwrap();
        e.eval(
            "b",
            r#"__nb.run("c2", async (show) => { show(typeof x); });"#,
            "null",
        )
        .unwrap();
        let out = e.flush("b").unwrap();
        assert!(out.contains("undefined"), "{out}");
    }

    #[test]
    fn cells_queue_within_a_notebook() {
        let mut e = engine();
        e.eval(
            "nb",
            r#"__nb.run("c1", async () => { await frame(); await frame(); });"#,
            "null",
        )
        .unwrap();
        e.eval(
            "nb",
            r#"__nb.run("c2", async (show) => { show("second"); });"#,
            "null",
        )
        .unwrap();
        let first = e.flush("nb").unwrap();
        assert!(!first.contains("second"), "{first}");
        let out1 = all(&e.frame(&snapshot(1)).unwrap());
        assert!(!out1.contains("second"), "{out1}");
        let out2 = all(&e.frame(&snapshot(2)).unwrap());
        assert!(out2.contains("second"), "{out2}");
    }

    #[test]
    fn abort_drops_queued_cells() {
        let mut e = engine();
        e.eval(
            "nb",
            r#"__nb.run("c1", async () => { for(;;) await frame(); });"#,
            "null",
        )
        .unwrap();
        e.eval(
            "nb",
            r#"__nb.run("c2", async (show) => { show("ran"); });"#,
            "null",
        )
        .unwrap();
        let out = all(&e.abort_all("stop").unwrap());
        assert!(!out.contains("ran"), "{out}");
        assert_eq!(out.matches(r#""kind":"aborted""#).count(), 2, "{out}");
    }

    #[test]
    fn default_robot_and_single_robot_shorthand() {
        let mut e = engine();
        e.eval(
            "nb",
            r#"__nb.run("c1", async (show) => { show(robot().id); });"#,
            "null",
        )
        .unwrap();
        let first = e.flush("nb").unwrap();
        assert!(first.contains(r#""value":4"#), "{first}");
        e.eval(
            "nb",
            r#"__nb.run("c2", async (show) => { show({ id: robot().id }); });"#,
            r#"{"team":"Blue","id":7}"#,
        )
        .unwrap();
        let out = e.flush("nb").unwrap();
        assert!(out.contains(r#""id":7"#), "{out}");
    }

    #[test]
    fn reset_drops_scope_and_commands() {
        let mut e = engine();
        e.eval(
            "nb",
            r#"var y = 1; __nb.run("c1", async () => { robot(4).set({ velocity: { Global: [5, 0] } }); });"#,
            "null",
        )
        .unwrap();
        assert!(all(&e.frame(&snapshot(1)).unwrap()).contains(r#""Global":[5,0]"#));
        e.reset("nb").unwrap();
        assert!(!e.has_notebook("nb"));
        assert!(!all(&e.frame(&snapshot(2)).unwrap()).contains("Global"));
    }
}
