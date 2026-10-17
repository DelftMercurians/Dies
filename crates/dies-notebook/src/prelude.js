// Notebook prelude — the entire user-facing API, written in plain JS on top of
// a data-only boundary with the host:
//
//   host → JS   __nb.frameBegin(snapshotJson)   once per executor frame
//               (promise jobs drained by the host)
//               __nb.frameEnd() → outputJson     commands/debug/events/outputs
//               __nb.flush()    → outputJson     outputs/events only (between frames)
//               __nb.abort(reason)
//   cell code   __nb.run(cellId, async (show) => { ... })
//
// Rule: this file may only contain generic helpers. Nothing written for one
// specific test belongs here (or in the host).
(() => {
  "use strict";

  const S = {
    now: null, // latest snapshot
    waiters: [], // { resolve, reject } awaiting the next frame
    held: new Map(), // robot key → { team, id, cmd, issuedAt, skillWaiters }
    released: new Map(), // robot key → { team, id }, emitted once
    debug: [],
    events: [],
    fw: [], // firmware variable writes: { team, id, vars: [[name, u32]] }
    outputs: [],
    frameLogs: new Map(), // robot key → { name: value } logged this frame
    recorders: new Set(), // active trials
    trialSeq: 0,
    // Cells run one at a time per notebook: each run chains onto `tail`.
    tail: Promise.resolve(),
    // Bumped by abort; queued runs from an older epoch never start.
    epoch: 0,
    // What `robot()` resolves to, chosen in the UI ({ team, id } or null).
    defaultRobot: null,
  };

  const TERMINAL = new Set(["Succeeded", "Failed"]);

  class AbortError extends Error {
    constructor(reason) {
      super(`aborted: ${reason}`);
      this.name = "AbortError";
      this.isAbort = true;
    }
  }

  // ---------------------------------------------------------------- formatting

  function fmtError(e) {
    if (e && typeof e === "object" && "message" in e) {
      const stack = typeof e.stack === "string" ? e.stack.trim() : "";
      return stack ? `${e.name || "Error"}: ${e.message}\n${stack}` : `${e.name || "Error"}: ${e.message}`;
    }
    return String(e);
  }

  function isPlotCall(x) {
    return x !== null && typeof x === "object" && typeof x.__plot === "string";
  }

  /** Field names referenced by a Plot call's options (strings anywhere in
   * them, including nested transforms like Plot.windowY({k}, {x: "t", y: "v"})). */
  function referencedFields(opts, out) {
    if (typeof opts === "string") out.add(opts);
    else if (Array.isArray(opts)) opts.forEach((o) => referencedFields(o, out));
    else if (opts && typeof opts === "object") Object.values(opts).forEach((o) => referencedFields(o, out));
    return out;
  }

  /** Shrink a recorded Plot tree before sending it: mark data (arrays of row
   * objects) keeps only the fields its options reference. Trial rows carry
   * ~100 columns; a plot typically needs 2-4. */
  function prunePlot(x) {
    if (Array.isArray(x)) return x.map(prunePlot);
    if (!x || typeof x !== "object") return x;
    if (isPlotCall(x)) {
      const [data, ...rest] = x.args;
      const isRows =
        Array.isArray(data) && data.length > 0 && data.every((r) => r && typeof r === "object" && !Array.isArray(r));
      const fields = referencedFields(rest, new Set());
      if (isRows && fields.size > 0) {
        const keep = [...fields];
        const slim = data.map((r) => {
          const o = {};
          for (const k of keep) if (k in r) o[k] = r[k];
          return o;
        });
        return { __plot: x.__plot, args: [slim, ...rest.map(prunePlot)] };
      }
      return { __plot: x.__plot, args: x.args.map(prunePlot) };
    }
    const o = {};
    for (const [k, v] of Object.entries(x)) o[k] = prunePlot(v);
    return o;
  }

  /** Encode a value for display in the UI. */
  function encodeShow(x) {
    if (isPlotCall(x)) return { type: "plot", spec: toJson(prunePlot(x)) };
    if (x instanceof Trial) return { type: "json", value: x.summary() };
    if (typeof x === "string") return { type: "text", text: x };
    if (Array.isArray(x) && x.length > 0 && x.every((r) => r && typeof r === "object" && !Array.isArray(r))) {
      return { type: "table", rows: toJson(x.map((r) => (r instanceof Trial ? r.summary() : r))) };
    }
    return { type: "json", value: toJson(x) };
  }

  function toJson(x) {
    return JSON.parse(
      JSON.stringify(x, (_k, v) => {
        if (typeof v === "function") {
          throw new Error(
            "functions can't be sent to the browser (e.g. Plot accessors) — compute a column and pass its field name instead",
          );
        }
        if (typeof v === "number" && !Number.isFinite(v)) return null;
        return v;
      }),
    );
  }

  // ---------------------------------------------------------------- snapshot access

  function normTeam(team) {
    if (team == null) return null;
    const t = String(team).toLowerCase();
    if (t === "blue" || t === "b") return "Blue";
    if (t === "yellow" || t === "y") return "Yellow";
    throw new Error(`unknown team '${team}' (use "blue" or "yellow")`);
  }

  function rosterOf(snap, team) {
    if (!snap || !snap.world) return [];
    return (team === "Blue" ? snap.world.blue_team : snap.world.yellow_team) || [];
  }

  function robotEntry(snap, team, id) {
    if (!snap || !snap.robots) return undefined;
    return snap.robots.find((r) => r.team === team && r.id === id);
  }

  // ---------------------------------------------------------------- robots

  class Robot {
    constructor(id, team) {
      this.id = id;
      this._team = normTeam(team);
    }

    /** Team color, resolved lazily from the controlled teams if not given. */
    get team() {
      if (this._team) return this._team;
      const teams = (S.now && S.now.teams) || [];
      if (teams.length === 1) return teams[0];
      if (teams.length === 0) throw new Error(`robot(${this.id}): no controlled team (is the executor running?)`);
      throw new Error(`robot(${this.id}): both teams are controlled — pass the team, e.g. robot(${this.id}, "blue")`);
    }

    get key() {
      return `${this.team}:${this.id}`;
    }

    /** This frame's tracked state (absolute field coordinates), or undefined if not seen. */
    get state() {
      return rosterOf(S.now, this.team).find((p) => p.id === this.id);
    }

    /** Latest raw robot feedback (telemetry), or undefined. */
    get feedback() {
      const e = robotEntry(S.now, this.team, this.id);
      return e ? e.feedback || undefined : undefined;
    }

    /** Status of the skill the robot executed last frame ("Running", "Succeeded", ...). */
    get status() {
      const e = robotEntry(S.now, this.team, this.id);
      return e ? e.skill || null : null;
    }

    /** Hold a controller input (PlayerControlInput, partial) until changed or released. */
    set(input) {
      this._hold({ Input: input || {} });
    }

    /** Hold a skill command; resolves with "Succeeded" | "Failed" | "Cancelled". */
    skill(cmd) {
      return new Promise((resolve, reject) => {
        const h = this._hold({ Skill: cmd });
        h.skillWaiters.push({ resolve, reject });
      });
    }

    /** Hand the robot back to the strategy (or idle). */
    release() {
      const key = this.key;
      const h = S.held.get(key);
      if (h) settleSkill(h, "Cancelled");
      S.held.delete(key);
      S.released.set(key, { team: this.team, id: this.id });
    }

    /** Write firmware config variables ({ SAS_RESERVED_1: 1.5 }); sticky on the robot until reboot. */
    writeVars(vars, type = "f32") {
      const dv = new DataView(new ArrayBuffer(4));
      const bits = (v) => {
        if (type === "f32") dv.setFloat32(0, v);
        else if (type === "u32" || type === "i32") dv.setInt32(0, v | 0);
        else throw new Error(`writeVars: unknown type ${type} (f32 | u32 | i32)`);
        return dv.getUint32(0);
      };
      const out = Object.entries(vars).map(([name, v]) => [name, bits(v)]);
      S.fw.push({ team: this.team, id: this.id, vars: out });
      emit("fw.write", { team: this.team, id: this.id, type, vars });
    }

    /** Debug values for this frame, recorded in the log as team_<Color>.p<id>.<name>. */
    log(values) {
      const key = this.key;
      const frame = S.frameLogs.get(key) || {};
      for (const [name, v] of Object.entries(values)) {
        const prefix = `team_${this.team}.p${this.id}.${name}`;
        if (Array.isArray(v) && v.length === 2) {
          pushDebug(`${prefix}_x`, v[0]);
          pushDebug(`${prefix}_y`, v[1]);
          frame[`${name}_x`] = v[0];
          frame[`${name}_y`] = v[1];
        } else {
          pushDebug(prefix, v);
          frame[name] = v;
        }
      }
      S.frameLogs.set(key, frame);
    }

    _hold(cmd) {
      const key = this.key;
      const prev = S.held.get(key);
      if (prev) settleSkill(prev, "Cancelled");
      const h = { team: this.team, id: this.id, cmd, issuedAt: null, skillWaiters: [] };
      S.held.set(key, h);
      S.released.delete(key);
      for (const rec of S.recorders) rec.robots.add(key);
      return h;
    }
  }

  function pushDebug(key, value) {
    if (typeof value === "boolean") value = value ? 1 : 0;
    if (typeof value === "number" || typeof value === "string") S.debug.push({ key, value });
  }

  function settleSkill(h, status) {
    const ws = h.skillWaiters;
    h.skillWaiters = [];
    for (const w of ws) w.resolve(status);
  }

  function releaseKeys(keys) {
    for (const key of keys) {
      const h = S.held.get(key);
      if (!h) continue;
      settleSkill(h, "Cancelled");
      S.held.delete(key);
      S.released.set(key, { team: h.team, id: h.id });
    }
  }

  const robots = new Map();

  /** Our robots vision sees, across controlled teams. */
  function visibleOwnRobots() {
    const teams = (S.now && S.now.teams) || [];
    return teams.flatMap((team) => rosterOf(S.now, team).map((p) => ({ team, id: p.id })));
  }

  function robot(id, team) {
    if (id === undefined) {
      // robot(): the robot picked in the UI, else the only one on the field.
      let pick = S.defaultRobot;
      if (!pick) {
        const own = visibleOwnRobots();
        if (own.length !== 1) {
          throw new Error(
            own.length === 0 ? "robot(): no robot on the field" : "robot(): more than one robot — pick one in the editor",
          );
        }
        pick = own[0];
      }
      id = pick.id;
      team = pick.team;
    }
    const k = `${normTeam(team) || "?"}:${id}`;
    if (!robots.has(k)) robots.set(k, new Robot(id, team));
    return robots.get(k);
  }

  // ---------------------------------------------------------------- time

  function frame() {
    return new Promise((resolve, reject) => S.waiters.push({ resolve, reject }));
  }

  async function wait(seconds) {
    const t0 = (S.now || (await frame())).t;
    for (;;) {
      const s = await frame();
      if (s.t - t0 >= seconds) return s;
    }
  }

  async function until(pred, timeout) {
    const t0 = S.now ? S.now.t : null;
    for (;;) {
      const s = await frame();
      if (pred(s)) return s;
      if (timeout != null && t0 != null && s.t - t0 >= timeout) {
        throw new Error(`until: timed out after ${timeout}s`);
      }
    }
  }

  // ---------------------------------------------------------------- events & output

  function emit(type, payload) {
    S.events.push({ type: String(type), payload: payload === undefined ? null : toJson(payload) });
  }

  function makeShow(cell) {
    return (...items) => {
      for (const item of items) S.outputs.push({ cell, kind: "show", data: encodeShow(item) });
    };
  }

  // ---------------------------------------------------------------- trials

  /** Flatten a record into scalar columns: vec2 → _x/_y, arrays → _i, objects → prefix_. */
  function flatten(obj, prefix, out) {
    if (!obj || typeof obj !== "object") return out;
    for (const [k, v] of Object.entries(obj)) {
      const name = prefix + k;
      if (v == null) continue;
      if (typeof v === "number" || typeof v === "boolean" || typeof v === "string") {
        out[name] = v;
      } else if (Array.isArray(v)) {
        if (v.length === 2 && typeof v[0] === "number" && typeof v[1] === "number") {
          out[`${name}_x`] = v[0];
          out[`${name}_y`] = v[1];
        } else {
          v.forEach((x, i) => {
            if (typeof x === "number" || typeof x === "boolean" || typeof x === "string") out[`${name}_${i}`] = x;
          });
        }
      } else if (typeof v === "object") {
        flatten(v, `${name}_`, out);
      }
    }
    return out;
  }

  function robotRef(r, defaultTeam) {
    if (r instanceof Robot) return { team: r.team, id: r.id };
    return { team: normTeam(defaultTeam) || robot(r).team, id: r };
  }

  class Trial {
    constructor(rec) {
      this._rec = rec;
      this.id = rec.id;
      this.name = rec.name;
      this.params = rec.params;
      this.status = rec.status;
      this.error = rec.error;
      this.result = rec.result;
      this.duration = rec.frames.length ? rec.frames[rec.frames.length - 1].snap.t - rec.t0 : 0;
    }

    /** One row per recorded frame for a robot: state, feedback (fb_*), controller
     * info, logged values and the trial params, as flat columns. */
    rows(r, team) {
      const { team: tm, id } = robotRef(r, team);
      const key = `${tm}:${id}`;
      // Cached: rows are immutable once the trial ended, and building them is
      // the expensive part of most analyses.
      this._rows = this._rows || new Map();
      if (this._rows.has(key)) return this._rows.get(key);
      const out = [];
      for (const { snap, logs } of this._rec.frames) {
        const p = rosterOf(snap, tm).find((x) => x.id === id);
        if (!p) continue;
        const row = { t: snap.t - this._rec.t0, frame: snap.frame };
        flatten(p, "", row);
        const e = robotEntry(snap, tm, id);
        if (e) {
          flatten(e.feedback, "fb_", row);
          const { feedback: _f, team: _t, id: _i, ...ctl } = e;
          flatten(ctl, "", row);
        }
        Object.assign(row, logs.get(key) || {});
        Object.assign(row, this.params, { trial: this.id });
        out.push(row);
      }
      this._rows.set(key, out);
      return out;
    }

    col(r, name, team) {
      return this.rows(r, team).map((row) => row[name]);
    }

    last(r, name, team) {
      const rows = this.rows(r, team);
      for (let i = rows.length - 1; i >= 0; i--) if (rows[i][name] !== undefined) return rows[i][name];
      return undefined;
    }

    summary() {
      return { trial: this.id, name: this.name, ...this.params, status: this.status, duration: this.duration, error: this.error };
    }
  }

  /** Run `body` as a recorded, logged trial. Never throws on failure (status
   * "failed", robots commanded inside it are released); aborts propagate. */
  async function trial(name, params, body) {
    if (typeof params === "function") {
      body = params;
      params = {};
    }
    params = params || {};
    const id = ++S.trialSeq;
    const rec = { id, name, params, frames: [], robots: new Set(), t0: S.now ? S.now.t : 0, status: "running" };
    S.recorders.add(rec);
    emit("trial.start", { id, name, params });
    try {
      rec.result = await body();
      rec.status = "ok";
    } catch (e) {
      if (e && e.isAbort) {
        rec.status = "aborted";
        throw e;
      }
      rec.status = "failed";
      rec.error = e && e.message ? e.message : String(e);
      releaseKeys([...rec.robots]);
    } finally {
      S.recorders.delete(rec);
      // The frame the body finished in hasn't reached frameEnd yet; keep it.
      const lastRec = rec.frames[rec.frames.length - 1];
      if (S.now && (!lastRec || lastRec.snap.frame !== S.now.frame)) {
        rec.frames.push({ snap: S.now, logs: S.frameLogs });
      }
      if (rec.frames.length) rec.t0 = rec.frames[0].snap.t;
      const duration = rec.frames.length ? rec.frames[rec.frames.length - 1].snap.t - rec.t0 : 0;
      emit("trial.end", { id, name, status: rec.status, error: rec.error, duration });
    }
    return new Trial(rec);
  }

  // ---------------------------------------------------------------- plotting

  // `Plot.*` calls are recorded as data and replayed by the browser with the real
  // Observable Plot library. Options must be field names, not functions.
  function plotProxy(path) {
    return new Proxy(function () {}, {
      get(_t, prop) {
        if (typeof prop !== "string") return undefined;
        return plotProxy(path ? `${path}.${prop}` : prop);
      },
      apply(_t, _this, args) {
        return { __plot: path, args };
      },
    });
  }

  // ---------------------------------------------------------------- math & stats

  const v2 = {
    add: (a, b) => [a[0] + b[0], a[1] + b[1]],
    sub: (a, b) => [a[0] - b[0], a[1] - b[1]],
    mul: (a, s) => [a[0] * s, a[1] * s],
    dot: (a, b) => a[0] * b[0] + a[1] * b[1],
    /** z of the 3D cross product: positive when b is left of a. */
    cross: (a, b) => a[0] * b[1] - a[1] * b[0],
    norm: (a) => Math.hypot(a[0], a[1]),
    dist: (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]),
    unit: (a) => {
      const n = Math.hypot(a[0], a[1]);
      return n > 0 ? [a[0] / n, a[1] / n] : [0, 0];
    },
    rot: (a, ang) => [a[0] * Math.cos(ang) - a[1] * Math.sin(ang), a[0] * Math.sin(ang) + a[1] * Math.cos(ang)],
    angle: (a) => Math.atan2(a[1], a[0]),
    fromAngle: (ang, len = 1) => [len * Math.cos(ang), len * Math.sin(ang)],
    lerp: (a, b, s) => [a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s],
  };

  const deg = (d) => (d * Math.PI) / 180;
  const toDeg = (r) => (r * 180) / Math.PI;
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  const wrapAngle = (a) => Math.atan2(Math.sin(a), Math.cos(a));

  const nums = (xs) => xs.filter((x) => typeof x === "number" && Number.isFinite(x));
  const stats = {
    mean: (xs) => {
      const v = nums(xs);
      return v.length ? v.reduce((s, x) => s + x, 0) / v.length : NaN;
    },
    rms: (xs) => {
      const v = nums(xs);
      return v.length ? Math.sqrt(v.reduce((s, x) => s + x * x, 0) / v.length) : NaN;
    },
    std: (xs) => {
      const v = nums(xs);
      if (v.length < 2) return NaN;
      const m = v.reduce((s, x) => s + x, 0) / v.length;
      return Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / (v.length - 1));
    },
    maxAbs: (xs) => nums(xs).reduce((m, x) => Math.max(m, Math.abs(x)), 0),
    /** Least-squares y = slope·x + intercept, with Pearson r. */
    linfit: (xs, ys) => {
      const pts = xs.map((x, i) => [x, ys[i]]).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
      const n = pts.length;
      if (n < 2) return { slope: NaN, intercept: NaN, r: NaN, n };
      const mx = pts.reduce((s, p) => s + p[0], 0) / n;
      const my = pts.reduce((s, p) => s + p[1], 0) / n;
      let sxx = 0, syy = 0, sxy = 0;
      for (const [x, y] of pts) {
        sxx += (x - mx) ** 2;
        syy += (y - my) ** 2;
        sxy += (x - mx) * (y - my);
      }
      const slope = sxy / sxx;
      return { slope, intercept: my - slope * mx, r: sxy / Math.sqrt(sxx * syy), n };
    },
  };

  // ---------------------------------------------------------------- host entry points

  function takeOutputs() {
    const out = { outputs: S.outputs, events: S.events };
    S.outputs = [];
    S.events = [];
    return out;
  }

  globalThis.__nb = {
    run(cell, fn) {
      const epoch = S.epoch;
      S.outputs.push({ cell, kind: "queued" });
      const p = S.tail.then(() => {
        if (S.epoch !== epoch) throw new AbortError("stopped");
        S.outputs.push({ cell, kind: "started" });
        return fn(makeShow(cell));
      });
      S.tail = p.catch(() => {});
      p.then(
        () => S.outputs.push({ cell, kind: "done" }),
        (e) => {
          if (e && e.isAbort) {
            S.outputs.push({ cell, kind: "aborted", data: e.message });
          } else {
            S.outputs.push({ cell, kind: "error", data: fmtError(e) });
            // Safety: an uncaught error stops every held robot.
            releaseKeys([...S.held.keys()]);
          }
        },
      );
    },

    setDefaultRobot(json) {
      const r = JSON.parse(json);
      S.defaultRobot = r && r.team != null && r.id != null ? { team: normTeam(r.team), id: r.id } : null;
    },

    frameBegin(json) {
      const snap = JSON.parse(json);
      S.now = snap;
      S.frameLogs = new Map();
      // Resolve skill waiters whose command has been processed by the executor.
      for (const h of S.held.values()) {
        if (h.issuedAt == null || h.skillWaiters.length === 0) continue;
        if (snap.frame <= h.issuedAt) continue;
        const e = robotEntry(snap, h.team, h.id);
        if (e && TERMINAL.has(e.skill)) settleSkill(h, e.skill);
      }
      const ws = S.waiters;
      S.waiters = [];
      for (const w of ws) w.resolve(snap);
    },

    frameEnd() {
      const commands = [];
      for (const h of S.held.values()) {
        if (h.issuedAt == null && S.now) h.issuedAt = S.now.frame;
        commands.push({ team: h.team, id: h.id, cmd: h.cmd, since: h.issuedAt });
      }
      for (const rec of S.recorders) {
        if (S.now) rec.frames.push({ snap: S.now, logs: S.frameLogs });
      }
      const out = {
        commands,
        release: [...S.released.values()],
        debug: S.debug,
        fw: S.fw,
        ...takeOutputs(),
      };
      S.released = new Map();
      S.debug = [];
      S.fw = [];
      return JSON.stringify(out);
    },

    flush() {
      return JSON.stringify(takeOutputs());
    },

    /** Another notebook took these robots over: forget our hold (no release). */
    dropHeld(json) {
      for (const r of JSON.parse(json)) {
        const key = `${r.team}:${r.id}`;
        const h = S.held.get(key);
        if (!h) continue;
        settleSkill(h, "Cancelled");
        S.held.delete(key);
      }
    },

    abort(reason) {
      const err = new AbortError(reason || "stopped");
      S.epoch++;
      const ws = S.waiters;
      S.waiters = [];
      for (const w of ws) w.reject(err);
      for (const h of S.held.values()) {
        const sw = h.skillWaiters;
        h.skillWaiters = [];
        for (const w of sw) w.reject(err);
      }
      releaseKeys([...S.held.keys()]);
    },
  };

  Object.defineProperty(globalThis, "now", { get: () => S.now, configurable: false });
  // Reserved: cells can't overwrite the API (the UI also rejects these names).
  const api = { frame, robot, emit, wait, until, trial, Plot: plotProxy(""), v2, deg, toDeg, clamp, wrapAngle, stats };
  for (const [name, value] of Object.entries(api)) {
    Object.defineProperty(globalThis, name, { value, writable: false, configurable: false, enumerable: true });
  }
  Object.defineProperty(globalThis, "__nb", { value: globalThis.__nb, writable: false, configurable: false });
})();
