"""Match analytics on top of ``dieslog`` — possession, shots, stoppages, goals.

Reads Dies binary logs (the ones ``dies-cli self-play --log-dir`` now writes) and
turns one match into a flat dict of metrics, or many matches into a summary
DataFrame + printed report. Built entirely on :mod:`dieslog`; it does not modify
it.

    from match_analytics import analyze, summarize, report, benchmark

    a  = analyze("logs/selfplay/selfplay_v0-strategy_vs_concerto_seed1")
    df = summarize(glob.glob("logs/selfplay/selfplay_*"))
    report(df)

    # run + analyze in one go (shells out to `cargo run -p dies-cli -- self-play`)
    df, rows = benchmark("v0-strategy", "concerto", seeds=range(5), duration=300)
    report(df)

What each metric means
----------------------
* **run_frac** — fraction of wall(sim)-time the ball was actually in play
  (game_state in Run/Kickoff/FreeKick/Penalty), vs stopped for the auto-ref.
* **n_stoppages** — number of times play entered a stopped state, with a
  per-cause breakdown from the ``sim_referee`` events (NoProgress, Boundary
  Crossing, Goal, …).
* **possession** — dt-weighted time each team was the *confident* ball owner
  (the world tracker's ``has_ball``: breakbeam-latched + proximity). Sparse by
  design — it's true control, not "near the ball".
* **control** — looser proxy: dt-weighted time each team had the nearest robot
  within ``CONTROL_RADIUS`` of the ball while in play.
* **ball_owned_frac** — fraction of in-play time *anyone* owned the ball; low
  values mean the ball spends most of the match loose.
* **shots / sot** — ball-speed kicks above ``SHOT_SPEED`` headed at a goal,
  attributed to the attacking team by direction; ``sot`` = on target (projected
  to cross the goal mouth).
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import dieslog

# --- tunables (mm, mm/s, s) -------------------------------------------------
SHOT_SPEED = 1500.0  # ball speed that counts as a kick/shot attempt
SHOT_DEBOUNCE = 0.4  # min gap between two counted shots
CONTROL_RADIUS = 500.0  # nearest robot within this => that team "controls" the ball
PLAY_STATES = {"Run", "Kickoff", "FreeKick", "Penalty", "PenaltyRun"}

# --- possession-chain / pass tunables ---------------------------------------
# A team must be the nearest-in-radius controller for at least MIN_DWELL before it
# counts as having *established* control — so a defender the ball merely flies past
# during a pass doesn't register as a turnover. A possession chain bridges loose
# ("none") frames and sub-dwell opponent touches; it ends as a TURNOVER when the
# other team establishes control, or as DEAD on a stoppage/goal (out-of-play gap).
MIN_DWELL = 0.30        # s of continuous nearest-control to "establish" possession
STOPPAGE_GAP = 0.50     # s gap in in-play frames => dead-ball chain boundary
KICK_SPEED = 800.0      # rising edge above this = a kick (pass or shot)
KICK_DEBOUNCE = 0.30    # min gap between counted kicks
PASS_WINDOW = 2.0       # s after a pass within which the next controller is judged

REPO_DEFAULT = Path(__file__).resolve().parent.parent


def _as_log(x):
    return x if isinstance(x, dieslog.DiesLog) else dieslog.load(x)


def _seed_from_name(name):
    m = re.search(r"seed(\d+)", str(name))
    return int(m.group(1)) if m else None


def _pos_defender(side):
    """The team defending the +x goal (so a shot toward +x belongs to the other)."""
    return "yellow" if str(side).startswith("yellow") else "blue"


def possession_metrics(log):
    """Possession-chain, turnover, and pass-completion proxies for one match.

    These are the *optimization targets* for reducing ball loss: they measure how
    well each team keeps and circulates the ball, independent of the scoreline.

    Method: build a per-in-play-frame "controller" (the team whose robot is nearest
    the ball within ``CONTROL_RADIUS``, else loose). Collapse into segments; a team
    is the *established* controller once its segment lasts ``MIN_DWELL`` (so a ball
    flying past a defender mid-pass isn't a turnover). A possession **chain** is a
    maximal run of one team's established control, bridging loose frames and brief
    opponent touches; it ends as a **turnover** when the other team establishes
    control, or **dead** on a stoppage/goal. A **pass** is a kick (ball-speed rising
    edge) by the controlling team that isn't a shot on goal; it's **completed** if
    the same team is the next established controller within ``PASS_WINDOW``.

    Returns per-team: ``{t}_chains``, ``{t}_chain_s`` (mean chain duration),
    ``{t}_ctrl_s`` (total established-control time), ``{t}_turnovers``,
    ``{t}_turnovers_per_min`` (per minute of own control), ``{t}_pass_att``,
    ``{t}_pass_comp``, ``{t}_pass_pct``.
    """
    log = _as_log(log)
    f = log["frames"]
    if not len(f):
        return {}
    fmap = f.reset_index().rename(columns={f.index.name or "index": "t"})
    fmap = fmap[["frame_id", "t", "dt", "game_state"]]

    # Per-frame nearest team to the ball.
    ball = log["ball"].reset_index()[["frame_id", "x", "y"]].rename(
        columns={"x": "bx", "y": "by"}
    )
    p = log["players"].reset_index()
    pc = p[["frame_id", "team", "x", "y"]].merge(ball, on="frame_id", how="inner")
    pc["d"] = np.hypot(pc["x"] - pc["bx"], pc["y"] - pc["by"])
    near = pc.loc[pc.groupby("frame_id")["d"].idxmin(), ["frame_id", "team", "d"]]
    tl = fmap.merge(near, on="frame_id", how="left").sort_values("t")
    inplay = tl["game_state"].isin(PLAY_STATES)
    ctrl = np.where(inplay & (tl["d"] <= CONTROL_RADIUS), tl["team"].fillna("none"), "none")
    tl = tl.assign(ctrl=ctrl, inplay=inplay)
    tl = tl[tl["inplay"]].reset_index(drop=True)
    if not len(tl):
        return {}

    tt = tl["t"].to_numpy()
    cc = tl["ctrl"].to_numpy()

    # 1. Collapse to (team, t_start, t_end) segments, splitting on stoppage gaps.
    segs = []  # (team_or_none, t0, t1, dead_after)
    i = 0
    n = len(tt)
    while i < n:
        j = i
        while j + 1 < n and cc[j + 1] == cc[i] and (tt[j + 1] - tt[j]) < STOPPAGE_GAP:
            j += 1
        dead_after = j + 1 < n and (tt[j + 1] - tt[j]) >= STOPPAGE_GAP
        segs.append([cc[i], tt[i], tt[j], dead_after or (j + 1 >= n)])
        i = j + 1

    # 2. Established controllers: team segments lasting >= MIN_DWELL.
    established = []  # (team, t0, t1, dead_after)
    for team, t0, t1, dead in segs:
        if team != "none" and (t1 - t0) >= MIN_DWELL:
            established.append((team, t0, t1, dead))
        elif dead and established:
            # A stoppage during a loose/sub-dwell stretch still ends the chain.
            established[-1] = (established[-1][0], established[-1][1],
                               established[-1][2], True)

    # 3. Walk established controllers into chains.
    chains = []  # (team, t_start, t_end, end_kind)  end_kind in turnover|dead|end
    cur = None
    for k, (team, t0, t1, dead) in enumerate(established):
        if cur is None:
            cur = [team, t0, t1]
        elif team == cur[0]:
            cur[2] = t1
        else:
            chains.append((cur[0], cur[1], cur[2], "turnover"))
            cur = [team, t0, t1]
        if dead:
            chains.append((cur[0], cur[1], cur[2], "dead"))
            cur = None
    if cur is not None:
        chains.append((cur[0], cur[1], cur[2], "end"))

    # 4. Kicks (rising edges of ball speed), for pass attribution.
    b = log["ball"]
    bt = b.index.to_numpy()
    vx, vy = b["vx"].to_numpy(), b["vy"].to_numpy()
    spd = np.hypot(vx, vy)
    side = log.meta.get("side_assignment", "yellow_on_positive")
    posdef = _pos_defender(side)
    attacker_pos = "blue" if posdef == "yellow" else "yellow"
    kicks = []  # (t, team_attributed, is_shot)
    if len(spd):
        above = spd > KICK_SPEED
        rising = np.concatenate([[above[0]], above[1:] & ~above[:-1]])
        last = -1e9
        for idx in np.nonzero(rising)[0]:
            tk = bt[idx]
            if tk - last < KICK_DEBOUNCE or vx[idx] == 0:
                continue
            last = tk
            # Attribute to the team controlling at/just-before the kick.
            owner = None
            for team, t0, t1, _ in established:
                if t0 - 0.2 <= tk <= t1 + 0.2:
                    owner = team
                    break
            if owner is None:
                # last established controller before the kick
                prev = [e for e in established if e[1] <= tk]
                owner = prev[-1][0] if prev else None
            if owner is None:
                continue
            toward_pos = vx[idx] > 0
            shooter_dir = attacker_pos if toward_pos else posdef
            is_shot = spd[idx] > SHOT_SPEED and shooter_dir == owner
            kicks.append((tk, owner, is_shot))

    # 5. Pass completion: next established controller within PASS_WINDOW.
    def next_controller(tk):
        for team, t0, t1, _ in established:
            if t0 >= tk:
                if t0 - tk <= PASS_WINDOW:
                    return team
                return None
        return None

    out = {}
    for team in ("yellow", "blue"):
        tc = [c for c in chains if c[0] == team]
        ctrl_s = sum(t1 - t0 for _, t0, t1, _ in tc)
        n_to = sum(1 for c in tc if c[3] == "turnover")
        durs = [t1 - t0 for _, t0, t1, _ in tc]
        passes = [k for k in kicks if k[1] == team and not k[2]]
        comp = sum(1 for tk, _, _ in passes if next_controller(tk) == team)
        att = len(passes)
        out[f"{team}_chains"] = len(tc)
        out[f"{team}_chain_s"] = round(float(np.mean(durs)), 2) if durs else 0.0
        out[f"{team}_ctrl_s"] = round(ctrl_s, 1)
        out[f"{team}_turnovers"] = n_to
        out[f"{team}_turnovers_per_min"] = (
            round(60.0 * n_to / ctrl_s, 2) if ctrl_s > 1 else None
        )
        out[f"{team}_pass_att"] = att
        out[f"{team}_pass_comp"] = comp
        out[f"{team}_pass_pct"] = round(100.0 * comp / att, 1) if att else None
    return out


def analyze(log):
    """Compute a flat dict of analytics for one match log (path or DiesLog)."""
    log = _as_log(log)
    meta = log.meta
    side = meta.get("side_assignment", "yellow_on_positive")
    fg = meta.get("field_geom") or {}
    goal_x = float(fg.get("field_length", 9000.0)) / 2.0
    goal_half = float(fg.get("goal_width", 1000.0)) / 2.0
    posdef = _pos_defender(side)
    attacker_pos = "blue" if posdef == "yellow" else "yellow"  # attacks the +x goal

    f = log["frames"]
    dur = float(f.index.max() - f.index.min()) if len(f) else 0.0
    state_s = f.groupby("game_state")["dt"].sum()
    playing_s = float(sum(v for k, v in state_s.items() if k in PLAY_STATES))
    stopped_s = float(state_s.sum() - playing_s)

    # Stoppage transitions: entries into any non-play state.
    gs = f["game_state"].to_numpy()
    entered = np.concatenate([[True], gs[1:] != gs[:-1]]) if len(gs) else gs
    stoppages = {}
    for st, en in zip(gs, entered):
        if en and st not in PLAY_STATES:
            stoppages[st] = stoppages.get(st, 0) + 1

    # Sim referee events: goals (with scorer) + stoppage causes.
    sim_events, goals = {}, []
    gb = gy = 0
    if "events" in log:
        ev = log["events"]
        for t, row in ev.iterrows():
            if row["event_type"] != "sim_referee":
                continue
            pl = json.loads(row["payload_json"])
            kind, team = pl.get("kind"), pl.get("team")
            sim_events[kind] = sim_events.get(kind, 0) + 1
            if kind == "Goal":
                goals.append({"t": round(float(t), 2), "team": team})
                gb += team == "Blue"
                gy += team == "Yellow"

    # Per-frame dt + state, to weight time-based aggregates.
    fr = f.reset_index()[["frame_id", "dt", "game_state"]]
    p = log["players"].reset_index()

    # Possession: confident owner (has_ball), dt-weighted. Older logs lack it.
    poss_blue = poss_yellow = float("nan")
    if "has_ball" in p.columns:
        owned = p[p["has_ball"]].merge(fr, on="frame_id", how="left")
        hs = owned.groupby("team")["dt"].sum()
        poss_blue, poss_yellow = float(hs.get("blue", 0.0)), float(hs.get("yellow", 0.0))

    # Control proxy: nearest robot within CONTROL_RADIUS of the ball, in play.
    ball = log["ball"].reset_index()[["frame_id", "x", "y"]].rename(
        columns={"x": "bx", "y": "by"}
    )
    pc = p[["frame_id", "team", "x", "y"]].merge(ball, on="frame_id", how="inner")
    pc["d"] = np.hypot(pc["x"] - pc["bx"], pc["y"] - pc["by"])
    near = pc.loc[pc.groupby("frame_id")["d"].idxmin(), ["frame_id", "team", "d"]]
    near = near.merge(fr, on="frame_id", how="left")
    near = near[near["game_state"].isin(PLAY_STATES) & (near["d"] <= CONTROL_RADIUS)]
    cs = near.groupby("team")["dt"].sum()
    ctrl_blue, ctrl_yellow = float(cs.get("blue", 0.0)), float(cs.get("yellow", 0.0))

    # Shots: rising edges of ball speed above SHOT_SPEED, debounced, classified.
    b = log["ball"]
    t = b.index.to_numpy()
    bx, by = b["x"].to_numpy(), b["y"].to_numpy()
    vx, vy = b["vx"].to_numpy(), b["vy"].to_numpy()
    spd = np.hypot(vx, vy)
    shots = {"blue": 0, "yellow": 0}
    sot = {"blue": 0, "yellow": 0}
    if len(spd):
        above = spd > SHOT_SPEED
        rising = np.concatenate([[above[0]], above[1:] & ~above[:-1]])
        last = -1e9
        for i in np.nonzero(rising)[0]:
            if t[i] - last < SHOT_DEBOUNCE or vx[i] == 0:
                continue
            last = t[i]
            toward_pos = vx[i] > 0
            shooter = attacker_pos if toward_pos else posdef
            shots[shooter] += 1
            gx = goal_x if toward_pos else -goal_x
            tt = (gx - bx[i]) / vx[i]
            if tt > 0 and abs(by[i] + vy[i] * tt) <= goal_half:
                sot[shooter] += 1

    def pct(a, b):
        return round(100 * a / (a + b), 1) if (a + b) else None

    return {
        "name": log.path.name,
        "seed": _seed_from_name(log.path.name),
        "blue": meta.get("blue_strategy"),
        "yellow": meta.get("yellow_strategy"),
        "duration_s": round(dur, 1),
        "playing_s": round(playing_s, 1),
        "run_frac": round(playing_s / dur, 3) if dur else 0.0,
        "stopped_s": round(stopped_s, 1),
        "n_stoppages": int(sum(stoppages.values())),
        "stoppages": stoppages,
        "sim_events": sim_events,
        "goals_blue": gb,
        "goals_yellow": gy,
        "goals": goals,
        "poss_blue_s": round(poss_blue, 2),
        "poss_yellow_s": round(poss_yellow, 2),
        "poss_blue_pct": pct(poss_blue, poss_yellow),
        "control_blue_pct": pct(ctrl_blue, ctrl_yellow),
        "ball_owned_frac": (
            round((poss_blue + poss_yellow) / playing_s, 4)
            if playing_s and not np.isnan(poss_blue)
            else None
        ),
        "shots_blue": shots["blue"],
        "shots_yellow": shots["yellow"],
        "sot_blue": sot["blue"],
        "sot_yellow": sot["yellow"],
        "max_ball_speed": round(float(spd.max()), 0) if len(spd) else 0.0,
        **possession_metrics(log),
    }


SUMMARY_COLS = [
    "seed", "blue", "yellow", "duration_s", "run_frac", "n_stoppages",
    "ball_owned_frac", "poss_blue_pct", "control_blue_pct",
    "shots_blue", "shots_yellow", "sot_blue", "sot_yellow",
    "goals_blue", "goals_yellow",
]


def summarize(items):
    """Build a per-match DataFrame from logs / paths / analyze() dicts."""
    rows = [x if isinstance(x, dict) else analyze(x) for x in items]
    return pd.DataFrame(rows)


def report(df):
    """Print a per-match table, an aggregate line, and a short diagnosis."""
    if isinstance(df, list):
        df = summarize(df)
    cols = [c for c in SUMMARY_COLS if c in df.columns]
    print(df[cols].to_string(index=False))
    n = len(df)
    gb, gy = int(df["goals_blue"].sum()), int(df["goals_yellow"].sum())
    blue = df["blue"].iloc[0] if n else "blue"
    yellow = df["yellow"].iloc[0] if n else "yellow"
    print("\n" + "-" * 60)
    print(f"matches: {n}   {blue} {gb} - {gy} {yellow}   (total goals)")
    print(f"mean run fraction : {df['run_frac'].mean():.1%}")
    print(f"total stoppages   : {int(df['n_stoppages'].sum())} "
          f"(mean {df['n_stoppages'].mean():.1f}/match)")
    if "ball_owned_frac" in df and df["ball_owned_frac"].notna().any():
        print(f"ball-owned frac   : {df['ball_owned_frac'].mean():.2%} of in-play time")
    print(f"shots (blue/yellow): {int(df['shots_blue'].sum())}/"
          f"{int(df['shots_yellow'].sum())}   "
          f"on target: {int(df['sot_blue'].sum())}/{int(df['sot_yellow'].sum())}")
    # Aggregate sim-event causes across matches.
    causes = {}
    for d in df.get("sim_events", []):
        if isinstance(d, dict):
            for k, v in d.items():
                causes[k] = causes.get(k, 0) + v
    if causes:
        print("stoppage causes   : "
              + ", ".join(f"{k}={v}" for k, v in sorted(causes.items())))


# --- optional: run matches via the CLI, then analyze ------------------------

def run_match(blue, yellow, seed, duration=300, log_dir="logs/selfplay",
              repo=REPO_DEFAULT, launch=False):
    """Run one self-play match via the CLI; return the parsed MatchResult dict
    (which includes ``log_path``)."""
    repo = Path(repo)
    out = Path(tempfile.mkstemp(suffix=".json")[1])
    # `--launch` is a top-level flag (skip rebuild); it must precede the subcommand.
    globals_ = ["--launch"] if launch else []
    args = ["cargo", "run", "-q", "-p", "dies-cli", "--", *globals_, "self-play",
            "--blue-strategy", blue, "--yellow-strategy", yellow,
            "--seed", str(seed), "--duration", str(duration),
            "--log-dir", str(log_dir), "--output", str(out)]
    r = subprocess.run(args, cwd=repo, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"self-play seed {seed} failed:\n{r.stderr[-2000:]}")
    res = json.loads(out.read_text())
    out.unlink(missing_ok=True)
    return res


def benchmark(blue, yellow, seeds, duration=300, log_dir="logs/selfplay",
              repo=REPO_DEFAULT):
    """Run a seed sweep and analyze each match. Returns (DataFrame, rows).

    Builds the strategies on the first match, then ``--launch`` (no rebuild) for
    the rest so the sweep is fast.
    """
    repo = Path(repo)
    rows = []
    for i, s in enumerate(seeds):
        res = run_match(blue, yellow, s, duration, log_dir, repo, launch=(i > 0))
        path = Path(res["log_path"])
        if not path.is_absolute():
            path = repo / path
        a = analyze(path)
        a["match_blue"], a["match_yellow"] = res["blue_score"], res["yellow_score"]
        rows.append(a)
    return summarize(rows), rows


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        print("usage: match_analytics.py <log_path> [<log_path> ...]")
        raise SystemExit(2)
    report(summarize(paths))
