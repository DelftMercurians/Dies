"""Read a Dies columnar log (directory or zip) into pandas DataFrames.

Every accessor returns a DataFrame indexed by ``t`` — seconds relative to the
start of the log, so all robots and series share one clock. Window with
``df.loc[t0:t1]`` and take magnitudes with ``np.hypot``.

    log = load("logs/dies-2026-06-21_10-39-19")

    log.robots()                 # -> [('blue', 0), ..., ('yellow', 5)]
    log["players"]               # raw players table, t-indexed
    log.debug("team_Blue.p0.")   # that robot's debug values, pivoted wide
    r = log.robot("blue", 0)     # player frames fused with team_Blue.p0.* debug

A robot frame carries both the structured columns (x, y, vx, vy, yaw, ...) and
its debug values (target_vel_x, target_vel_y, breakbeam, ...). Vec2 debug
strings ("x y") are split into ``<tag>_x`` / ``<tag>_y``. Casing is handled for
you: ``robot("blue", 0)`` filters the lowercase players table and reads the
capitalized ``team_Blue.p0.`` debug keys.

Commanded vs measured speed over the first 15 s:

    r = log.robot("blue", 0).loc[0:15]
    plt.plot(r.index, np.hypot(r.vx, r.vy), label="measured")
    plt.plot(r.index, np.hypot(r.target_vel_x, r.target_vel_y), label="cmd")
    plt.legend()
"""

import io
import json
import pathlib
import zipfile

import numpy as np
import pandas as pd
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

TABLES = (
    "frames",
    "ball",
    "players",
    "debug_values",
    "debug_shapes",
    "debug_tree",
    "settings_changes",
    "events",
    "markers",
    "logs",
    "vision",
)


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _read_parquet(b):
    return pq.read_table(io.BytesIO(b) if isinstance(b, bytes) else b).to_pandas()


def _read_arrow(b):
    with ipc.open_stream(io.BytesIO(b) if isinstance(b, bytes) else b) as r:
        return r.read_all().to_pandas()


def _split_vec2(wide):
    """Split debug columns holding "x y" strings into <col>_x / <col>_y floats."""
    for col in list(wide.columns):
        s = wide[col].dropna()
        if not len(s):
            continue
        toks = str(s.iloc[0]).split()
        if len(toks) == 2 and all(_is_float(t) for t in toks):
            xy = wide[col].astype("string").str.split(expand=True)
            wide[f"{col}_x"] = pd.to_numeric(xy[0], errors="coerce")
            wide[f"{col}_y"] = pd.to_numeric(xy[1], errors="coerce")
            wide = wide.drop(columns=col)
    return wide


class DiesLog:
    def __init__(self, path):
        path = pathlib.Path(path)
        self.path = path
        self.tables = {}
        if path.is_dir():
            self.meta = json.loads((path / "meta.json").read_text())
            for t in TABLES:
                p, a = path / f"{t}.parquet", path / f"{t}.arrow"
                if p.exists():
                    self.tables[t] = _read_parquet(str(p))
                elif a.exists():
                    self.tables[t] = _read_arrow(a.read_bytes())
        else:
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
                self.meta = json.loads(z.read(next(n for n in names if n.endswith("meta.json"))))
                for t in TABLES:
                    p = next((n for n in names if n.endswith(f"{t}.parquet")), None)
                    a = next((n for n in names if n.endswith(f"{t}.arrow")), None)
                    if p:
                        self.tables[t] = _read_parquet(z.read(p))
                    elif a:
                        self.tables[t] = _read_arrow(z.read(a))
        f = self.tables.get("frames")
        self._t0 = float(f["t_received"].min()) if f is not None else 0.0
        self._t_of = dict(zip(f["frame_id"], f["t_received"])) if f is not None else {}

    def _t(self, frame_id):
        return np.array([self._t_of.get(x, np.nan) for x in frame_id]) - self._t0

    def _index_t(self, df):
        """Index a frame by relative time `t`, from frame_id or its own t column."""
        df = df.copy()
        if "frame_id" in df.columns:
            df["t"] = self._t(df["frame_id"].to_numpy())
        elif "t" in df.columns:
            df["t"] = df["t"].to_numpy() - self._t0
        else:
            return df
        return df.set_index("t").sort_index()

    def __getitem__(self, name):
        return self._index_t(self.tables[name])

    def __contains__(self, name):
        return name in self.tables

    def __getattr__(self, name):
        # bare table access: l.markers, l.players, l.ball, ...
        tables = self.__dict__.get("tables", {})
        if name in tables:
            return self[name]
        raise AttributeError(name)

    def seg(self, df, i):
        """The i-th time segment of `df`, cut by markers. Markers split the span
        into len(markers)+1 pieces; `i` may be negative. seg(df, 0) is the span
        up to the first marker, seg(df, -1) the span after the last."""
        bounds = [df.index.min(), *self.markers.index, df.index.max()]
        return df.loc[bounds[i]:bounds[i + 1]]

    def robots(self):
        """Distinct (team, id) present in the players table."""
        p = self.tables["players"]
        return sorted(set(zip(p["team"].astype(str), p["player_id"].astype(int))))

    def _wide(self, prefix=""):
        """`debug_values` pivoted wide, indexed by frame_id (no time)."""
        dv = self.tables["debug_values"][["frame_id", "key", "value", "value_str"]].copy()
        if prefix:
            dv = dv[dv["key"].astype(str).str.startswith(prefix)]
            dv = dv.assign(key=dv["key"].astype(str).str[len(prefix):])
        parts = []
        num = dv[dv["value"].notna()]
        if len(num):
            parts.append(num.pivot_table("value", "frame_id", "key", aggfunc="first"))
        strv = dv[dv["value_str"].notna()]
        if len(strv):
            parts.append(_split_vec2(strv.pivot_table("value_str", "frame_id", "key", aggfunc="first")))
        wide = parts[0].join(parts[1:], how="outer") if parts else pd.DataFrame()
        wide.columns.name = None
        return wide

    def debug(self, prefix=""):
        """`debug_values` pivoted wide on frame_id, t-indexed. With a prefix,
        keep only matching keys and strip it (vec2 strings split into _x/_y)."""
        return self._index_t(self._wide(prefix).reset_index())

    def robot(self, team, player_id):
        """One robot's player frames fused with its team_<Color>.p<id>.* debug."""
        team, player_id = str(team).lower(), int(player_id)
        p = self.tables["players"]
        rows = p[(p["team"].astype(str) == team) & (p["player_id"].astype(int) == player_id)]
        wide = self._wide(f"team_{team.capitalize()}.p{player_id}.")
        merged = rows.set_index("frame_id").join(wide, how="left", rsuffix="_dbg")
        return self._index_t(merged.reset_index())

    def timeline(self, team=None, ax=None):
        """Measured speed of every robot as overlaid filled areas, with markers
        as vertical lines. `team` filters to one color. Returns the Axes."""
        import matplotlib.pyplot as plt

        p = self["players"]
        robots = [r for r in self.robots() if team is None or r[0] == team]
        if ax is None:
            _, ax = plt.subplots(figsize=(15, 3.2))
        colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, max(len(robots), 1)))
        for c, (tm, pid) in zip(colors, robots):
            sub = p[(p["team"].astype(str) == tm) & (p["player_id"].astype(int) == pid)]
            spd = np.hypot(sub["vx"], sub["vy"])
            ax.fill_between(sub.index, spd, color=c, alpha=0.22, lw=0)
            ax.plot(sub.index, spd, color=c, lw=1.0, alpha=0.9, label=f"{tm[0]}{pid}")
        for t in self.markers.index:
            ax.axvline(t, color="0.3", ls="--", lw=0.8, alpha=0.6)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("speed [mm/s]")
        ax.set_ylim(bottom=0)
        ax.margins(x=0)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(ncol=len(robots) or 1, loc="upper right", fontsize=7,
                  framealpha=0.5, columnspacing=1.0, handlelength=1.0)
        return ax


def load(path):
    return DiesLog(path)
