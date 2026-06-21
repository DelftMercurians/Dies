import io
import json
import pathlib
import zipfile

import numpy as np
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


def _to_numpy(table):
    return {
        name: table.column(name).to_numpy(zero_copy_only=False)
        for name in table.column_names
    }


def _read_parquet(b):
    return _to_numpy(pq.read_table(io.BytesIO(b) if isinstance(b, bytes) else b))


def _read_arrow(b):
    with ipc.open_stream(io.BytesIO(b) if isinstance(b, bytes) else b) as r:
        return _to_numpy(r.read_all())


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
        self.t_of = {}
        f = self.tables.get("frames")
        if f:
            self.t_of = dict(zip(f["frame_id"], f["t_received"]))

    def __getitem__(self, name):
        return self.tables[name]

    def __contains__(self, name):
        return name in self.tables

    def __getattr__(self, name):
        tables = self.__dict__.get("tables", {})
        if name in tables:
            return tables[name]
        raise AttributeError(name)

    def t(self, frame_id):
        return np.array([self.t_of.get(f, np.nan) for f in np.asarray(frame_id)])

    def value_keys(self):
        dv = self.tables.get("debug_values")
        return np.unique(dv["key"]) if dv else np.array([], dtype=object)

    def shape_keys(self):
        ds = self.tables.get("debug_shapes")
        return np.unique(ds["key"]) if ds else np.array([], dtype=object)

    def value(self, key):
        dv = self.tables["debug_values"]
        m = dv["key"] == key
        out = {c: v[m] for c, v in dv.items()}
        out["t"] = self.t(out["frame_id"])
        return out

    def shape(self, key):
        ds = self.tables["debug_shapes"]
        m = ds["key"] == key
        out = {c: v[m] for c, v in ds.items()}
        out["t"] = self.t(out["frame_id"])
        return out

    def vec2(self, key):
        d = self.value(key)
        s = d["value_str"]
        xy = np.array([t.split() for t in s], dtype=float) if len(s) else np.zeros((0, 2))
        return {"frame_id": d["frame_id"], "t": d["t"], "x": xy[:, 0], "y": xy[:, 1]}

    def command(self, team, player_id):
        d = self.vec2(f"team_{team}.p{player_id}.target_vel")
        return {"frame_id": d["frame_id"], "t": d["t"], "vx": d["x"], "vy": d["y"]}


def load(path):
    return DiesLog(path)
