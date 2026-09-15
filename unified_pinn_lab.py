"""
NUMERICAL PINN EXPERIMENTATION AND EVALUATION FRAMEWORK
=======================================================

One-file numerical twin, rapid retraining, and ablation framework for the Kepler/J2 residual PINN.

It combines three workflows:

1. FIXED-WEIGHT CHECKPOINT EVALUATION
   - loads the real trained checkpoint
   - uses the production Kepler/J2/OSC2MEAN library when available
   - reproduces Linear -> Tanh hidden layers numerically
   - evaluates exact and counterfactual normalization/gate configurations
   - reports accuracy, RSW error, activation saturation, gate suppression,
     baseline gap closed, optional physics residual, and pass/fail metrics

2. RAPID CONFIGURATION SCREENING
   - preferably trains on a propagated CSV dataset
   - otherwise generates truth with Orekit when available, else SciPy DOP853
   - trains a new small/full-width residual MLP for each proposed configuration
   - compares gates, time normalization, and physical feature sets
   - saves checkpoints, metrics, plots, and a ranked decision report

3. FEATURE AND PHYSICS ABLATION
   - measures which orbital basis functions explain the analytical residual
   - compares per-satellite, shared, and orbit-modulated fits
   - measures PDE/data gradient alignment across feature configurations
   - runs paired proxy-network experiments with and without physics losses

Nothing must be edited inside this file. Use command-line options, automatic discovery,
or an optional JSON experiment file.

Typical commands
----------------
Discover files:
    python unified_pinn_lab.py discover --root /path/to/project

Evaluate a trained checkpoint:
    python unified_pinn_lab.py evaluate --root /path/to/project --horizon 40000

Rapidly screen new configurations using an auto-discovered propagated CSV:
    python unified_pinn_lab.py sweep --root /path/to/project --horizon 80000 \
        --sweep-preset j2 --epochs 400 --train-sats 200 --val-sats 60

Run both workflows:
    python unified_pinn_lab.py all --root /path/to/project --horizon 80000

Run feature and physics ablation together:
    python unified_pinn_lab.py ablation --data /path/to/propagated_40k.csv

Dependencies
------------
Required: numpy, pandas, scipy, torch, matplotlib
Optional: orekit (only for the no-CSV Orekit truth fallback)
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import torch
import torch.nn as nn

# Harmless compatibility shim for some PyTorch nightly builds.
try:
    import torch._utils
    torch._utils = sys.modules["torch._utils"]
    if not hasattr(torch._utils, "_element_size"):
        def _element_size(dtype):
            if dtype.is_floating_point:
                return torch.finfo(dtype).bits // 8
            if dtype == torch.bool:
                return 1
            return torch.iinfo(dtype).bits // 8
        torch._utils._element_size = _element_size
except Exception:
    pass


# =============================================================================
# Constants and standard columns
# =============================================================================

MU_EARTH = 3.986004418e14
R_EARTH = 6378137.0
J2_EARTH = 1.08262668e-3
V0_SCALE_DEFAULT = 7500.0

POSITION_COLUMN_SETS = [
    ["x_eme2000_km", "y_eme2000_km", "z_eme2000_km"],
    ["x_km", "y_km", "z_km"],
    ["x", "y", "z"],
]
VELOCITY_COLUMN_SETS = [
    ["vx_eme2000_km_s", "vy_eme2000_km_s", "vz_eme2000_km_s"],
    ["vx_km_s", "vy_km_s", "vz_km_s"],
    ["vx", "vy", "vz"],
]
SATELLITE_COLUMN_CANDIDATES = ["satellite", "sat_id", "satellite_id", "object_id"]
TIME_COLUMN_CANDIDATES = ["time_sec", "t_sec", "time_s", "elapsed_seconds"]
TIMESTAMP_COLUMN_CANDIDATES = ["timestamp", "epoch", "datetime", "date"]


# =============================================================================
# Small utilities
# =============================================================================


def log(message: str) -> None:
    print(message, flush=True)


def warn(message: str) -> None:
    print(f"[warning] {message}", file=sys.stderr, flush=True)


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return text or "unnamed"


def utc_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def json_dump(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.dtype):
        return str(value)
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def parse_bool_auto(value: str) -> Optional[bool]:
    value = str(value).strip().lower()
    if value in {"auto", "none", ""}:
        return None
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected auto/true/false")


def parse_float_auto(value: str) -> Optional[float]:
    if str(value).lower() == "auto":
        return None
    return float(value)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def choose_dtype(name: str, checkpoint_dtype: Optional[torch.dtype] = None) -> torch.dtype:
    if name == "auto":
        return checkpoint_dtype or torch.float32
    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
        "fp32": torch.float32,
        "fp64": torch.float64,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def import_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path.resolve()))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def nearest_indices(time_s: np.ndarray, targets: np.ndarray) -> np.ndarray:
    time_s = np.asarray(time_s)
    targets = np.asarray(targets)
    idx = np.searchsorted(time_s, targets)
    idx = np.clip(idx, 0, len(time_s) - 1)
    left = np.clip(idx - 1, 0, len(time_s) - 1)
    choose_left = np.abs(time_s[left] - targets) <= np.abs(time_s[idx] - targets)
    idx = np.where(choose_left, left, idx)
    return np.unique(idx.astype(int))


def stratified_time_indices(time_s: np.ndarray, n_anchors: Optional[int], horizon_s: float) -> np.ndarray:
    """Prefer dense early-window coverage while preserving the end point."""
    n = len(time_s)
    if n_anchors is None or n <= n_anchors:
        return np.arange(n, dtype=int)
    n_anchors = max(4, int(n_anchors))
    n_early = n_anchors // 2
    early_end = min(0.20 * horizon_s, float(time_s[-1]))
    early_targets = np.linspace(0.0, early_end, n_early, endpoint=False)
    late_targets = np.linspace(early_end, min(horizon_s, float(time_s[-1])), n_anchors - n_early)
    idx = nearest_indices(time_s, np.concatenate([early_targets, late_targets]))
    idx = np.unique(np.concatenate([[0], idx, [n - 1]])).astype(int)
    if len(idx) > n_anchors:
        idx = idx[np.linspace(0, len(idx) - 1, n_anchors).round().astype(int)]
    return idx


def hash_payload(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


# =============================================================================
# Automatic project discovery
# =============================================================================


@dataclass
class DiscoveryResult:
    root: Path
    dataset: Optional[Path]
    library: Optional[Path]
    checkpoint: Optional[Path]
    dataset_candidates: List[Tuple[str, float]] = field(default_factory=list)
    library_candidates: List[Tuple[str, float]] = field(default_factory=list)
    checkpoint_candidates: List[Tuple[str, float]] = field(default_factory=list)


def _filename_horizon_score(path: Path, horizon: Optional[float]) -> float:
    if horizon is None:
        return 0.0
    name = str(path).lower()
    k = int(round(horizon / 1000.0))
    score = 0.0
    if re.search(rf"(^|[^0-9]){k}k([^0-9]|$)", name):
        score += 30.0
    if str(int(horizon)) in name:
        score += 15.0
    return score


def _csv_looks_compatible(path: Path) -> bool:
    try:
        cols = set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return False
    has_pos = any(set(group).issubset(cols) for group in POSITION_COLUMN_SETS)
    has_vel = any(set(group).issubset(cols) for group in VELOCITY_COLUMN_SETS)
    has_sat = any(c in cols for c in SATELLITE_COLUMN_CANDIDATES)
    has_time = any(c in cols for c in TIME_COLUMN_CANDIDATES + TIMESTAMP_COLUMN_CANDIDATES)
    return has_pos and has_vel and has_sat and has_time


def discover_project(root: Path, horizon: Optional[float] = None) -> DiscoveryResult:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    dataset_scored: List[Tuple[Path, float]] = []
    for path in root.rglob("*.csv"):
        if not path.is_file() or not _csv_looks_compatible(path):
            continue
        name = path.name.lower()
        score = _filename_horizon_score(path, horizon)
        if "propagated" in name:
            score += 60
        if "performance" in name:
            score += 30
        if "leo" in str(path).lower():
            score += 10
        score += min(path.stat().st_mtime / 1e10, 0.2)
        dataset_scored.append((path, score))

    library_scored: List[Tuple[Path, float]] = []
    for path in root.rglob("*.py"):
        if not path.is_file() or path.resolve() == Path(__file__).resolve():
            continue
        name = path.name.lower()
        if any(bad in name for bad in ["train", "inference", "plot", "testbed", "diagnose"]):
            base_penalty = -20
        else:
            base_penalty = 0
        score = base_penalty + _filename_horizon_score(path, horizon)
        if "osc2mean" in name or "osctomean" in name:
            score += 70
        if "pinn_lib" in name:
            score += 60
        if "benchmark_kepler_main" in name:
            score += 45
        if "kepler" in name:
            score += 20
        if score <= 0:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")[:200000]
            if "class Net" in text and "GeneralistModel" in text:
                score += 25
            if "_kepler_j2_secular_propagate_torch" in text:
                score += 20
        except Exception:
            continue
        library_scored.append((path, score))

    checkpoint_scored: List[Tuple[Path, float]] = []
    for pattern in ("*.pth", "*.pt", "*.ckpt"):
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            name = path.name.lower()
            score = _filename_horizon_score(path, horizon)
            if "final" in name:
                score += 30
            if re.fullmatch(r"\d+\.pth", name):
                score += 10
            if "osc" in str(path).lower():
                score += 10
            score += min(path.stat().st_mtime / 1e9, 2.0)
            checkpoint_scored.append((path, score))

    dataset_scored.sort(key=lambda x: (x[1], x[0].stat().st_mtime), reverse=True)
    library_scored.sort(key=lambda x: (x[1], x[0].stat().st_mtime), reverse=True)
    checkpoint_scored.sort(key=lambda x: (x[1], x[0].stat().st_mtime), reverse=True)

    return DiscoveryResult(
        root=root,
        dataset=dataset_scored[0][0] if dataset_scored else None,
        library=library_scored[0][0] if library_scored else None,
        checkpoint=checkpoint_scored[0][0] if checkpoint_scored else None,
        dataset_candidates=[(str(p), float(s)) for p, s in dataset_scored[:10]],
        library_candidates=[(str(p), float(s)) for p, s in library_scored[:10]],
        checkpoint_candidates=[(str(p), float(s)) for p, s in checkpoint_scored[:10]],
    )


def resolve_path(explicit: Optional[str], discovered: Optional[Path], label: str) -> Optional[Path]:
    if explicit:
        if str(explicit).lower() in {"none", "off"}:
            return None
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
        return path
    return discovered


# =============================================================================
# Truth data representation and CSV repository
# =============================================================================


@dataclass
class OrbitArc:
    satellite: str
    time_s: np.ndarray
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    source: str

    @property
    def r0(self) -> np.ndarray:
        return self.position_m[0].copy()

    @property
    def v0(self) -> np.ndarray:
        return self.velocity_m_s[0].copy()

    def subset(self, idx: np.ndarray) -> "OrbitArc":
        return OrbitArc(
            satellite=self.satellite,
            time_s=self.time_s[idx],
            position_m=self.position_m[idx],
            velocity_m_s=self.velocity_m_s[idx],
            source=self.source,
        )


class CsvTruthRepository:
    def __init__(self, path: Path):
        self.path = path
        self.df = pd.read_csv(path)
        self.sat_col = self._find_one(SATELLITE_COLUMN_CANDIDATES, "satellite ID")
        self.pos_cols = self._find_group(POSITION_COLUMN_SETS, "position")
        self.vel_cols = self._find_group(VELOCITY_COLUMN_SETS, "velocity")
        self.timestamp_col = next((c for c in TIMESTAMP_COLUMN_CANDIDATES if c in self.df.columns), None)
        self.time_col = next((c for c in TIME_COLUMN_CANDIDATES if c in self.df.columns), None)
        if self.timestamp_col is None and self.time_col is None:
            raise KeyError("CSV must contain a timestamp or relative time column")
        self.df[self.sat_col] = self.df[self.sat_col].astype(str)
        if self.timestamp_col is not None:
            self.df[self.timestamp_col] = pd.to_datetime(self.df[self.timestamp_col])
        self.position_scale = self._infer_position_scale(self.pos_cols)
        self.velocity_scale = self._infer_velocity_scale(self.vel_cols)

    def _find_one(self, names: Sequence[str], label: str) -> str:
        for name in names:
            if name in self.df.columns:
                return name
        raise KeyError(f"Could not identify {label} column. Tried: {list(names)}")

    def _find_group(self, groups: Sequence[Sequence[str]], label: str) -> List[str]:
        cols = set(self.df.columns)
        for group in groups:
            if set(group).issubset(cols):
                return list(group)
        raise KeyError(f"Could not identify {label} columns")

    @staticmethod
    def _infer_position_scale(cols: Sequence[str]) -> float:
        joined = " ".join(cols).lower()
        return 1000.0 if "km" in joined else 1.0

    @staticmethod
    def _infer_velocity_scale(cols: Sequence[str]) -> float:
        joined = " ".join(cols).lower()
        return 1000.0 if "km" in joined else 1.0

    def satellite_ids(self) -> List[str]:
        return sorted(self.df[self.sat_col].unique().tolist())

    def get_arc(self, sat_id: str, horizon_s: Optional[float] = None) -> OrbitArc:
        sat = self.df[self.df[self.sat_col] == str(sat_id)].copy()
        if sat.empty:
            raise KeyError(f"Satellite {sat_id} not found")
        if self.timestamp_col is not None:
            sat = sat.sort_values(self.timestamp_col)
            t0 = sat[self.timestamp_col].iloc[0]
            time_s = (sat[self.timestamp_col] - t0).dt.total_seconds().to_numpy(np.float64)
        else:
            sat = sat.sort_values(self.time_col)
            raw = sat[self.time_col].to_numpy(np.float64)
            time_s = raw - raw[0]
        if horizon_s is not None:
            mask = time_s <= float(horizon_s) + 1e-9
            sat = sat.loc[mask]
            time_s = time_s[mask]
        if len(time_s) < 2:
            raise ValueError(f"Satellite {sat_id} has fewer than two points in requested horizon")
        r = sat[self.pos_cols].to_numpy(np.float64) * self.position_scale
        v = sat[self.vel_cols].to_numpy(np.float64) * self.velocity_scale
        return OrbitArc(str(sat_id), time_s, r, v, f"csv:{self.path}")

    def infer_horizon(self) -> float:
        durations = []
        for sat_id in self.satellite_ids()[: min(50, len(self.satellite_ids()))]:
            try:
                durations.append(float(self.get_arc(sat_id).time_s[-1]))
            except Exception:
                pass
        if not durations:
            raise ValueError("Could not infer horizon from CSV")
        return float(np.median(durations))

    def load_arcs(
        self,
        horizon_s: float,
        satellites: Optional[Sequence[str]] = None,
        max_satellites: Optional[int] = None,
        seed: int = 42,
    ) -> List[OrbitArc]:
        ids = self.satellite_ids() if satellites is None else [str(x) for x in satellites]
        valid = []
        for sat_id in ids:
            try:
                arc = self.get_arc(sat_id, horizon_s)
                if arc.time_s[-1] >= 0.95 * horizon_s:
                    valid.append(arc)
            except Exception:
                continue
        if max_satellites is not None and len(valid) > max_satellites:
            rng = np.random.default_rng(seed)
            idx = sorted(rng.choice(len(valid), size=max_satellites, replace=False).tolist())
            valid = [valid[i] for i in idx]
        if not valid:
            raise ValueError(f"No CSV satellites contain enough data for horizon {horizon_s}s")
        return valid


# =============================================================================
# Orbital mechanics helpers and fallback truth generation
# =============================================================================


def central_j2_acceleration_np(position_m: np.ndarray) -> np.ndarray:
    r = np.asarray(position_m, dtype=np.float64)
    original_shape = r.shape
    r = np.atleast_2d(r)
    rn2 = np.sum(r * r, axis=1, keepdims=True)
    rn = np.sqrt(np.maximum(rn2, 1e-24))
    inv_r3 = 1.0 / (rn2 * rn)
    a_c = -MU_EARTH * r * inv_r3
    z2_r2 = (r[:, 2:3] ** 2) / np.maximum(rn2, 1e-24)
    factor = 1.5 * J2_EARTH * MU_EARTH * R_EARTH**2 / np.maximum(rn**5, 1e-24)
    a_j2 = np.column_stack([
        factor[:, 0] * r[:, 0] * (5.0 * z2_r2[:, 0] - 1.0),
        factor[:, 0] * r[:, 1] * (5.0 * z2_r2[:, 0] - 1.0),
        factor[:, 0] * r[:, 2] * (5.0 * z2_r2[:, 0] - 3.0),
    ])
    out = a_c + a_j2
    return out[0] if len(original_shape) == 1 else out


def dop853_rhs(_t: float, state: np.ndarray) -> np.ndarray:
    return np.concatenate([state[3:], central_j2_acceleration_np(state[:3])])


def propagate_dop853(r0: np.ndarray, v0: np.ndarray, time_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sol = solve_ivp(
        dop853_rhs,
        (float(time_s[0]), float(time_s[-1])),
        np.concatenate([r0, v0]),
        t_eval=np.asarray(time_s, dtype=np.float64),
        method="DOP853",
        rtol=1e-11,
        atol=1e-9,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:3].T, sol.y[3:].T


def state_from_elements(a: float, e: float, inc: float, raan: float, argp: float, nu: float) -> Tuple[np.ndarray, np.ndarray]:
    p = a * (1.0 - e * e)
    rmag = p / (1.0 + e * np.cos(nu))
    rp = np.array([rmag * np.cos(nu), rmag * np.sin(nu), 0.0])
    vp = np.sqrt(MU_EARTH / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)
    rotation = (
        np.array([[cO, -sO, 0], [sO, cO, 0], [0, 0, 1]])
        @ np.array([[1, 0, 0], [0, ci, -si], [0, si, ci]])
        @ np.array([[cw, -sw, 0], [sw, cw, 0], [0, 0, 1]])
    )
    return rotation @ rp, rotation @ vp


def _rotate_vector(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / max(np.linalg.norm(axis), 1e-15)
    c, s = np.cos(angle), np.sin(angle)
    return vector * c + np.cross(axis, vector) * s + axis * np.dot(axis, vector) * (1.0 - c)


def _internal_delta_a(r: np.ndarray, a: float, e: float, sin2_i: float) -> float:
    rn = np.linalg.norm(r)
    z2 = (r[2] / rn) ** 2
    return (J2_EARTH * R_EARTH**2 / a) * (
        (a / rn) ** 3 * (1.0 - 3.0 * z2)
        - (1.0 - 1.5 * sin2_i) * (1.0 - e * e) ** -1.5
    )


def internal_osc2mean_baseline(r0: np.ndarray, v0: np.ndarray, time_s: np.ndarray, use_osc2mean: bool = True) -> np.ndarray:
    """Fallback approximation. Production-library baseline is preferred."""
    eps = 1e-12
    rn = np.linalg.norm(r0)
    h = np.cross(r0, v0)
    hn = np.linalg.norm(h)
    a = 1.0 / max(2.0 / rn - np.dot(v0, v0) / MU_EARTH, eps)
    evec = np.cross(v0, h) / MU_EARTH - r0 / rn
    e = max(np.linalg.norm(evec), eps)
    inc = np.arccos(np.clip(h[2] / hn, -1.0, 1.0))
    sin2_i = np.sin(inc) ** 2
    ehat = evec / e if e > 1e-9 else r0 / rn
    hhat = h / hn
    qhat = np.cross(hhat, ehat)
    cos_nu0 = np.clip(np.dot(r0, ehat) / rn, -1.0, 1.0)
    nu0 = np.arccos(cos_nu0)
    if np.dot(r0, v0) < 0:
        nu0 = 2.0 * np.pi - nu0
    E0 = 2.0 * np.arctan2(
        np.sqrt(max(1.0 - e, eps)) * np.sin(nu0 / 2.0),
        np.sqrt(1.0 + e) * np.cos(nu0 / 2.0),
    ) % (2.0 * np.pi)
    M0 = E0 - e * np.sin(E0)
    a_rate = a - _internal_delta_a(r0, a, e, sin2_i) if use_osc2mean else a
    n = np.sqrt(MU_EARTH / a_rate**3)
    p_rate = a_rate * (1.0 - e * e)
    factor = 1.5 * n * J2_EARTH * R_EARTH**2 / p_rate**2
    d_raan = -factor * np.cos(inc)
    d_argp = factor * (2.0 - 2.5 * sin2_i)
    n_eff = n + factor * np.sqrt(max(1.0 - e * e, 0.0)) * (1.0 - 1.5 * sin2_i)
    M = (M0 + n_eff * time_s) % (2.0 * np.pi)
    E = M.copy()
    for _ in range(14):
        E -= (E - e * np.sin(E) - M) / np.maximum(1.0 - e * np.cos(E), eps)
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(E / 2.0),
        np.sqrt(max(1.0 - e, eps)) * np.cos(E / 2.0),
    )
    rmag = a * (1.0 - e * np.cos(E))
    zhat = np.array([0.0, 0.0, 1.0])
    out = np.zeros((len(time_s), 3), dtype=np.float64)
    for k, tk in enumerate(time_s):
        e2 = _rotate_vector(ehat, hhat, d_argp * tk)
        q2 = _rotate_vector(qhat, hhat, d_argp * tk)
        e2 = _rotate_vector(e2, zhat, d_raan * tk)
        q2 = _rotate_vector(q2, zhat, d_raan * tk)
        out[k] = rmag[k] * (np.cos(nu[k]) * e2 + np.sin(nu[k]) * q2)
    return out


def try_propagate_orekit(r0: np.ndarray, v0: np.ndarray, time_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Numerical Orekit J2 truth. Imported lazily; raises ImportError when unavailable."""
    try:
        import orekit
        from orekit.pyhelpers import setup_orekit_curdir
        from org.orekit.time import AbsoluteDate, TimeScalesFactory
        from org.orekit.frames import FramesFactory
        from org.orekit.utils import PVCoordinates, IERSConventions
        from org.hipparchus.geometry.euclidean.threed import Vector3D
        from org.orekit.orbits import CartesianOrbit, OrbitType
        from org.orekit.propagation import SpacecraftState
        from org.orekit.propagation.numerical import NumericalPropagator
        from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
        from org.orekit.forces.gravity import J2OnlyPerturbation
    except Exception as exc:
        raise ImportError(f"Orekit Python/JVM bindings are unavailable: {exc}") from exc

    try:
        orekit.initVM()
    except Exception:
        pass
    setup_orekit_curdir(from_pip_library=True)
    utc = TimeScalesFactory.getUTC()
    epoch = AbsoluteDate(2026, 1, 1, 0, 0, 0.0, utc)
    inertial = FramesFactory.getEME2000()
    pv = PVCoordinates(Vector3D(*map(float, r0)), Vector3D(*map(float, v0)))
    orbit = CartesianOrbit(pv, inertial, epoch, MU_EARTH)
    tolerances = NumericalPropagator.tolerances(1e-5, orbit, OrbitType.CARTESIAN)
    tol0 = orekit.JArray("double").cast_(tolerances[0])
    tol1 = orekit.JArray("double").cast_(tolerances[1])
    integrator = DormandPrince853Integrator(0.1, 60.0, tol0, tol1)
    integrator.setInitialStepSize(10.0)
    propagator = NumericalPropagator(integrator)
    propagator.setOrbitType(OrbitType.CARTESIAN)
    propagator.setInitialState(SpacecraftState(orbit))
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    propagator.addForceModel(J2OnlyPerturbation(MU_EARTH, R_EARTH, J2_EARTH, itrf))
    pos = np.zeros((len(time_s), 3), dtype=np.float64)
    vel = np.zeros((len(time_s), 3), dtype=np.float64)
    for i, dt in enumerate(time_s):
        state = propagator.propagate(epoch.shiftedBy(float(dt)))
        pv_i = state.getPVCoordinates(inertial)
        pos[i] = pv_i.getPosition().toArray()
        vel[i] = pv_i.getVelocity().toArray()
    return pos, vel


def generate_synthetic_arcs(
    count: int,
    horizon_s: float,
    anchors: int,
    seed: int,
    backend: str,
    cache_dir: Path,
    alt_range_km: Tuple[float, float] = (450.0, 1200.0),
    ecc_range: Tuple[float, float] = (0.0005, 0.015),
    inclination_choices_deg: Sequence[float] = (5, 20, 35, 51.6, 63.4, 75, 90, 97),
) -> List[OrbitArc]:
    payload = {
        "count": count,
        "horizon": horizon_s,
        "anchors": anchors,
        "seed": seed,
        "backend": backend,
        "alt_range_km": alt_range_km,
        "ecc_range": ecc_range,
        "inc": list(inclination_choices_deg),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"synthetic_truth_{hash_payload(payload)}.npz"
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=False)
        arcs = []
        for i in range(len(d["r0"])):
            arcs.append(OrbitArc(
                satellite=f"synthetic_{i:05d}",
                time_s=d["time_s"][i],
                position_m=d["position_m"][i],
                velocity_m_s=d["velocity_m_s"][i],
                source=str(d["source"][i]),
            ))
        log(f"[truth] loaded cached synthetic truth: {cache_path}")
        return arcs

    rng = np.random.default_rng(seed)
    n_early = anchors // 2
    early = np.linspace(0.0, 0.20 * horizon_s, n_early, endpoint=False)
    late = np.linspace(0.20 * horizon_s, horizon_s, anchors - n_early)
    time_grid = np.unique(np.concatenate([early, late])).astype(np.float64)
    if time_grid[0] != 0:
        time_grid = np.concatenate([[0.0], time_grid])

    selected_backend = backend
    if backend == "auto":
        try:
            import orekit  # noqa: F401
            selected_backend = "orekit"
        except Exception:
            selected_backend = "dop853"
    log(f"[truth] generating {count} synthetic satellites with {selected_backend}")

    positions, velocities, sources = [], [], []
    for i in range(count):
        a = R_EARTH + rng.uniform(*alt_range_km) * 1000.0
        e = rng.uniform(*ecc_range)
        inc = np.radians(rng.choice(inclination_choices_deg) + rng.uniform(-3.0, 3.0))
        r0, v0 = state_from_elements(
            a, e, inc,
            rng.uniform(0, 2 * np.pi),
            rng.uniform(0, 2 * np.pi),
            rng.uniform(0, 2 * np.pi),
        )
        if selected_backend == "orekit":
            try:
                r, v = try_propagate_orekit(r0, v0, time_grid)
                source = "orekit:J2OnlyPerturbation"
            except Exception as exc:
                if backend == "orekit":
                    raise
                warn(f"Orekit failed ({exc}); falling back to DOP853")
                selected_backend = "dop853"
                r, v = propagate_dop853(r0, v0, time_grid)
                source = "scipy:DOP853 central+J2"
        elif selected_backend == "dop853":
            r, v = propagate_dop853(r0, v0, time_grid)
            source = "scipy:DOP853 central+J2"
        else:
            raise ValueError(f"Unsupported synthetic backend: {selected_backend}")
        positions.append(r)
        velocities.append(v)
        sources.append(source)
        if (i + 1) % 25 == 0 or i + 1 == count:
            log(f"  generated {i + 1}/{count}")

    np.savez_compressed(
        cache_path,
        r0=np.array([x[0] for x in positions]),
        time_s=np.array([time_grid for _ in positions]),
        position_m=np.array(positions),
        velocity_m_s=np.array(velocities),
        source=np.array(sources),
    )
    return [
        OrbitArc(f"synthetic_{i:05d}", time_grid.copy(), positions[i], velocities[i], sources[i])
        for i in range(count)
    ]


# =============================================================================
# Checkpoint and production-library adapter
# =============================================================================


def load_state_dict(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        obj = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location=device)
    state = None
    if isinstance(obj, dict):
        for key in ("model_state", "state_dict", "model_state_dict", "net_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                state = obj[key]
                break
        if state is None and obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            state = obj
    if state is None:
        raise ValueError("Checkpoint is not a raw state_dict and contains no recognized state_dict key")
    cleaned = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "model.net.", "net."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def infer_layers_from_state_dict(state: Mapping[str, torch.Tensor]) -> List[int]:
    found = []
    pattern = re.compile(r"^linear\.(\d+)\.weight$")
    for key, tensor in state.items():
        match = pattern.match(key)
        if match:
            found.append((int(match.group(1)), tensor))
    if not found:
        # Sequential fallback: net.0.weight, net.2.weight, ... after prefix cleaning.
        pattern2 = re.compile(r"^(?:net\.)?(\d+)\.weight$")
        for key, tensor in state.items():
            match = pattern2.match(key)
            if match:
                found.append((int(match.group(1)), tensor))
    if not found:
        raise KeyError("Could not infer architecture from checkpoint weights")
    found.sort(key=lambda item: item[0])
    layers = [int(found[0][1].shape[1])] + [int(t.shape[0]) for _, t in found]
    return layers


class GenericNet(nn.Module):
    """Same architecture as the user's Net: ModuleList linear, Tanh hidden, linear output."""
    def __init__(self, layers: Sequence[int]):
        super().__init__()
        self.layers = list(map(int, layers))
        self.linear = nn.ModuleList([
            nn.Linear(self.layers[i], self.layers[i + 1])
            for i in range(len(self.layers) - 1)
        ])
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.linear[:-1]:
            x = torch.tanh(layer(x))
        return self.linear[-1](x)


@dataclass
class NetworkTrace:
    inputs: np.ndarray
    preactivations: List[np.ndarray]
    activations: List[np.ndarray]
    output: np.ndarray


class NumpyMLP:
    def __init__(self, weights: Sequence[np.ndarray], biases: Sequence[np.ndarray]):
        self.weights = [np.asarray(x).copy() for x in weights]
        self.biases = [np.asarray(x).copy() for x in biases]
        self.dtype = self.weights[0].dtype

    @classmethod
    def from_torch(cls, net: nn.Module) -> "NumpyMLP":
        if not hasattr(net, "linear"):
            raise AttributeError("Network must expose .linear ModuleList")
        return cls(
            [layer.weight.detach().cpu().numpy() for layer in net.linear],
            [layer.bias.detach().cpu().numpy() for layer in net.linear],
        )

    def forward(self, inputs: np.ndarray, capture: bool = True) -> NetworkTrace:
        h = np.asarray(inputs, dtype=self.dtype)
        pre, acts = [], []
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            z = h @ weight.T + bias
            h = np.tanh(z)
            if capture:
                pre.append(z.copy())
                acts.append(h.copy())
        out = h @ self.weights[-1].T + self.biases[-1]
        return NetworkTrace(np.asarray(inputs), pre, acts, out)


class ProductionAdapter:
    """Keeps the user's baseline and checkpoint forward path when those files exist."""
    def __init__(
        self,
        library_path: Optional[Path],
        checkpoint_path: Optional[Path],
        device: torch.device,
    ):
        self.library_path = library_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.lib = None
        self.state_dict = None
        self.layers: Optional[List[int]] = None
        self.net: Optional[nn.Module] = None
        self.numpy_net: Optional[NumpyMLP] = None
        self.dtype = torch.float32
        self.model_wrapper = None
        if library_path is not None:
            self.lib = import_module_from_path("unified_pinn_original_library", library_path)
        if checkpoint_path is not None:
            self.state_dict = load_state_dict(checkpoint_path, device)
            self.layers = infer_layers_from_state_dict(self.state_dict)
            first_weight = next(v for k, v in self.state_dict.items() if k.endswith("weight"))
            self.dtype = first_weight.dtype
            net_cls = self.lib.Net if self.lib is not None and hasattr(self.lib, "Net") else GenericNet
            self.net = net_cls(self.layers).to(device=device, dtype=self.dtype)
            try:
                self.net.load_state_dict(self.state_dict, strict=True)
            except RuntimeError:
                # If checkpoint came from Sequential rapid model, map ordered weights.
                self._load_ordered_state(self.net, self.state_dict)
            self.net.eval()
            self.numpy_net = NumpyMLP.from_torch(self.net)
            if self.lib is not None and hasattr(self.lib, "GeneralistModel"):
                self.model_wrapper = self.lib.GeneralistModel(self.net, W_DATA=1.0, W_PDE=0.0)

    @staticmethod
    def _load_ordered_state(net: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
        weights = [v for k, v in sorted(state.items()) if k.endswith("weight")]
        biases = [v for k, v in sorted(state.items()) if k.endswith("bias")]
        if len(weights) != len(net.linear) or len(biases) != len(net.linear):
            raise RuntimeError("Could not map checkpoint layers to GenericNet")
        with torch.no_grad():
            for layer, w, b in zip(net.linear, weights, biases):
                layer.weight.copy_(w.to(layer.weight))
                layer.bias.copy_(b.to(layer.bias))

    @property
    def has_checkpoint(self) -> bool:
        return self.net is not None

    @property
    def has_production_library(self) -> bool:
        return self.lib is not None

    def set_flags(
        self,
        use_j2_baseline: bool,
        use_osc2mean: bool,
        use_j2_physics: bool,
        use_lat_features: bool,
        gate_power: float,
    ) -> None:
        if self.lib is None:
            return
        for name, value in {
            "USE_J2_BASELINE": bool(use_j2_baseline),
            "USE_OSC2MEAN": bool(use_osc2mean),
            "USE_J2": bool(use_j2_physics),
            "USE_LAT_FEATURES": bool(use_lat_features),
            "GATE_POWER": float(gate_power),
        }.items():
            if hasattr(self.lib, name):
                setattr(self.lib, name, value)

    def baseline(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        time_s: np.ndarray,
        use_j2_baseline: bool = True,
        use_osc2mean: bool = True,
    ) -> np.ndarray:
        if self.lib is None:
            if use_j2_baseline:
                return internal_osc2mean_baseline(r0, v0, time_s, use_osc2mean)
            return internal_kepler_baseline(r0, v0, time_s)
        self.set_flags(use_j2_baseline, use_osc2mean, True, False, 2.0)
        n = len(time_s)
        r0_t = torch.tensor(r0, device=self.device, dtype=self.dtype).reshape(1, 3).repeat(n, 1)
        v0_t = torch.tensor(v0, device=self.device, dtype=self.dtype).reshape(1, 3).repeat(n, 1)
        t_t = torch.tensor(time_s, device=self.device, dtype=self.dtype).reshape(-1, 1)
        if use_j2_baseline and hasattr(self.lib, "_kepler_j2_secular_propagate_torch"):
            propagator = self.lib._kepler_j2_secular_propagate_torch
        elif hasattr(self.lib, "_kepler_propagate_torch"):
            propagator = self.lib._kepler_propagate_torch
        else:
            warn("Production library has no recognized propagator; using internal baseline")
            return internal_osc2mean_baseline(r0, v0, time_s, use_osc2mean)
        with torch.no_grad():
            result = propagator(r0_t, v0_t, t_t)
        return result.detach().cpu().numpy().astype(np.float64)

    def physics_acceleration(self, positions: np.ndarray, use_j2: bool = True) -> np.ndarray:
        if self.lib is None or not hasattr(self.lib, "earth_gravity_acc_torch"):
            return central_j2_acceleration_np(positions) if use_j2 else central_acceleration_np(positions)
        if hasattr(self.lib, "USE_J2"):
            self.lib.USE_J2 = bool(use_j2)
        r = torch.tensor(positions, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            out = self.lib.earth_gravity_acc_torch(r)
        return out.detach().cpu().numpy().astype(np.float64)

    def exact_reference_positions(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        time_s: np.ndarray,
        window_s: float,
        use_j2_baseline: bool,
        use_osc2mean: bool,
        use_lat_features: bool,
        gate_power: float,
    ) -> Optional[np.ndarray]:
        if self.model_wrapper is None or not hasattr(self.model_wrapper, "_forward_scaled_position"):
            return None
        self.set_flags(use_j2_baseline, use_osc2mean, True, use_lat_features, gate_power)
        n = len(time_s)
        r0_t = torch.tensor(r0, device=self.device, dtype=self.dtype).reshape(1, 3).repeat(n, 1)
        v0_t = torch.tensor(v0, device=self.device, dtype=self.dtype).reshape(1, 3).repeat(n, 1)
        t_t = torch.tensor(time_s, device=self.device, dtype=self.dtype).reshape(-1, 1)
        tseg = torch.full_like(t_t, float(window_s))
        final_scaled, _, R0, _, _ = self.model_wrapper._forward_scaled_position(
            t_t, r0_t, v0_t, Tseg_batch=tseg
        )
        return (final_scaled * R0).detach().cpu().numpy().astype(np.float64)


def central_acceleration_np(position_m: np.ndarray) -> np.ndarray:
    r = np.asarray(position_m, dtype=np.float64)
    rn = np.linalg.norm(r, axis=-1, keepdims=True)
    return -MU_EARTH * r / np.maximum(rn**3, 1e-24)


def internal_kepler_baseline(r0: np.ndarray, v0: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """DOP853 central-gravity fallback, used only if production Kepler is absent."""
    def rhs(_t, state):
        return np.concatenate([state[3:], central_acceleration_np(state[:3])])
    sol = solve_ivp(
        rhs,
        (0.0, float(time_s[-1])),
        np.concatenate([r0, v0]),
        t_eval=time_s,
        method="DOP853",
        rtol=1e-11,
        atol=1e-9,
    )
    return sol.y[:3].T


# =============================================================================
# Feature engineering and gate definitions
# =============================================================================


@dataclass
class OrbitalFeatures:
    R0: float
    V0: float
    T0: float
    Torb: float
    eccentricity: float
    cos_i: float
    sin2_i: float
    d_raan_dt: float
    d_argp_dt: float


def orbital_features(r0: np.ndarray, v0: np.ndarray) -> OrbitalFeatures:
    R0 = float(np.linalg.norm(r0))
    V0 = float(np.linalg.norm(v0))
    T0 = R0 / max(V0, 1e-12)
    h = np.cross(r0, v0)
    hn = max(np.linalg.norm(h), 1e-12)
    cos_i = float(np.clip(h[2] / hn, -1.0, 1.0))
    sin2_i = max(0.0, 1.0 - cos_i**2)
    evec = np.cross(v0, h) / MU_EARTH - r0 / R0
    e = float(np.linalg.norm(evec))
    inv_a = 2.0 / R0 - V0**2 / MU_EARTH
    a = 1.0 / max(inv_a, 1e-15)
    Torb = 2.0 * np.pi * np.sqrt(a**3 / MU_EARTH)
    p = a * max(1.0 - e**2, 1e-12)
    n = np.sqrt(MU_EARTH / a**3)
    factor = 1.5 * n * J2_EARTH * R_EARTH**2 / p**2
    return OrbitalFeatures(
        R0=R0,
        V0=V0,
        T0=T0,
        Torb=float(Torb),
        eccentricity=e,
        cos_i=cos_i,
        sin2_i=sin2_i,
        d_raan_dt=float(-factor * cos_i),
        d_argp_dt=float(factor * (2.0 - 2.5 * sin2_i)),
    )


def argument_of_latitude_features(r: np.ndarray, r0: np.ndarray, v0: np.ndarray) -> np.ndarray:
    """Returns sin u, cos u, sin 2u, cos 2u for each position."""
    h = np.cross(r0, v0)
    hn = max(np.linalg.norm(h), 1e-12)
    hhat = h / hn
    node = np.array([-h[1], h[0], 0.0])
    nnorm = np.linalg.norm(node)
    if nnorm < 1e-12:
        nhat = r0 / max(np.linalg.norm(r0), 1e-12)
    else:
        nhat = node / nnorm
    shat = np.cross(hhat, nhat)
    rhat = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-12)
    cos_u = np.clip(rhat @ nhat, -1.0, 1.0)
    sin_u = np.clip(rhat @ shat, -1.0, 1.0)
    return np.column_stack([sin_u, cos_u, 2.0 * sin_u * cos_u, cos_u**2 - sin_u**2])


def j2_acceleration_only_np(position_m: np.ndarray) -> np.ndarray:
    return central_j2_acceleration_np(position_m) - central_acceleration_np(position_m)


def build_feature_array(
    time_s: np.ndarray,
    r0: np.ndarray,
    v0: np.ndarray,
    baseline_m: np.ndarray,
    horizon_s: float,
    time_feature: str,
    feature_tokens: Sequence[str],
    v0_scale: float = V0_SCALE_DEFAULT,
) -> Tuple[np.ndarray, List[str], OrbitalFeatures]:
    of = orbital_features(r0, v0)
    n = len(time_s)
    t = np.asarray(time_s, dtype=np.float64).reshape(-1, 1)
    if time_feature == "t_over_T0":
        tfeat = t / of.T0
        tname = "t/T0"
    elif time_feature in {"t_over_window", "tau"}:
        tfeat = t / horizon_s
        tname = "t/Twindow"
    elif time_feature == "t_over_T0_clipped":
        tfeat = np.clip(t / of.T0, 0.0, 1.0)
        tname = "clip(t/T0)"
    else:
        raise ValueError(f"Unsupported time_feature={time_feature}")
    columns = [
        tfeat,
        np.repeat((r0 / of.R0).reshape(1, 3), n, axis=0),
        np.repeat((v0 / of.V0).reshape(1, 3), n, axis=0),
        np.full((n, 1), of.R0 / R_EARTH),
        np.full((n, 1), of.V0 / v0_scale),
    ]
    names = [
        tname,
        "x0/R0", "y0/R0", "z0/R0",
        "vx0/V0", "vy0/V0", "vz0/V0",
        "R0/Re", "V0/Vscale",
    ]
    tokens = set(feature_tokens)
    if "lat" in tokens:
        rn = np.maximum(np.linalg.norm(baseline_m, axis=1, keepdims=True), 1e-12)
        sl = baseline_m[:, 2:3] / rn
        columns.extend([sl, sl**2])
        names.extend(["z/r", "(z/r)^2"])
    if "inc" in tokens:
        columns.extend([
            np.full((n, 1), of.cos_i),
            np.full((n, 1), of.sin2_i),
        ])
        names.extend(["cos(i)", "sin^2(i)"])
    if "ecc" in tokens:
        columns.append(np.full((n, 1), of.eccentricity))
        names.append("eccentricity")
    if "acc" in tokens:
        a_c = central_acceleration_np(baseline_m)
        a_j = j2_acceleration_only_np(baseline_m)
        ac_mag = np.maximum(np.linalg.norm(a_c, axis=1, keepdims=True), 1e-12)
        columns.extend([a_j / ac_mag, np.linalg.norm(a_j, axis=1, keepdims=True) / ac_mag])
        names.extend(["aJ2_x/|ac|", "aJ2_y/|ac|", "aJ2_z/|ac|", "|aJ2|/|ac|"])
    if "phase" in tokens:
        columns.append(argument_of_latitude_features(baseline_m, r0, v0))
        names.extend(["sin(u)", "cos(u)", "sin(2u)", "cos(2u)"])
    if "rates" in tokens:
        columns.extend([
            np.full((n, 1), of.d_raan_dt * of.T0),
            np.full((n, 1), of.d_argp_dt * of.T0),
        ])
        names.extend(["Omega_dot*T0", "omega_dot*T0"])
    values = np.concatenate(columns, axis=1).astype(np.float64)
    return values, names, of


def compute_gate_np(
    time_s: np.ndarray,
    gate_type: str,
    gate_power: float,
    horizon_s: float,
    T0: float,
    Torb: float,
    tau_div: float = 3.0,
    local_scale_T0: float = 1.0,
    legacy_t_scaled_max: Optional[float] = None,
) -> np.ndarray:
    t = np.asarray(time_s, dtype=np.float64)
    if gate_type in {"power", "legacy"}:
        gate = (t / horizon_s) ** gate_power
    elif gate_type == "legacy_global_power":
        tmax = legacy_t_scaled_max or round(horizon_s / 907.0, 1)
        gate = ((t / T0) / tmax) ** gate_power
    elif gate_type == "orbit":
        gate = np.clip(t / Torb, 0.0, 1.0) ** gate_power
    elif gate_type == "tanh":
        gate = np.tanh(t / max(Torb / tau_div, 1e-12)) ** 2
    elif gate_type == "local_exponential":
        gate = 1.0 - np.exp(-((t / max(local_scale_T0 * T0, 1e-12)) ** 2))
    elif gate_type == "none":
        gate = np.ones_like(t)
    else:
        raise ValueError(f"Unsupported gate_type={gate_type}")
    return gate.reshape(-1, 1)


def compute_gate_torch(
    time_s: torch.Tensor,
    gate_type: str,
    gate_power: float,
    horizon_s: float,
    T0: torch.Tensor,
    Torb: torch.Tensor,
    tau_div: float = 3.0,
    local_scale_T0: float = 1.0,
) -> torch.Tensor:
    if gate_type in {"power", "legacy"}:
        return (time_s / horizon_s).clamp_min(0.0) ** gate_power
    if gate_type == "orbit":
        return (time_s / Torb).clamp(0.0, 1.0) ** gate_power
    if gate_type == "tanh":
        return torch.tanh(time_s / (Torb / tau_div).clamp_min(1e-12)) ** 2
    if gate_type == "local_exponential":
        s = time_s / (local_scale_T0 * T0).clamp_min(1e-12)
        return 1.0 - torch.exp(-(s**2))
    if gate_type == "none":
        return torch.ones_like(time_s)
    raise ValueError(gate_type)


# =============================================================================
# Metrics
# =============================================================================


@dataclass
class Thresholds:
    mean_error_km_max: float = 2.0
    rmse_error_km_max: float = 2.5
    p95_error_km_max: float = 3.0
    max_error_km_max: float = 5.0
    final_error_km_max: float = 3.0
    early_mean_error_km_max: float = 2.5
    max_dead_gradient_fraction_max: float = 0.25
    min_active_gradient_fraction_min: float = 0.60
    baseline_gap_closed_pct_min: float = 0.0
    exactness_max_position_m_max: float = 2.0
    satellite_pass_rate_min: float = 0.90


def position_metrics(pred: np.ndarray, truth: np.ndarray, baseline: Optional[np.ndarray] = None, time_s: Optional[np.ndarray] = None, horizon_s: Optional[float] = None) -> Dict[str, float]:
    err = np.linalg.norm(pred - truth, axis=1) / 1000.0
    metrics = {
        "mean_error_km": float(np.mean(err)),
        "median_error_km": float(np.median(err)),
        "rmse_error_km": float(np.sqrt(np.mean(err**2))),
        "p90_error_km": float(np.quantile(err, 0.90)),
        "p95_error_km": float(np.quantile(err, 0.95)),
        "p99_error_km": float(np.quantile(err, 0.99)),
        "max_error_km": float(np.max(err)),
        "final_error_km": float(err[-1]),
        "max_error_time_s": float(time_s[np.argmax(err)]) if time_s is not None else float(np.argmax(err)),
    }
    if time_s is not None and horizon_s is not None:
        early = np.asarray(time_s) <= 0.25 * horizon_s
        metrics["early_mean_error_km"] = float(np.mean(err[early])) if np.any(early) else float("nan")
    if baseline is not None:
        berr = np.linalg.norm(baseline - truth, axis=1) / 1000.0
        bmean = float(np.mean(berr))
        metrics.update({
            "baseline_mean_error_km": bmean,
            "baseline_p95_error_km": float(np.quantile(berr, 0.95)),
            "baseline_max_error_km": float(np.max(berr)),
            "baseline_final_error_km": float(berr[-1]),
            "baseline_gap_closed_pct": float(100.0 * (1.0 - metrics["mean_error_km"] / max(bmean, 1e-12))),
        })
    return metrics


def rsw_components(pred: np.ndarray, truth: np.ndarray, truth_velocity: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    error = pred - truth
    rhat = truth / np.maximum(np.linalg.norm(truth, axis=1, keepdims=True), 1e-12)
    h = np.cross(truth, truth_velocity)
    what = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
    shat = np.cross(what, rhat)
    comp = np.column_stack([
        np.sum(error * rhat, axis=1),
        np.sum(error * shat, axis=1),
        np.sum(error * what, axis=1),
    ]) / 1000.0
    out = {}
    for i, name in enumerate(["radial", "along_track", "cross_track"]):
        vals = comp[:, i]
        out[f"{name}_mae_km"] = float(np.mean(np.abs(vals)))
        out[f"{name}_rmse_km"] = float(np.sqrt(np.mean(vals**2)))
        out[f"{name}_max_abs_km"] = float(np.max(np.abs(vals)))
        out[f"{name}_final_km"] = float(vals[-1])
    return comp, out


def activation_metrics(trace: NetworkTrace) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    rows = []
    for i, (z, a) in enumerate(zip(trace.preactivations, trace.activations), start=1):
        derivative = 1.0 - a**2
        active_time = np.mean(derivative >= 0.05, axis=1)
        dead_time = np.mean(derivative < 1e-3, axis=1)
        rows.append({
            "layer": i,
            "neurons": int(a.shape[1]),
            "mean_abs_z": float(np.mean(np.abs(z))),
            "p95_abs_z": float(np.quantile(np.abs(z), 0.95)),
            "saturation_fraction_abs_activation_gt_0p99": float(np.mean(np.abs(a) > 0.99)),
            "mean_tanh_derivative": float(np.mean(derivative)),
            "minimum_time_active_fraction": float(np.min(active_time)),
            "mean_time_active_fraction": float(np.mean(active_time)),
            "maximum_time_dead_gradient_fraction": float(np.max(dead_time)),
            "mean_time_dead_gradient_fraction": float(np.mean(dead_time)),
        })
    summary = {
        "worst_layer_max_dead_gradient_fraction": float(max(r["maximum_time_dead_gradient_fraction"] for r in rows)),
        "minimum_layer_active_fraction": float(min(r["minimum_time_active_fraction"] for r in rows)),
        "minimum_layer_mean_tanh_derivative": float(min(r["mean_tanh_derivative"] for r in rows)),
    }
    return rows, summary


def correction_metrics(raw_scaled: np.ndarray, gate: np.ndarray, R0: float, time_s: np.ndarray) -> Dict[str, float]:
    raw_km = raw_scaled * R0 / 1000.0
    gated_km = raw_km * gate
    raw_norm = np.linalg.norm(raw_km, axis=1)
    gated_norm = np.linalg.norm(gated_km, axis=1)
    def at(target, values):
        return float(values[np.argmin(np.abs(time_s - target))])
    return {
        "raw_correction_mean_km": float(np.mean(raw_norm)),
        "raw_correction_max_km": float(np.max(raw_norm)),
        "gated_correction_mean_km": float(np.mean(gated_norm)),
        "gated_correction_max_km": float(np.max(gated_norm)),
        "gate_mean": float(np.mean(gate)),
        "gate_at_1000s": at(1000.0, gate[:, 0]),
        "gate_at_2000s": at(2000.0, gate[:, 0]),
        "gate_at_5000s": at(5000.0, gate[:, 0]),
        "gate_at_10000s": at(10000.0, gate[:, 0]),
        "gated_correction_at_5000s_km": at(5000.0, gated_norm),
    }


def physics_residual_metrics(predicted_m: np.ndarray, time_s: np.ndarray, target_acc: np.ndarray) -> Dict[str, float]:
    if len(time_s) < 7:
        return {"physics_residual_rms_m_s2": float("nan")}
    edge_order = 2 if len(time_s) >= 3 else 1
    velocity = np.gradient(predicted_m, time_s, axis=0, edge_order=edge_order)
    acceleration = np.gradient(velocity, time_s, axis=0, edge_order=edge_order)
    sl = slice(2, -2)
    residual = acceleration[sl] - target_acc[sl]
    mag = np.linalg.norm(residual, axis=1)
    return {
        "physics_residual_rms_m_s2": float(np.sqrt(np.mean(mag**2))),
        "physics_residual_mean_m_s2": float(np.mean(mag)),
        "physics_residual_p95_m_s2": float(np.quantile(mag, 0.95)),
    }


def exactness_metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    d = a - b
    n = np.linalg.norm(d, axis=1)
    return {
        "exactness_max_abs_component_m": float(np.max(np.abs(d))),
        "exactness_rms_component_m": float(np.sqrt(np.mean(d**2))),
        "exactness_max_position_m": float(np.max(n)),
        "exactness_mean_position_m": float(np.mean(n)),
    }


def threshold_checks(metrics: Mapping[str, Any], thresholds: Thresholds, include_exactness: bool = False) -> Dict[str, Any]:
    checks = [
        ("mean_error_km", thresholds.mean_error_km_max, "max"),
        ("rmse_error_km", thresholds.rmse_error_km_max, "max"),
        ("p95_error_km", thresholds.p95_error_km_max, "max"),
        ("max_error_km", thresholds.max_error_km_max, "max"),
        ("final_error_km", thresholds.final_error_km_max, "max"),
        ("early_mean_error_km", thresholds.early_mean_error_km_max, "max"),
        ("worst_layer_max_dead_gradient_fraction", thresholds.max_dead_gradient_fraction_max, "max"),
        ("minimum_layer_active_fraction", thresholds.min_active_gradient_fraction_min, "min"),
        ("baseline_gap_closed_pct", thresholds.baseline_gap_closed_pct_min, "min"),
    ]
    if include_exactness:
        checks.append(("exactness_max_position_m", thresholds.exactness_max_position_m_max, "max"))
    rows = []
    for key, limit, direction in checks:
        if key not in metrics:
            continue
        try:
            value = float(metrics[key])
        except Exception:
            continue
        if not np.isfinite(value):
            continue
        passed = value <= limit if direction == "max" else value >= limit
        rows.append({"metric": key, "value": value, "rule": f"{direction} {limit}", "passed": bool(passed)})
    return {
        "passed": bool(rows) and all(x["passed"] for x in rows),
        "checks": rows,
        "failed_checks": [x for x in rows if not x["passed"]],
    }


def ranking_score(metrics: Mapping[str, Any], thresholds: Thresholds) -> float:
    terms = []
    for key, limit in [
        ("mean_error_km", thresholds.mean_error_km_max),
        ("p95_error_km", thresholds.p95_error_km_max),
        ("max_error_km", thresholds.max_error_km_max),
        ("final_error_km", thresholds.final_error_km_max),
        ("early_mean_error_km", thresholds.early_mean_error_km_max),
        ("worst_layer_max_dead_gradient_fraction", thresholds.max_dead_gradient_fraction_max),
    ]:
        if key in metrics and np.isfinite(float(metrics[key])) and limit > 0:
            terms.append(float(metrics[key]) / limit)
    return float(np.mean(terms)) if terms else float("inf")


# =============================================================================
# Plot functions
# =============================================================================


def save_deviation_plot(time_s, predicted, truth, baseline, path: Path, title: str) -> None:
    pred_err = np.linalg.norm(predicted - truth, axis=1) / 1000.0
    base_err = np.linalg.norm(baseline - truth, axis=1) / 1000.0
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(time_s, base_err, label="Analytical baseline")
    ax.plot(time_s, pred_err, label="Model")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Euclidean position error [km]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_rsw_plot(time_s, rsw_km, path: Path, title: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for i, label in enumerate(["Radial", "Along-track", "Cross-track"]):
        axes[i].plot(time_s, rsw_km[:, i])
        axes[i].axhline(0.0, linewidth=0.8)
        axes[i].set_ylabel(f"{label} [km]")
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_gate_plot(time_s, gate, raw_output, R0, path: Path, title: str) -> None:
    raw_km = raw_output * R0 / 1000.0
    gated_km = raw_km * gate
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(time_s, gate[:, 0])
    axes[0].set_ylabel("Gate")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(time_s, np.linalg.norm(raw_km, axis=1), label="Raw |R0 N(t)|")
    axes[1].plot(time_s, np.linalg.norm(gated_km, axis=1), label="Gated |g R0 N(t)|")
    axes[1].set_ylabel("Correction [km]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    for i, label in enumerate("xyz"):
        axes[2].plot(time_s, gated_km[:, i], label=label)
    axes[2].set_ylabel("Gated components [km]")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_activation_plot(time_s, trace: NetworkTrace, path: Path, title: str) -> None:
    n = len(trace.activations)
    fig, axes = plt.subplots(n, 1, figsize=(11, max(3.0 * n, 5)), sharex=True)
    if n == 1:
        axes = [axes]
    for i, (ax, act) in enumerate(zip(axes, trace.activations), start=1):
        deriv = 1.0 - act**2
        ax.plot(time_s, np.mean(deriv >= 0.05, axis=1), label="Active fraction")
        ax.plot(time_s, np.mean(deriv < 1e-3, axis=1), "--", label="Dead-gradient fraction")
        ax.set_ylabel(f"Layer {i}\nfraction")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_first_layer_heatmap(time_s, trace: NetworkTrace, path: Path, title: str) -> None:
    z = trace.preactivations[0]
    act = trace.activations[0]
    deriv = 1.0 - act**2
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    items = [
        (z, -20, 20, "Pre-activation z = W a_prev + b"),
        (act, -1, 1, "Post-activation tanh(z)"),
        (deriv, 0, 1, "Tanh derivative 1 - tanh^2(z)"),
    ]
    for ax, (values, vmin, vmax, label) in zip(axes, items):
        im = ax.imshow(values.T, aspect="auto", origin="lower", extent=[time_s[0], time_s[-1], 0, values.shape[1]], vmin=vmin, vmax=vmax)
        ax.set_ylabel("Neuron")
        ax.set_title(label)
        fig.colorbar(im, ax=ax)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_training_loss(losses: Sequence[float], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(np.arange(1, len(losses) + 1), losses)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_error_distribution(errors_km: np.ndarray, path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(errors_km, bins=50)
    axes[0].set_xlabel("Position error [km]")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Error histogram")
    sorted_err = np.sort(errors_km)
    cdf = np.linspace(0, 1, len(sorted_err), endpoint=True)
    axes[1].plot(sorted_err, cdf)
    axes[1].set_xlabel("Position error [km]")
    axes[1].set_ylabel("Empirical CDF")
    axes[1].set_title("Error CDF")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_mean_error_time(time_s: np.ndarray, error_matrix_km: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(time_s, np.mean(error_matrix_km, axis=0), label="Mean")
    ax.plot(time_s, np.quantile(error_matrix_km, 0.95, axis=0), label="P95")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position error [km]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_comparison_plot(summary: pd.DataFrame, path: Path, title: str) -> None:
    if summary.empty:
        return
    names = summary["configuration"].astype(str).tolist()
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, 1.7 * len(names)), 5.5))
    ax.bar(x - width, summary["mean_error_km"], width, label="Mean")
    ax.bar(x, summary["p95_error_km"], width, label="P95")
    ax.bar(x + width, summary["max_error_km"], width, label="Max")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Position error [km]")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


# =============================================================================
# Fixed-weight checkpoint evaluator
# =============================================================================


@dataclass
class EvalExperiment:
    name: str
    time_feature: str = "t_over_T0"
    gate_type: str = "power"
    gate_power: float = 2.0
    use_j2_baseline: bool = True
    use_osc2mean: bool = True
    use_lat_features: bool = False
    tau_div: float = 3.0
    local_scale_T0: float = 1.0
    exact_checkpoint: bool = False


def eval_preset(name: str, gate_power: float, use_lat: bool) -> List[EvalExperiment]:
    current = EvalExperiment(
        "exact_current",
        time_feature="t_over_T0",
        gate_type="power",
        gate_power=gate_power,
        use_lat_features=use_lat,
        exact_checkpoint=True,
    )
    if name == "current":
        return [current]
    diagnostic = [
        current,
        EvalExperiment("counterfactual_window_time", "t_over_window", "power", gate_power, use_lat_features=use_lat),
        EvalExperiment("counterfactual_orbit_gate", "t_over_T0", "orbit", gate_power, use_lat_features=use_lat),
        EvalExperiment("counterfactual_tanh_gate", "t_over_T0", "tanh", gate_power, use_lat_features=use_lat),
        EvalExperiment("counterfactual_local_exponential", "t_over_T0", "local_exponential", gate_power, use_lat_features=use_lat),
    ]
    if name == "diagnostic":
        return diagnostic
    if name == "full":
        diagnostic.extend([
            EvalExperiment("counterfactual_linear_power_gate", "t_over_T0", "power", 1.0, use_lat_features=use_lat),
            EvalExperiment("counterfactual_cubic_power_gate", "t_over_T0", "power", 3.0, use_lat_features=use_lat),
            EvalExperiment("counterfactual_window_time_orbit_gate", "t_over_window", "orbit", gate_power, use_lat_features=use_lat),
        ])
        return diagnostic
    raise ValueError(f"Unknown evaluation preset: {name}")


def fixed_feature_array(
    arc: OrbitArc,
    baseline: np.ndarray,
    horizon_s: float,
    exp: EvalExperiment,
    v0_scale: float,
) -> Tuple[np.ndarray, List[str], OrbitalFeatures]:
    tokens = ["lat"] if exp.use_lat_features else []
    return build_feature_array(
        arc.time_s, arc.r0, arc.v0, baseline, horizon_s,
        exp.time_feature, tokens, v0_scale,
    )


def aggregate_rows(rows: List[Dict[str, Any]], thresholds: Thresholds, include_exactness: bool = False) -> Dict[str, Any]:
    df = pd.DataFrame(rows)
    out = {
        "satellites_evaluated": int(len(df)),
        "satellite_pass_rate": float(df["passed"].mean()),
        "mean_error_km": float(df["mean_error_km"].mean()),
        "rmse_error_km": float(df["rmse_error_km"].mean()),
        "p95_error_km": float(df["p95_error_km"].mean()),
        "max_error_km": float(df["max_error_km"].max()),
        "final_error_km": float(df["final_error_km"].mean()),
        "early_mean_error_km": float(df["early_mean_error_km"].mean()),
        "baseline_gap_closed_pct": float(df["baseline_gap_closed_pct"].mean()),
        "worst_layer_max_dead_gradient_fraction": float(df["worst_layer_max_dead_gradient_fraction"].max()),
        "minimum_layer_active_fraction": float(df["minimum_layer_active_fraction"].min()),
        "runtime_sec": float(df["runtime_sec"].sum()),
    }
    if include_exactness and "exactness_max_position_m" in df:
        out["exactness_max_position_m"] = float(df["exactness_max_position_m"].max())
    checks = threshold_checks(out, thresholds, include_exactness=include_exactness)
    pass_rate_ok = out["satellite_pass_rate"] >= thresholds.satellite_pass_rate_min
    out["passed"] = bool(checks["passed"] and pass_rate_ok)
    out["failed_checks_json"] = json.dumps(checks["failed_checks"])
    out["configuration_score_lower_is_better"] = ranking_score(out, thresholds)
    return out


def run_fixed_evaluation(
    arcs: List[OrbitArc],
    adapter: ProductionAdapter,
    experiments: List[EvalExperiment],
    horizon_s: float,
    window_s: float,
    thresholds: Thresholds,
    output_dir: Path,
    save_plots: bool,
    save_heatmap: bool,
    compute_physics: bool,
    max_plot_satellites: int,
) -> pd.DataFrame:
    if not adapter.has_checkpoint or adapter.numpy_net is None or adapter.layers is None:
        raise RuntimeError("Fixed evaluation requires a checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_input_dim = adapter.layers[0]
    inferred_lat = checkpoint_input_dim == 11
    if checkpoint_input_dim not in {9, 11}:
        warn(f"Checkpoint input dimension is {checkpoint_input_dim}; exact feature reconstruction supports 9 or 11 only")

    summaries = []
    all_rows = []
    all_layers = []
    for exp in experiments:
        if exp.use_lat_features != inferred_lat and exp.exact_checkpoint:
            raise ValueError(
                f"Exact checkpoint input dimension {checkpoint_input_dim} implies use_lat_features={inferred_lat}, "
                f"but experiment has {exp.use_lat_features}"
            )
        exp_dir = output_dir / safe_name(exp.name)
        exp_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for sat_index, arc in enumerate(arcs):
            started = time.time()
            baseline = adapter.baseline(arc.r0, arc.v0, arc.time_s, exp.use_j2_baseline, exp.use_osc2mean)
            features, feature_names, of = fixed_feature_array(arc, baseline, window_s, exp, getattr(adapter.lib, "V0_SCALE", V0_SCALE_DEFAULT) if adapter.lib else V0_SCALE_DEFAULT)
            if features.shape[1] != checkpoint_input_dim:
                raise ValueError(
                    f"Experiment {exp.name}: feature dimension {features.shape[1]} does not match checkpoint {checkpoint_input_dim}"
                )
            trace = adapter.numpy_net.forward(features)
            gate = compute_gate_np(
                arc.time_s, exp.gate_type, exp.gate_power, window_s, of.T0, of.Torb,
                tau_div=exp.tau_div, local_scale_T0=exp.local_scale_T0,
            )
            residual_m = gate * trace.output.astype(np.float64) * of.R0
            predicted = baseline + residual_m
            metrics = {
                "configuration": exp.name,
                "satellite": arc.satellite,
                "evaluation_mode": "exact_checkpoint" if exp.exact_checkpoint else "counterfactual_fixed_weights",
                "truth_source": arc.source,
                "runtime_sec": time.time() - started,
            }
            metrics.update(position_metrics(predicted, arc.position_m, baseline, arc.time_s, horizon_s))
            rsw, rsw_metrics = rsw_components(predicted, arc.position_m, arc.velocity_m_s)
            metrics.update(rsw_metrics)
            layer_rows, act_summary = activation_metrics(trace)
            metrics.update(act_summary)
            metrics.update(correction_metrics(trace.output, gate, of.R0, arc.time_s))
            if compute_physics:
                target_acc = adapter.physics_acceleration(predicted, use_j2=True)
                metrics.update(physics_residual_metrics(predicted, arc.time_s, target_acc))
            if exp.exact_checkpoint:
                reference = adapter.exact_reference_positions(
                    arc.r0, arc.v0, arc.time_s, window_s,
                    exp.use_j2_baseline, exp.use_osc2mean, exp.use_lat_features, exp.gate_power,
                )
                if reference is not None:
                    metrics.update(exactness_metrics(predicted, reference))
                else:
                    warn("Production GeneralistModel exact reference is unavailable; exactness metric omitted")
            checks = threshold_checks(metrics, thresholds, include_exactness=exp.exact_checkpoint and "exactness_max_position_m" in metrics)
            metrics["passed"] = bool(checks["passed"])
            metrics["failed_checks_json"] = json.dumps(checks["failed_checks"])
            rows.append(metrics)
            all_rows.append(metrics)
            for layer in layer_rows:
                layer.update(configuration=exp.name, satellite=arc.satellite)
                all_layers.append(layer)

            sat_dir = exp_dir / f"sat_{safe_name(arc.satellite)}"
            sat_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([metrics]).to_csv(sat_dir / "metrics.csv", index=False)
            pd.DataFrame(layer_rows).to_csv(sat_dir / "activation_layers.csv", index=False)
            json_dump({"features": feature_names, "checks": checks}, sat_dir / "report.json")
            if save_plots and sat_index < max_plot_satellites:
                save_deviation_plot(arc.time_s, predicted, arc.position_m, baseline, sat_dir / "01_deviation.png", f"{exp.name} — satellite {arc.satellite}")
                save_rsw_plot(arc.time_s, rsw, sat_dir / "02_rsw.png", f"RSW error — {exp.name}, satellite {arc.satellite}")
                save_gate_plot(arc.time_s, gate, trace.output, of.R0, sat_dir / "03_gate_and_correction.png", f"Gate and correction — {exp.name}")
                save_activation_plot(arc.time_s, trace, sat_dir / "04_activation_health.png", f"Activation health — {exp.name}")
                if save_heatmap:
                    save_first_layer_heatmap(arc.time_s, trace, sat_dir / "05_first_layer_heatmap.png", f"First layer — {exp.name}")

        exp_df = pd.DataFrame(rows)
        exp_df.to_csv(exp_dir / "satellite_metrics.csv", index=False)
        summary = aggregate_rows(rows, thresholds, include_exactness=exp.exact_checkpoint and "exactness_max_position_m" in exp_df.columns)
        summary.update({
            "configuration": exp.name,
            "workflow": "fixed_checkpoint",
            "evaluation_mode": "exact_checkpoint" if exp.exact_checkpoint else "counterfactual_fixed_weights",
            "status": ("PASS" if summary["passed"] else "FAIL") if exp.exact_checkpoint else ("PASS_DIAGNOSTIC" if summary["passed"] else "FAIL_DIAGNOSTIC"),
        })
        summaries.append(summary)
        json_dump(summary, exp_dir / "summary.json")
        log(f"[evaluate] {exp.name}: {summary['status']} mean={summary['mean_error_km']:.3f} km p95={summary['p95_error_km']:.3f} km max={summary['max_error_km']:.3f} km")

    pd.DataFrame(all_rows).to_csv(output_dir / "all_satellite_metrics.csv", index=False)
    pd.DataFrame(all_layers).to_csv(output_dir / "all_activation_layers.csv", index=False)
    summary_df = pd.DataFrame(summaries).sort_values("configuration_score_lower_is_better")
    summary_df.to_csv(output_dir / "configuration_comparison.csv", index=False)
    json_dump(summary_df.to_dict(orient="records"), output_dir / "configuration_comparison.json")
    if save_plots:
        save_comparison_plot(summary_df, output_dir / "configuration_comparison.png", "Fixed checkpoint configuration comparison")
    return summary_df


# =============================================================================
# Rapid retraining testbed
# =============================================================================


@dataclass
class SweepExperiment:
    name: str
    gate_type: str
    time_feature: str
    features: Tuple[str, ...] = ()
    gate_power: float = 2.0
    tau_div: float = 3.0
    local_scale_T0: float = 1.0


def sweep_preset(name: str) -> List[SweepExperiment]:
    current = SweepExperiment("current_power_base9", "power", "t_over_T0", ())
    core = [
        current,
        SweepExperiment("window_time_power", "power", "t_over_window", ()),
        SweepExperiment("orbit_gate_base9", "orbit", "t_over_T0", ()),
        SweepExperiment("tanh_gate_base9", "tanh", "t_over_T0", ()),
        SweepExperiment("local_exp_gate_base9", "local_exponential", "t_over_T0", ()),
    ]
    if name == "core":
        return core
    j2 = core + [
        SweepExperiment("tanh_lat", "tanh", "t_over_window", ("lat",)),
        SweepExperiment("tanh_lat_inc", "tanh", "t_over_window", ("lat", "inc")),
        SweepExperiment("tanh_lat_inc_ecc_acc", "tanh", "t_over_window", ("lat", "inc", "ecc", "acc")),
        SweepExperiment("tanh_full_j2_features", "tanh", "t_over_window", ("lat", "inc", "ecc", "acc", "phase", "rates")),
    ]
    if name == "j2":
        return j2
    if name == "full":
        return j2 + [
            SweepExperiment("orbit_full_j2_features", "orbit", "t_over_window", ("lat", "inc", "ecc", "acc", "phase", "rates")),
            SweepExperiment("local_exp_full_j2_features", "local_exponential", "t_over_window", ("lat", "inc", "ecc", "acc", "phase", "rates")),
            SweepExperiment("linear_power_full_features", "power", "t_over_window", ("lat", "inc", "ecc", "acc", "phase", "rates"), gate_power=1.0),
        ]
    raise ValueError(f"Unknown sweep preset: {name}")


def load_custom_sweep(path: Optional[Path]) -> Optional[List[SweepExperiment]]:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["experiments"] if isinstance(data, dict) else data
    result = []
    for item in items:
        payload = dict(item)
        payload["features"] = tuple(payload.get("features", []))
        result.append(SweepExperiment(**payload))
    return result


class ScreenMLP(nn.Module):
    def __init__(self, input_dim: int, width: int, depth: int, output_dim: int = 3):
        super().__init__()
        layers = [input_dim] + [width] * depth + [output_dim]
        self.layers = layers
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        for layer in self.linear[:-1]:
            nn.init.xavier_normal_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.linear[-1].weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.linear[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.linear[:-1]:
            x = torch.tanh(layer(x))
        return self.linear[-1](x)

    def trace(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        zlist, alist = [], []
        h = x
        for layer in self.linear[:-1]:
            z = layer(h)
            h = torch.tanh(z)
            zlist.append(z)
            alist.append(h)
        return zlist, alist, self.linear[-1](h)


@dataclass
class FlatDataset:
    satellite_ids: List[str]
    sat_index: np.ndarray
    time_s: np.ndarray
    r0: np.ndarray
    v0: np.ndarray
    truth_r: np.ndarray
    truth_v: np.ndarray
    baseline_r: np.ndarray
    R0: np.ndarray
    V0: np.ndarray
    T0: np.ndarray
    Torb: np.ndarray
    arcs: List[OrbitArc]


def prepare_flat_dataset(
    arcs: List[OrbitArc],
    adapter: ProductionAdapter,
    horizon_s: float,
    anchors: Optional[int],
    use_j2_baseline: bool,
    use_osc2mean: bool,
) -> FlatDataset:
    sat_index, times, r0s, v0s, truths, truth_vs, baselines = [], [], [], [], [], [], []
    selected_arcs = []
    for si, arc in enumerate(arcs):
        idx = stratified_time_indices(arc.time_s, anchors, horizon_s)
        a = arc.subset(idx)
        baseline = adapter.baseline(a.r0, a.v0, a.time_s, use_j2_baseline, use_osc2mean)
        n = len(a.time_s)
        sat_index.append(np.full(n, si, dtype=np.int64))
        times.append(a.time_s.reshape(-1, 1))
        r0s.append(np.repeat(a.r0.reshape(1, 3), n, axis=0))
        v0s.append(np.repeat(a.v0.reshape(1, 3), n, axis=0))
        truths.append(a.position_m)
        truth_vs.append(a.velocity_m_s)
        baselines.append(baseline)
        selected_arcs.append(a)
    r0_arr = np.concatenate(r0s)
    v0_arr = np.concatenate(v0s)
    of_values = [orbital_features(a.r0, a.v0) for a in selected_arcs]
    sat_idx = np.concatenate(sat_index)
    return FlatDataset(
        satellite_ids=[a.satellite for a in selected_arcs],
        sat_index=sat_idx,
        time_s=np.concatenate(times),
        r0=r0_arr,
        v0=v0_arr,
        truth_r=np.concatenate(truths),
        truth_v=np.concatenate(truth_vs),
        baseline_r=np.concatenate(baselines),
        R0=np.array([of_values[i].R0 for i in sat_idx], dtype=np.float64).reshape(-1, 1),
        V0=np.array([of_values[i].V0 for i in sat_idx], dtype=np.float64).reshape(-1, 1),
        T0=np.array([of_values[i].T0 for i in sat_idx], dtype=np.float64).reshape(-1, 1),
        Torb=np.array([of_values[i].Torb for i in sat_idx], dtype=np.float64).reshape(-1, 1),
        arcs=selected_arcs,
    )


def build_flat_features(dataset: FlatDataset, exp: SweepExperiment, horizon_s: float) -> Tuple[np.ndarray, List[str]]:
    features = []
    names = None
    for arc_idx, arc in enumerate(dataset.arcs):
        mask = dataset.sat_index == arc_idx
        vals, feature_names, _ = build_feature_array(
            dataset.time_s[mask, 0], arc.r0, arc.v0, dataset.baseline_r[mask],
            horizon_s, exp.time_feature, exp.features,
        )
        features.append(vals)
        names = feature_names
    return np.concatenate(features), names or []


def torch_trace_to_numpy(model: ScreenMLP, features: np.ndarray, device: torch.device, dtype: torch.dtype) -> NetworkTrace:
    x = torch.tensor(features, device=device, dtype=dtype)
    with torch.no_grad():
        z, a, out = model.trace(x)
    return NetworkTrace(
        inputs=features,
        preactivations=[v.detach().cpu().numpy() for v in z],
        activations=[v.detach().cpu().numpy() for v in a],
        output=out.detach().cpu().numpy(),
    )


def run_one_sweep(
    exp: SweepExperiment,
    train: FlatDataset,
    val: FlatDataset,
    horizon_s: float,
    width: int,
    depth: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    dtype: torch.dtype,
    thresholds: Thresholds,
    output_dir: Path,
    seed: int,
    save_plots: bool,
    save_heatmap: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    exp_dir = output_dir / safe_name(exp.name)
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_x_np, feature_names = build_flat_features(train, exp, horizon_s)
    val_x_np, _ = build_flat_features(val, exp, horizon_s)
    train_x = torch.tensor(train_x_np, device=device, dtype=dtype)
    val_x = torch.tensor(val_x_np, device=device, dtype=dtype)
    train_t = torch.tensor(train.time_s, device=device, dtype=dtype)
    val_t = torch.tensor(val.time_s, device=device, dtype=dtype)
    train_R0 = torch.tensor(train.R0, device=device, dtype=dtype)
    val_R0 = torch.tensor(val.R0, device=device, dtype=dtype)
    train_T0 = torch.tensor(train.T0, device=device, dtype=dtype)
    val_T0 = torch.tensor(val.T0, device=device, dtype=dtype)
    train_Torb = torch.tensor(train.Torb, device=device, dtype=dtype)
    val_Torb = torch.tensor(val.Torb, device=device, dtype=dtype)
    train_baseline = torch.tensor(train.baseline_r, device=device, dtype=dtype)
    val_baseline = torch.tensor(val.baseline_r, device=device, dtype=dtype)
    train_truth = torch.tensor(train.truth_r, device=device, dtype=dtype)
    val_truth = torch.tensor(val.truth_r, device=device, dtype=dtype)

    model = ScreenMLP(train_x.shape[1], width, depth).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    ntrain = len(train_x)
    losses = []
    started = time.time()
    model.train()
    for ep in range(epochs):
        if batch_size >= ntrain:
            idx = torch.arange(ntrain, device=device)
        else:
            idx = torch.randperm(ntrain, device=device)[:batch_size]
        gate = compute_gate_torch(
            train_t[idx], exp.gate_type, exp.gate_power, horizon_s,
            train_T0[idx], train_Torb[idx], exp.tau_div, exp.local_scale_T0,
        )
        pred = train_baseline[idx] + gate * model(train_x[idx]) * train_R0[idx]
        loss = torch.mean(((pred - train_truth[idx]) / train_R0[idx]) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))

    model.eval()
    with torch.no_grad():
        val_gate = compute_gate_torch(
            val_t, exp.gate_type, exp.gate_power, horizon_s,
            val_T0, val_Torb, exp.tau_div, exp.local_scale_T0,
        )
        raw = model(val_x)
        pred = val_baseline + val_gate * raw * val_R0
    pred_np = pred.detach().cpu().numpy().astype(np.float64)
    raw_np = raw.detach().cpu().numpy().astype(np.float64)
    gate_np = val_gate.detach().cpu().numpy().astype(np.float64)

    sat_rows, layer_rows = [], []
    point_errors = np.linalg.norm(pred_np - val.truth_r, axis=1) / 1000.0
    equal_lengths = len({len(a.time_s) for a in val.arcs}) == 1
    error_matrix = []
    representative_trace = None
    representative_gate = None
    representative_raw = None
    representative_pred = None
    representative_rsw = None
    offset = 0
    for si, arc in enumerate(val.arcs):
        mask = val.sat_index == si
        p = pred_np[mask]
        b = val.baseline_r[mask]
        metrics = {
            "configuration": exp.name,
            "satellite": arc.satellite,
            "workflow": "rapid_retraining_screen",
            "runtime_sec": 0.0,
        }
        metrics.update(position_metrics(p, arc.position_m, b, arc.time_s, horizon_s))
        rsw, rmetrics = rsw_components(p, arc.position_m, arc.velocity_m_s)
        metrics.update(rmetrics)
        trace = torch_trace_to_numpy(model, val_x_np[mask], device, dtype)
        layers, act_summary = activation_metrics(trace)
        metrics.update(act_summary)
        metrics.update(correction_metrics(raw_np[mask], gate_np[mask], orbital_features(arc.r0, arc.v0).R0, arc.time_s))
        checks = threshold_checks(metrics, thresholds, include_exactness=False)
        metrics["passed"] = bool(checks["passed"])
        metrics["failed_checks_json"] = json.dumps(checks["failed_checks"])
        sat_rows.append(metrics)
        for layer in layers:
            layer.update(configuration=exp.name, satellite=arc.satellite)
            layer_rows.append(layer)
        if equal_lengths:
            error_matrix.append(np.linalg.norm(p - arc.position_m, axis=1) / 1000.0)
        if si == 0:
            representative_trace = trace
            representative_gate = gate_np[mask]
            representative_raw = raw_np[mask]
            representative_pred = p
            representative_rsw = rsw
        offset += len(arc.time_s)

    summary = aggregate_rows(sat_rows, thresholds, include_exactness=False)
    summary.update({
        "configuration": exp.name,
        "workflow": "rapid_retraining_screen",
        "status": "PASS_SCREEN" if summary["passed"] else "FAIL_SCREEN",
        "gate_type": exp.gate_type,
        "time_feature": exp.time_feature,
        "features": "+".join(exp.features) if exp.features else "base9",
        "input_dim": int(train_x.shape[1]),
        "width": int(width),
        "depth": int(depth),
        "epochs": int(epochs),
        "training_runtime_sec": float(time.time() - started),
        "final_training_loss": float(losses[-1]),
    })

    torch.save({
        "model_state": model.state_dict(),
        "layers": model.layers,
        "configuration": asdict(exp),
        "feature_names": feature_names,
        "horizon_sec": horizon_s,
    }, exp_dir / "screened_model.pth")
    pd.DataFrame(sat_rows).to_csv(exp_dir / "satellite_metrics.csv", index=False)
    pd.DataFrame(layer_rows).to_csv(exp_dir / "activation_layers.csv", index=False)
    json_dump(summary, exp_dir / "summary.json")
    json_dump({"configuration": asdict(exp), "feature_names": feature_names}, exp_dir / "manifest.json")

    if save_plots:
        save_training_loss(losses, exp_dir / "01_training_loss.png", f"Training loss — {exp.name}")
        save_error_distribution(point_errors, exp_dir / "02_error_distribution.png", f"Validation errors — {exp.name}")
        if equal_lengths and error_matrix:
            save_mean_error_time(val.arcs[0].time_s, np.asarray(error_matrix), exp_dir / "03_mean_error_over_time.png", f"Validation error over time — {exp.name}")
        if representative_trace is not None:
            arc = val.arcs[0]
            save_deviation_plot(arc.time_s, representative_pred, arc.position_m, val.baseline_r[val.sat_index == 0], exp_dir / "04_representative_deviation.png", f"Representative validation satellite — {exp.name}")
            save_rsw_plot(arc.time_s, representative_rsw, exp_dir / "05_representative_rsw.png", f"Representative RSW — {exp.name}")
            save_gate_plot(arc.time_s, representative_gate, representative_raw, orbital_features(arc.r0, arc.v0).R0, exp_dir / "06_gate_and_correction.png", f"Gate and correction — {exp.name}")
            save_activation_plot(arc.time_s, representative_trace, exp_dir / "07_activation_health.png", f"Activation health — {exp.name}")
            if save_heatmap:
                save_first_layer_heatmap(arc.time_s, representative_trace, exp_dir / "08_first_layer_heatmap.png", f"First hidden layer — {exp.name}")

    log(f"[sweep] {exp.name}: {summary['status']} mean={summary['mean_error_km']:.3f} km early={summary['early_mean_error_km']:.3f} km max={summary['max_error_km']:.3f} km")
    return summary, sat_rows, layer_rows


def run_sweep(
    train_arcs: List[OrbitArc],
    val_arcs: List[OrbitArc],
    adapter: ProductionAdapter,
    experiments: List[SweepExperiment],
    horizon_s: float,
    anchors: Optional[int],
    use_j2_baseline: bool,
    use_osc2mean: bool,
    width: int,
    depth: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    dtype: torch.dtype,
    thresholds: Thresholds,
    output_dir: Path,
    seed: int,
    save_plots: bool,
    save_heatmap: bool,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    log("[sweep] preparing baseline and flattened train/validation datasets")
    train = prepare_flat_dataset(train_arcs, adapter, horizon_s, anchors, use_j2_baseline, use_osc2mean)
    val = prepare_flat_dataset(val_arcs, adapter, horizon_s, anchors, use_j2_baseline, use_osc2mean)
    summaries, all_rows, all_layers = [], [], []
    for exp in experiments:
        summary, rows, layers = run_one_sweep(
            exp, train, val, horizon_s, width, depth, epochs, batch_size, lr,
            device, dtype, thresholds, output_dir, seed, save_plots, save_heatmap,
        )
        summaries.append(summary)
        all_rows.extend(rows)
        all_layers.extend(layers)
    pd.DataFrame(all_rows).to_csv(output_dir / "all_satellite_metrics.csv", index=False)
    pd.DataFrame(all_layers).to_csv(output_dir / "all_activation_layers.csv", index=False)
    summary_df = pd.DataFrame(summaries).sort_values("configuration_score_lower_is_better")
    summary_df.to_csv(output_dir / "configuration_comparison.csv", index=False)
    json_dump(summary_df.to_dict(orient="records"), output_dir / "configuration_comparison.json")
    if save_plots:
        save_comparison_plot(summary_df, output_dir / "configuration_comparison.png", "Rapid retraining configuration comparison")
    return summary_df


# =============================================================================
# Input resolution and orchestration
# =============================================================================


def parse_satellites(text: str) -> Optional[List[str]]:
    if text.lower() == "all":
        return None
    return [x.strip() for x in text.split(",") if x.strip()]


def determine_horizon(explicit: Optional[float], repo: Optional[CsvTruthRepository], paths: DiscoveryResult) -> float:
    if explicit is not None:
        return float(explicit)
    if repo is not None:
        return repo.infer_horizon()
    # Filename fallback.
    for candidate in [paths.dataset, paths.checkpoint]:
        if candidate is None:
            continue
        match = re.search(r"(16|40|60|80)k", str(candidate).lower())
        if match:
            return float(match.group(1)) * 1000.0
    return 40000.0


def split_arcs(arcs: List[OrbitArc], train_count: int, val_count: int, seed: int) -> Tuple[List[OrbitArc], List[OrbitArc]]:
    if len(arcs) < 2:
        raise ValueError("Need at least two satellites for train/validation split")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(arcs))
    max_train = min(train_count, max(1, len(arcs) - 1))
    max_val = min(val_count, len(arcs) - max_train)
    if max_val < 1:
        max_val = 1
        max_train = len(arcs) - 1
    train_idx = order[:max_train]
    val_idx = order[max_train:max_train + max_val]
    return [arcs[i] for i in train_idx], [arcs[i] for i in val_idx]


def build_thresholds(args) -> Thresholds:
    return Thresholds(
        mean_error_km_max=args.threshold_mean,
        rmse_error_km_max=args.threshold_rmse,
        p95_error_km_max=args.threshold_p95,
        max_error_km_max=args.threshold_max,
        final_error_km_max=args.threshold_final,
        early_mean_error_km_max=args.threshold_early,
        max_dead_gradient_fraction_max=args.threshold_dead,
        min_active_gradient_fraction_min=args.threshold_active,
        baseline_gap_closed_pct_min=args.threshold_gap_closed,
        exactness_max_position_m_max=args.threshold_exactness_m,
        satellite_pass_rate_min=args.threshold_pass_rate,
    )


def resolve_inputs(args) -> Tuple[DiscoveryResult, Optional[Path], Optional[Path], Optional[Path], Optional[CsvTruthRepository], float, Path]:
    root = Path(args.root).expanduser().resolve()
    discovery = discover_project(root, args.horizon)
    dataset = resolve_path(args.dataset, discovery.dataset, "dataset")
    library = resolve_path(args.library, discovery.library, "library")
    checkpoint = resolve_path(args.checkpoint, discovery.checkpoint, "checkpoint")
    repo = CsvTruthRepository(dataset) if dataset is not None else None
    horizon = determine_horizon(args.horizon, repo, discovery)
    output = Path(args.output).expanduser().resolve() if args.output else root / "unified_pinn_lab_results" / utc_stamp()
    output.mkdir(parents=True, exist_ok=True)
    return discovery, dataset, library, checkpoint, repo, horizon, output


def load_truth_arcs(
    args,
    repo: Optional[CsvTruthRepository],
    horizon: float,
    output: Path,
    total_needed: int,
) -> List[OrbitArc]:
    satellites = parse_satellites(args.satellites)
    backend = args.truth_backend
    if repo is not None and backend in {"auto", "csv"}:
        max_sats = args.max_satellites
        if max_sats is None and total_needed > 0:
            max_sats = total_needed
        arcs = repo.load_arcs(horizon, satellites, max_sats, args.seed)
        log(f"[truth] using CSV dataset: {repo.path} ({len(arcs)} satellites)")
        return arcs
    if backend == "csv":
        raise ValueError("truth_backend=csv but no dataset was supplied or discovered")
    count = max(total_needed, 2)
    return generate_synthetic_arcs(
        count=count,
        horizon_s=horizon,
        anchors=args.anchors,
        seed=args.seed,
        backend=backend,
        cache_dir=output / "cache",
    )


def write_decision_report(
    output: Path,
    fixed_summary: Optional[pd.DataFrame],
    sweep_summary: Optional[pd.DataFrame],
) -> None:
    lines = [
        "UNIFIED PINN LAB DECISION REPORT",
        "=" * 40,
        "",
        "Fixed-checkpoint results describe the supplied trained model.",
        "Counterfactual fixed-weight results are diagnostic only.",
        "Rapid retraining results train a new model and are screening evidence before a full production run.",
        "",
    ]
    if fixed_summary is not None and not fixed_summary.empty:
        lines.append("FIXED CHECKPOINT")
        for row in fixed_summary.to_dict(orient="records"):
            lines.append(
                f"- {row['configuration']}: {row['status']} | score={row['configuration_score_lower_is_better']:.3f} | "
                f"mean={row['mean_error_km']:.3f} km | p95={row['p95_error_km']:.3f} km | max={row['max_error_km']:.3f} km"
            )
        lines.append("")
    if sweep_summary is not None and not sweep_summary.empty:
        lines.append("RAPID RETRAINING SCREEN")
        for row in sweep_summary.to_dict(orient="records"):
            lines.append(
                f"- {row['configuration']}: {row['status']} | score={row['configuration_score_lower_is_better']:.3f} | "
                f"mean={row['mean_error_km']:.3f} km | early={row['early_mean_error_km']:.3f} km | max={row['max_error_km']:.3f} km"
            )
        best = sweep_summary.iloc[0]
        lines.extend(["", f"Best screened configuration: {best['configuration']}"])
    (output / "decision.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_workflow(args) -> int:
    discovery, dataset, library, checkpoint, repo, horizon, output = resolve_inputs(args)
    device = choose_device(args.device)
    adapter = ProductionAdapter(library, checkpoint, device)
    dtype = choose_dtype(args.dtype, adapter.dtype if adapter.has_checkpoint else None)
    thresholds = build_thresholds(args)
    window_s = args.window_trained if args.window_trained is not None else horizon

    manifest = {
        "command": args.command,
        "root": str(discovery.root),
        "dataset": str(dataset) if dataset else None,
        "library": str(library) if library else None,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "horizon_sec": horizon,
        "window_trained_sec": window_s,
        "device": str(device),
        "dtype": str(dtype),
        "thresholds": asdict(thresholds),
        "arguments": vars(args),
        "discovery": asdict(discovery),
        "production_baseline_exact": bool(adapter.has_production_library),
    }
    json_dump(manifest, output / "run_manifest.json")

    log(f"[paths] output:     {output}")
    log(f"[paths] dataset:    {dataset or 'none -> generated truth'}")
    log(f"[paths] library:    {library or 'none -> internal baseline approximation'}")
    log(f"[paths] checkpoint: {checkpoint or 'none'}")
    log(f"[run] horizon={horizon:.0f}s window_trained={window_s:.0f}s device={device} dtype={dtype}")

    do_eval = args.command in {"evaluate", "all"}
    do_sweep = args.command in {"sweep", "all"}
    total_needed = (args.train_sats + args.val_sats) if do_sweep else args.eval_max_satellites
    arcs = load_truth_arcs(args, repo, horizon, output, total_needed)

    fixed_summary = None
    if do_eval:
        if not adapter.has_checkpoint:
            if args.command == "evaluate":
                raise RuntimeError("No checkpoint was supplied or discovered")
            warn("No checkpoint found; skipping fixed evaluation")
        else:
            use_lat = adapter.layers[0] == 11 if args.lat_features is None else args.lat_features
            experiments = eval_preset(args.eval_preset, args.gate_power, use_lat)
            eval_arcs = arcs[: min(args.eval_max_satellites, len(arcs))]
            fixed_summary = run_fixed_evaluation(
                eval_arcs, adapter, experiments, horizon, window_s, thresholds,
                output / "fixed_checkpoint", args.save_plots, args.first_layer_heatmap,
                args.physics_residual, args.max_plot_satellites,
            )

    sweep_summary = None
    if do_sweep:
        custom = load_custom_sweep(Path(args.sweep_json).resolve() if args.sweep_json else None)
        experiments = custom or sweep_preset(args.sweep_preset)
        train_arcs, val_arcs = split_arcs(arcs, args.train_sats, args.val_sats, args.seed)
        if args.width is None or args.depth is None:
            if adapter.layers is not None:
                inferred_width = adapter.layers[1]
                inferred_depth = len(adapter.layers) - 2
            else:
                inferred_width, inferred_depth = 256, 5
        else:
            inferred_width, inferred_depth = args.width, args.depth
        width = args.width or inferred_width
        depth = args.depth or inferred_depth
        sweep_summary = run_sweep(
            train_arcs, val_arcs, adapter, experiments, horizon, args.anchors,
            args.use_j2_baseline, args.use_osc2mean,
            width, depth, args.epochs, args.batch_size, args.lr,
            device, dtype, thresholds, output / "rapid_retraining",
            args.seed, args.save_plots, args.first_layer_heatmap,
        )

    write_decision_report(output, fixed_summary, sweep_summary)
    log(f"[done] results written to {output}")
    return 0


# =============================================================================
# CLI
# =============================================================================


def add_ablation_arguments(parser: argparse.ArgumentParser) -> None:
    ablation_dir = Path(__file__).resolve().parent / "run_ablation"
    parser.add_argument("--data", required=True, help="Propagated truth CSV")
    parser.add_argument("--library", default=None, help="Production PINN library .py")
    parser.add_argument("--output-dir", default="numerical_pinn_ablation_results")
    parser.add_argument(
        "--feature-config",
        default=str(ablation_dir / "ablation_config.json"),
        help="Feature-ablation JSON configuration",
    )
    parser.add_argument(
        "--physics-config",
        default=str(ablation_dir / "physics_config.json"),
        help="Physics-probe JSON configuration",
    )
    parser.add_argument("--horizon", type=float, default=None)
    parser.add_argument("--max-satellites", type=int, default=400)
    parser.add_argument("--points-per-satellite", type=int, default=400)
    parser.add_argument("--physics-points-per-satellite", type=int, default=200)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--physics-checkpoint", default=None)
    parser.add_argument("--alignment-only", action="store_true")


def _append_option(arguments: List[str], name: str, value: Any) -> None:
    if value is not None:
        arguments.extend([name, str(value)])


def run_integrated_ablation(args) -> int:
    from run_ablation import feature_ablation_lab, physics_probe_lab

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_feature = args.command in {"feature-ablation", "ablation"}
    run_physics = args.command in {"physics-probe", "ablation"}
    combined = args.command == "ablation"
    completed = []

    if run_feature:
        feature_output = output / "feature_ablation" if combined else output
        feature_arguments = [
            "--config", args.feature_config,
            "--data", args.data,
            "--output-dir", str(feature_output),
            "--max-satellites", str(args.max_satellites),
            "--points-per-satellite", str(args.points_per_satellite),
            "--seed", str(args.seed),
        ]
        _append_option(feature_arguments, "--library", args.library)
        _append_option(feature_arguments, "--horizon", args.horizon)
        result = feature_ablation_lab.main(feature_arguments)
        if result:
            return int(result)
        completed.append({"experiment": "feature_ablation", "output": str(feature_output)})

    if run_physics:
        physics_output = output / "physics_probe" if combined else output
        physics_arguments = [
            "--config", args.physics_config,
            "--data", args.data,
            "--output-dir", str(physics_output),
            "--max-satellites", str(args.max_satellites),
            "--points-per-satellite", str(args.physics_points_per_satellite),
            "--val-fraction", str(args.val_fraction),
            "--seed", str(args.seed),
        ]
        _append_option(physics_arguments, "--library", args.library)
        _append_option(physics_arguments, "--horizon", args.horizon)
        _append_option(physics_arguments, "--checkpoint", args.physics_checkpoint)
        if args.alignment_only:
            physics_arguments.append("--alignment-only")
        result = physics_probe_lab.main(physics_arguments)
        if result:
            return int(result)
        completed.append({"experiment": "physics_probe", "output": str(physics_output)})

    json_dump(
        {
            "framework": "numerical PINN experimentation and evaluation framework",
            "command": args.command,
            "data": str(Path(args.data).expanduser().resolve()),
            "library": args.library,
            "feature_config": args.feature_config,
            "physics_config": args.physics_config,
            "horizon": args.horizon,
            "max_satellites": args.max_satellites,
            "seed": args.seed,
            "experiments": completed,
        },
        output / "framework_manifest.json",
    )
    log(f"[done] integrated ablation results written to {output}")
    return 0


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default=".", help="Project root to search recursively")
    parser.add_argument("--dataset", default=None, help="Propagated truth CSV. Omit for auto-discovery; use 'none' to disable")
    parser.add_argument("--library", default=None, help="Production PINN library .py. Omit for auto-discovery")
    parser.add_argument("--checkpoint", default=None, help="Trained .pth/.pt checkpoint. Omit for auto-discovery")
    parser.add_argument("--output", default=None, help="Output directory. Default: ROOT/unified_pinn_lab_results/TIMESTAMP")
    parser.add_argument("--horizon", type=parse_float_auto, default=None, help="Propagation horizon seconds or 'auto'")
    parser.add_argument("--window-trained", type=parse_float_auto, default=None, help="Gate reference window seconds; default=horizon")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float64", "fp32", "fp64"], default="auto")
    parser.add_argument("--truth-backend", choices=["auto", "csv", "orekit", "dop853"], default="auto")
    parser.add_argument("--satellites", default="all", help="Comma-separated satellite IDs or 'all'")
    parser.add_argument("--max-satellites", type=int, default=None, help="Maximum CSV satellites to load")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anchors", type=int, default=120, help="Points per satellite for rapid retraining")
    parser.add_argument("--save-plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--first-layer-heatmap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-plot-satellites", type=int, default=3)
    parser.add_argument("--physics-residual", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-j2-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-osc2mean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gate-power", type=float, default=2.0)
    parser.add_argument("--lat-features", type=parse_bool_auto, default=None, help="auto/true/false for fixed checkpoint")

    parser.add_argument("--threshold-mean", type=float, default=2.0)
    parser.add_argument("--threshold-rmse", type=float, default=2.5)
    parser.add_argument("--threshold-p95", type=float, default=3.0)
    parser.add_argument("--threshold-max", type=float, default=5.0)
    parser.add_argument("--threshold-final", type=float, default=3.0)
    parser.add_argument("--threshold-early", type=float, default=2.5)
    parser.add_argument("--threshold-dead", type=float, default=0.25)
    parser.add_argument("--threshold-active", type=float, default=0.60)
    parser.add_argument("--threshold-gap-closed", type=float, default=0.0)
    parser.add_argument("--threshold-exactness-m", type=float, default=2.0)
    parser.add_argument("--threshold-pass-rate", type=float, default=0.90)

    parser.add_argument("--eval-preset", choices=["current", "diagnostic", "full"], default="diagnostic")
    parser.add_argument("--eval-max-satellites", type=int, default=10)
    parser.add_argument("--sweep-preset", choices=["core", "j2", "full"], default="j2")
    parser.add_argument("--sweep-json", default=None, help="Optional JSON list of custom sweep experiments")
    parser.add_argument("--train-sats", type=int, default=200)
    parser.add_argument("--val-sats", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--width", type=int, default=None, help="Default: infer from checkpoint, else 256")
    parser.add_argument("--depth", type=int, default=None, help="Default: infer from checkpoint, else 5")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Numerical PINN experimentation, evaluation, retraining, and ablation framework.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    discover = sub.add_parser("discover", help="Show automatically discovered dataset/library/checkpoint")
    discover.add_argument("--root", default=".")
    discover.add_argument("--horizon", type=parse_float_auto, default=None)

    for name, help_text in [
        ("evaluate", "Evaluate the actual trained checkpoint and fixed-weight counterfactuals"),
        ("sweep", "Train and rank new configurations rapidly"),
        ("all", "Run fixed-checkpoint evaluation and rapid retraining sweep"),
    ]:
        p = sub.add_parser(name, help=help_text, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        add_common_arguments(p)
    for name, help_text in [
        ("feature-ablation", "Measure which orbital features explain the baseline residual"),
        ("physics-probe", "Test PDE/data gradient alignment and paired physics-loss training"),
        ("ablation", "Run feature ablation and physics probing as one experiment"),
    ]:
        p = sub.add_parser(name, help=help_text, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        add_ablation_arguments(p)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "discover":
        result = discover_project(Path(args.root), args.horizon)
        print(json.dumps(asdict(result), indent=2, default=_json_default))
        return 0
    try:
        if args.command in {"feature-ablation", "physics-probe", "ablation"}:
            return run_integrated_ablation(args)
        return run_workflow(args)
    except KeyboardInterrupt:
        warn("Interrupted")
        return 130
    except Exception as exc:
        warn(str(exc))
        if os.environ.get("PINN_LAB_DEBUG", "0") == "1":
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
