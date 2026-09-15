"""
orekit_propagate.py

Standalone Orekit (Central + J2) propagation script.

Reads an input CSV with one state-vector per satellite, propagates each orbit
for a user-specified duration using Orekit's NumericalPropagator with the
J2OnlyPerturbation force model, and saves all propagated trajectories to an
output CSV in the same column format as the input.

Usage:
    python3 orekit_propagate.py \
        --input  geo_training_200.csv \
        --output propagated_output.csv \
        --duration 8000 \
        --step 10

Requirements:
    pip install orekit numpy pandas
    (The orekit Python wrapper must be installed with its data files.)
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# ── Orekit bootstrap ────────────────────────────────────────────────────────
import orekit

vm = orekit.initVM()

from orekit.pyhelpers import setup_orekit_curdir

setup_orekit_curdir(from_pip_library=True)

from org.orekit.time import TimeScalesFactory, AbsoluteDate
from org.orekit.frames import FramesFactory
from org.orekit.utils import PVCoordinates, IERSConventions, Constants
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.orbits import CartesianOrbit, OrbitType
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.numerical import NumericalPropagator
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.forces.gravity import J2OnlyPerturbation

# ── Physical constants (matching benchmarking.ipynb / pinn_lib_kepler.py) ───
MU_EARTH = 3.986004418e14       # m^3 / s^2
R_EARTH  = 6378137.0            # m
J2_COEFF = 1.08262668e-3

# ── Integrator defaults ─────────────────────────────────────────────────────
MIN_STEP  = 0.1    # s
MAX_STEP  = 60.0   # s
INIT_STEP = 10.0   # s
TOLERANCE = 1e-5

# ── CSV column names ────────────────────────────────────────────────────────
POS_COLS = ["x_eme2000_km", "y_eme2000_km", "z_eme2000_km"]
VEL_COLS = ["vx_eme2000_km_s", "vy_eme2000_km_s", "vz_eme2000_km_s"]


def parse_timestamp(ts_str: str) -> datetime:
    """Parse an ISO-8601-ish timestamp string into a timezone-aware datetime."""
    ts_str = ts_str.strip()
    # Handle trailing 'Z'
    if ts_str.endswith("Z"):
        ts_str = ts_str[:-1] + "+00:00"
    return datetime.fromisoformat(ts_str)


def datetime_to_orekit(dt: datetime) -> AbsoluteDate:
    """Convert a Python datetime to an Orekit AbsoluteDate (UTC)."""
    utc = TimeScalesFactory.getUTC()
    second_frac = dt.second + dt.microsecond / 1e6
    return AbsoluteDate(
        dt.year, dt.month, dt.day,
        dt.hour, dt.minute, second_frac,
        utc,
    )


def propagate_satellite(
    sat_id,
    epoch_dt: datetime,
    pos_km: np.ndarray,
    vel_km_s: np.ndarray,
    duration_s: float,
    step_s: float,
) -> pd.DataFrame:
    """
    Propagate a single satellite from its initial state vector using Central+J2.

    Parameters
    ----------
    sat_id : int or str
        Satellite identifier (written as-is into the output column).
    epoch_dt : datetime
        Epoch of the initial state vector.
    pos_km : ndarray of shape (3,)
        Initial position in km (EME2000).
    vel_km_s : ndarray of shape (3,)
        Initial velocity in km/s (EME2000).
    duration_s : float
        Total propagation time in seconds.
    step_s : float
        Time step between output rows in seconds.

    Returns
    -------
    pd.DataFrame
        Rows for this satellite in the standard CSV format.
    """
    # Convert km / km·s⁻¹ → m / m·s⁻¹ for Orekit
    pos_m   = pos_km * 1000.0
    vel_m_s = vel_km_s * 1000.0

    # Orekit objects
    inertial = FramesFactory.getEME2000()
    t0       = datetime_to_orekit(epoch_dt)

    p_vec = Vector3D(float(pos_m[0]), float(pos_m[1]), float(pos_m[2]))
    v_vec = Vector3D(float(vel_m_s[0]), float(vel_m_s[1]), float(vel_m_s[2]))
    pv    = PVCoordinates(p_vec, v_vec)

    cart_orbit    = CartesianOrbit(pv, inertial, t0, MU_EARTH)
    initial_state = SpacecraftState(cart_orbit)

    # Tolerances for Dormand-Prince 8(5,3)
    tols = NumericalPropagator.tolerances(TOLERANCE, cart_orbit, OrbitType.CARTESIAN)
    tol0 = orekit.JArray("double").cast_(tols[0])
    tol1 = orekit.JArray("double").cast_(tols[1])

    integrator = DormandPrince853Integrator(MIN_STEP, MAX_STEP, tol0, tol1)
    integrator.setInitialStepSize(INIT_STEP)

    propagator = NumericalPropagator(integrator)
    propagator.setOrbitType(OrbitType.CARTESIAN)
    propagator.setInitialState(initial_state)

    # J2 perturbation force model
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    j2_force = J2OnlyPerturbation(MU_EARTH, R_EARTH, J2_COEFF, itrf)
    propagator.addForceModel(j2_force)

    # Build time grid
    n_points = int(duration_s / step_s) + 1
    dt_values = np.linspace(0.0, duration_s, n_points)

    rows = []
    for dt in dt_values:
        state_i = propagator.propagate(t0.shiftedBy(float(dt)))
        pv_i    = state_i.getPVCoordinates(inertial)
        p_arr   = np.array(pv_i.getPosition().toArray())   # metres
        v_arr   = np.array(pv_i.getVelocity().toArray())    # m/s

        # Convert back to km / km·s⁻¹
        p_km   = p_arr / 1000.0
        v_km_s = v_arr / 1000.0

        # Compute timestamp
        ts = epoch_dt + timedelta(seconds=float(dt))
        ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        rows.append(
            {
                "satellite": sat_id,
                "timestamp": ts_str,
                "x_eme2000_km": p_km[0],
                "y_eme2000_km": p_km[1],
                "z_eme2000_km": p_km[2],
                "vx_eme2000_km_s": v_km_s[0],
                "vy_eme2000_km_s": v_km_s[1],
                "vz_eme2000_km_s": v_km_s[2],
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Propagate satellite state vectors using Orekit (Central + J2)."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input CSV (one row per satellite).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path for the output CSV with propagated trajectories.",
    )
    parser.add_argument(
        "--duration", type=float, required=True,
        help="Total propagation duration in seconds.",
    )
    parser.add_argument(
        "--step", type=float, default=10.0,
        help="Time step between output points in seconds (default: 10).",
    )
    args = parser.parse_args()

    # ── Load input ──────────────────────────────────────────────────────────
    input_df = pd.read_csv(args.input)
    n_sats = len(input_df)
    print(f"[Info] Loaded {n_sats} satellite(s) from {args.input}")
    print(f"[Info] Propagation: {args.duration:.0f}s total, {args.step:.1f}s step")

    all_frames: list[pd.DataFrame] = []

    # ── Make satellite IDs unique ──────────────────────────────────────────
    original_ids = list(input_df["satellite"])
    all_existing = set(str(sid) for sid in original_ids)
    seen_count: dict[str, int] = {}
    unique_ids: list[str] = []

    for sid in original_ids:
        sid_str = str(sid)
        if sid_str not in seen_count:
            seen_count[sid_str] = 1
            unique_ids.append(sid_str)
        else:
            seen_count[sid_str] += 1
            suffix = seen_count[sid_str]
            candidate = f"{sid_str}{suffix}"
            while candidate in all_existing:
                suffix += 1
                candidate = f"{sid_str}{suffix}"
            seen_count[sid_str] = suffix
            all_existing.add(candidate)
            unique_ids.append(candidate)

    for idx, row in input_df.iterrows():
        sat_id   = unique_ids[idx]
        epoch_dt = parse_timestamp(str(row["timestamp"]))
        pos_km   = np.array([row[c] for c in POS_COLS], dtype=np.float64)
        vel_km_s = np.array([row[c] for c in VEL_COLS], dtype=np.float64)

        print(f"  [{idx+1}/{n_sats}] Propagating sat {sat_id} from {epoch_dt.isoformat()} …", end=" ")
        sys.stdout.flush()

        df_sat = propagate_satellite(
            sat_id=sat_id,
            epoch_dt=epoch_dt,
            pos_km=pos_km,
            vel_km_s=vel_km_s,
            duration_s=args.duration,
            step_s=args.step,
        )
        all_frames.append(df_sat)
        print("done")

    # ── Write output ────────────────────────────────────────────────────────
    output_df = pd.concat(all_frames, ignore_index=True)
    output_df.to_csv(args.output, index=False)
    print(f"[Done] Wrote {len(output_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
