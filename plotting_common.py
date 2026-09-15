from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

import pinn_lib_kepler as lib


POSITION_COLUMNS = ["x_eme2000_km", "y_eme2000_km", "z_eme2000_km"]
VELOCITY_COLUMNS = ["vx_eme2000_km_s", "vy_eme2000_km_s", "vz_eme2000_km_s"]


def load_ephemeris(path):
    frame = pd.read_csv(path)
    required = {"timestamp", "satellite", *POSITION_COLUMNS, *VELOCITY_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required CSV columns: {', '.join(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    return frame.sort_values(["satellite", "timestamp"])


def _state_dict_and_config(payload):
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"], dict(payload.get("config", {}))
    if isinstance(payload, dict) and payload and all(
        isinstance(value, torch.Tensor) for value in payload.values()
    ):
        return payload, {}
    raise ValueError("Checkpoint is neither a training checkpoint nor a state_dict")


def _layers_from_state_dict(state_dict):
    weights = []
    for key, value in state_dict.items():
        if key.startswith("linear.") and key.endswith(".weight"):
            index = int(key.split(".")[1])
            weights.append((index, value))
    if not weights:
        raise ValueError("Checkpoint does not contain Net linear-layer weights")
    weights.sort()
    return [weights[0][1].shape[1], *[weight.shape[0] for _, weight in weights]]


def configure_library(config, overrides=None):
    values = {
        "regime": "LEO",
        "use_j2_baseline": False,
        "use_osc2mean": False,
        "use_lat_features": False,
        "use_orbit_features": False,
        "use_orbit_phase_features": True,
        "orbit_harmonics": 1,
        "use_fourier_time": False,
        "fourier_n_freqs": 4,
        "fourier_spacing": "integer",
        "fourier_max_freq": 50.0,
        "fourier_learnable": False,
        "gate_power": 3.0,
        "gate_kind": "legacy",
        "gate_tau_div": 3.0,
        "per_sample_gate": True,
        "use_j2_physics": False,
        "pde_norm": "total",
        "pde_gate_weighted": False,
        "pde_subtract_kepler": False,
        "use_quantum_input": False,
        "use_quantum_hidden": False,
        "quantum_hidden_position": 1,
        "quantum_qubits": 9,
        "quantum_depth": 4,
        "quantum_batch_size": 0,
    }
    values.update({key: config[key] for key in values if key in config})
    if overrides:
        values.update({key: value for key, value in overrides.items() if value is not None})
    if int(values["orbit_harmonics"]) < 1:
        raise ValueError("orbit_harmonics must be at least 1")
    if not bool(values["use_orbit_phase_features"]) and not bool(values["use_orbit_features"]):
        raise ValueError("Static-only orbit phase mode requires use_orbit_features")
    if int(values["fourier_n_freqs"]) < 1:
        raise ValueError("fourier_n_freqs must be at least 1")
    if values["fourier_spacing"] not in {"integer", "log", "geometric"}:
        raise ValueError(f"Unsupported Fourier spacing {values['fourier_spacing']!r}")
    if values["fourier_spacing"] == "geometric" and float(values["fourier_max_freq"]) <= 1:
        raise ValueError("fourier_max_freq must exceed 1 for geometric spacing")
    if values["gate_kind"] not in {"legacy", "orbit", "tanh"}:
        raise ValueError(f"Unsupported gate kind {values['gate_kind']!r}")
    if bool(values["use_quantum_input"]) and bool(values["use_quantum_hidden"]):
        raise ValueError("A checkpoint cannot enable both quantum network variants")
    window = float(config.get("window_duration", 0.0))
    lib.set_regime(values["regime"], window_duration_sec=window if window > 0 else None)
    lib.USE_J2_BASELINE = bool(values["use_j2_baseline"])
    lib.USE_OSC2MEAN = bool(values["use_osc2mean"])
    lib.USE_LAT_FEATURES = bool(values["use_lat_features"])
    lib.USE_ORBIT_FEATURES = bool(values["use_orbit_features"])
    lib.USE_ORBIT_PHASE_FEATURES = bool(values["use_orbit_phase_features"])
    lib.ORBIT_N_HARMONICS = int(values["orbit_harmonics"])
    lib.USE_FOURIER_TIME = bool(values["use_fourier_time"])
    lib.FOURIER_N_FREQS = int(values["fourier_n_freqs"])
    lib.FOURIER_SPACING = str(values["fourier_spacing"])
    lib.FOURIER_MAX_FREQ = float(values["fourier_max_freq"])
    lib.FOURIER_LEARNABLE = bool(values["fourier_learnable"])
    lib.GATE_POWER = float(values["gate_power"])
    lib.GATE_KIND = str(values["gate_kind"])
    lib.GATE_TAU_DIV = float(values["gate_tau_div"])
    lib.USE_J2 = bool(values["use_j2_physics"])
    lib.PDE_NORM = str(values["pde_norm"])
    lib.PDE_GATE_WEIGHTED = bool(values["pde_gate_weighted"])
    lib.PDE_SUBTRACT_KEPLER = bool(values["pde_subtract_kepler"])
    lib.USE_QUANTUM_INPUT = bool(values["use_quantum_input"])
    lib.USE_QUANTUM_HIDDEN = bool(values["use_quantum_hidden"])
    lib.QUANTUM_HIDDEN_POSITION = int(values["quantum_hidden_position"])
    lib.QUANTUM_N_QUBITS = int(values["quantum_qubits"])
    lib.QUANTUM_DEPTH = int(values["quantum_depth"])
    quantum_batch_size = int(values["quantum_batch_size"] or 0)
    lib.QUANTUM_BATCH_SIZE = quantum_batch_size or None
    return values


def load_model(checkpoint_path, overrides=None):
    payload = torch.load(checkpoint_path, map_location=lib.DEVICE, weights_only=False)
    state_dict, config = _state_dict_and_config(payload)
    layers = config.get("layers") or _layers_from_state_dict(state_dict)
    input_dim = int(layers[0])
    if "use_orbit_features" not in config or "use_lat_features" not in config:
        inferred_features = {
            9: (False, False),
            11: (True, False),
            14: (False, True),
            16: (True, True),
        }
        if input_dim not in inferred_features:
            raise ValueError(
                "A raw legacy checkpoint without feature metadata can only be inferred "
                f"for input widths 9, 11, 14, or 16; got {input_dim}. Use a structured "
                "checkpoint containing its training config."
            )
        inferred_lat, inferred_orbit = inferred_features[input_dim]
        config.setdefault("use_lat_features", inferred_lat)
        config.setdefault("use_orbit_features", inferred_orbit)
    settings = configure_library(config, overrides)
    expected_dim = lib.input_dim()
    if expected_dim != input_dim:
        raise ValueError(
            f"Checkpoint expects {input_dim} inputs, but the selected feature flags produce "
            f"{expected_dim}"
        )
    net = lib.build_net([int(value) for value in layers])
    lib.attach_fourier(net)
    net.load_state_dict(state_dict)
    model = lib.GeneralistModel(
        net,
        W_DATA=float(config.get("w_data", 1.0)),
        W_PDE=float(config.get("w_pde", 0.0)),
    )
    model.net.eval()
    settings["window_duration"] = float(config.get("window_duration", 0.0))
    settings["layers"] = list(layers)
    return model, settings


def satellite_truth(frame, satellite):
    sat = frame[frame["satellite"].astype(str) == str(satellite)].copy()
    if sat.empty:
        raise ValueError(f"Satellite {satellite!r} is not present in the CSV")
    sat = sat.sort_values("timestamp")
    times = (sat["timestamp"] - sat["timestamp"].iloc[0]).dt.total_seconds().to_numpy()
    positions = sat[POSITION_COLUMNS].to_numpy(dtype=np.float64) * 1000.0
    velocities = sat[VELOCITY_COLUMNS].to_numpy(dtype=np.float64) * 1000.0
    return times, positions, velocities, sat["timestamp"].iloc[0]


def _orekit_positions(initial_timestamp, position, velocity, times):
    try:
        import orekit
        from orekit.pyhelpers import setup_orekit_curdir
    except ImportError as exc:
        raise RuntimeError(
            "Orekit truth was requested but the Python Orekit package is unavailable"
        ) from exc

    try:
        vm = orekit.getVMEnv()
    except Exception:
        vm = None
    if vm is None:
        orekit.initVM()
        setup_orekit_curdir(from_pip_library=True)

    try:
        from org.hipparchus.geometry.euclidean.threed import Vector3D
        from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
        from org.orekit.forces.gravity import J2OnlyPerturbation
        from org.orekit.frames import FramesFactory
        from org.orekit.orbits import CartesianOrbit, OrbitType
        from org.orekit.propagation import SpacecraftState
        from org.orekit.propagation.numerical import NumericalPropagator
        from org.orekit.time import AbsoluteDate, TimeScalesFactory
        from org.orekit.utils import IERSConventions, PVCoordinates
    except ImportError as exc:
        raise RuntimeError(
            "Orekit truth was requested but the Python Orekit package is unavailable"
        ) from exc

    timestamp = pd.Timestamp(initial_timestamp)
    utc = TimeScalesFactory.getUTC()
    epoch = AbsoluteDate(
        timestamp.year,
        timestamp.month,
        timestamp.day,
        timestamp.hour,
        timestamp.minute,
        timestamp.second + timestamp.microsecond / 1e6,
        utc,
    )
    inertial = FramesFactory.getEME2000()
    pv = PVCoordinates(
        Vector3D(*map(float, position)),
        Vector3D(*map(float, velocity)),
    )
    orbit = CartesianOrbit(pv, inertial, epoch, lib.MU_EARTH)
    tolerances = NumericalPropagator.tolerances(1e-5, orbit, OrbitType.CARTESIAN)
    absolute_tolerance = orekit.JArray("double").cast_(tolerances[0])
    relative_tolerance = orekit.JArray("double").cast_(tolerances[1])
    integrator = DormandPrince853Integrator(
        0.1, 60.0, absolute_tolerance, relative_tolerance
    )
    integrator.setInitialStepSize(10.0)
    propagator = NumericalPropagator(integrator)
    propagator.setOrbitType(OrbitType.CARTESIAN)
    propagator.setInitialState(SpacecraftState(orbit))
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    propagator.addForceModel(
        J2OnlyPerturbation(lib.MU_EARTH, lib.R_EARTH, lib.J2, itrf)
    )

    truth_positions = np.empty((len(times), 3), dtype=np.float64)
    truth_velocities = np.empty((len(times), 3), dtype=np.float64)
    for index, offset in enumerate(times):
        state = propagator.propagate(epoch.shiftedBy(float(offset)))
        coordinates = state.getPVCoordinates(inertial)
        truth_positions[index] = coordinates.getPosition().toArray()
        truth_velocities[index] = coordinates.getVelocity().toArray()
    return truth_positions, truth_velocities


def baseline_positions(settings, position, velocity, times):
    x0 = torch.as_tensor(position, dtype=lib.DTYPE, device=lib.DEVICE).view(1, 3)
    u0 = torch.as_tensor(velocity, dtype=lib.DTYPE, device=lib.DEVICE).view(1, 3)
    time_tensor = torch.as_tensor(
        np.asarray(times).reshape(-1, 1), dtype=lib.DTYPE, device=lib.DEVICE
    )
    x0_batch = x0.expand(len(times), -1)
    u0_batch = u0.expand(len(times), -1)
    propagator = (
        lib._kepler_j2_secular_propagate_torch
        if settings["use_j2_baseline"]
        else lib._kepler_propagate_torch
    )
    with torch.no_grad():
        positions = propagator(x0_batch, u0_batch, time_tensor)
    return positions.detach().cpu().numpy().astype(np.float64)


def predict_against_truth(
    model,
    settings,
    times,
    positions,
    velocities,
    initial_timestamp,
    horizon=None,
    points=None,
    truth_source="auto",
):
    if points is not None and int(points) < 2:
        raise ValueError("points must be at least 2")
    csv_horizon = float(times[-1]) if len(times) > 1 else 0.0
    source = truth_source
    if source == "auto":
        source = "csv" if csv_horizon > 0 else "orekit"
    if source not in {"csv", "orekit"}:
        raise ValueError("truth_source must be 'auto', 'csv', or 'orekit'")

    if source == "csv":
        available_horizon = csv_horizon
        if available_horizon <= 0:
            raise ValueError(
                "CSV truth requires at least two distinct timestamps per satellite; "
                "use --truth-source orekit for initial-condition-only data"
            )
        duration = available_horizon if horizon is None else min(float(horizon), available_horizon)
    else:
        duration = float(horizon or settings["window_duration"])
        if duration <= 0:
            raise ValueError(
                "Orekit truth requires --horizon when the checkpoint has no saved training window"
            )

    if duration <= 0:
        raise ValueError("A satellite must contain at least two distinct timestamps")
    if points is None:
        cadence = np.median(np.diff(times)) if source == "csv" else 10.0
        points = max(2, int(round(duration / max(cadence, 1e-6))) + 1)
    training_window = settings["window_duration"] or duration
    gate_window = training_window if settings["per_sample_gate"] else None
    pinn_started = time.perf_counter()
    pred_times, pred_positions, pred_velocities = lib.predict_step(
        model,
        positions[0],
        velocities[0],
        T_total=duration,
        npts=int(points),
        Tseg_train=gate_window,
    )
    pred_times = np.asarray(pred_times).reshape(-1)
    pred_positions = np.asarray(pred_positions)
    pred_velocities = np.asarray(pred_velocities)
    pinn_time = time.perf_counter() - pinn_started
    truth_started = time.perf_counter()
    if source == "csv":
        truth_positions = np.column_stack(
            [np.interp(pred_times, times, positions[:, index]) for index in range(3)]
        )
        truth_velocities = np.column_stack(
            [np.interp(pred_times, times, velocities[:, index]) for index in range(3)]
        )
    else:
        truth_positions, truth_velocities = _orekit_positions(
            initial_timestamp, positions[0], velocities[0], pred_times
        )
    errors_km = np.linalg.norm(pred_positions - truth_positions, axis=1) / 1000.0
    truth_time = time.perf_counter() - truth_started
    return (
        pred_times,
        pred_positions,
        pred_velocities,
        truth_positions,
        truth_velocities,
        errors_km,
        source,
        {"pinn_time_sec": pinn_time, "truth_time_sec": truth_time},
    )


def output_directory(path):
    result = Path(path).expanduser().resolve()
    result.mkdir(parents=True, exist_ok=True)
    return result
