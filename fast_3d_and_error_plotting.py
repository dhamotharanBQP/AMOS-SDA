import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plotting_common import (
    load_ephemeris,
    load_model,
    output_directory,
    predict_against_truth,
    satellite_truth,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Create 3D and error plots from a PINN checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="plots/trajectories")
    parser.add_argument("--satellites", nargs="*")
    parser.add_argument("--stats-dir")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=float)
    parser.add_argument("--points", type=int)
    parser.add_argument("--truth-source", choices=("auto", "csv", "orekit"), default="auto")
    parser.add_argument("--regime", choices=("LEO", "MEO", "GEO"))
    parser.add_argument("--use-j2-baseline", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-osc2mean", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-lat-features", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-orbit-features", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-orbit-phase-features", action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--orbit-harmonics", type=int, choices=(1, 2, 3, 4, 5, 6))
    parser.add_argument("--use-fourier-time", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fourier-n-freqs", type=int)
    parser.add_argument("--fourier-spacing", choices=("integer", "log", "geometric"))
    parser.add_argument("--fourier-max-freq", type=float)
    parser.add_argument("--fourier-learnable", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--per-sample-gate", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gate-power", type=float)
    parser.add_argument("--gate-kind", choices=("legacy", "orbit", "tanh"))
    parser.add_argument("--gate-tau-div", type=float)
    return parser.parse_args(argv)


def model_overrides(args):
    return {
        key: getattr(args, key)
        for key in (
            "regime",
            "use_j2_baseline",
            "use_osc2mean",
            "use_lat_features",
            "use_orbit_features",
            "use_orbit_phase_features",
            "orbit_harmonics",
            "use_fourier_time",
            "fourier_n_freqs",
            "fourier_spacing",
            "fourier_max_freq",
            "fourier_learnable",
            "per_sample_gate",
            "gate_power",
            "gate_kind",
            "gate_tau_div",
        )
    }


def stats_satellites(stats_dir):
    satellites = []
    for path in Path(stats_dir).rglob("stats.txt"):
        values = {}
        with path.open() as handle:
            for line in handle:
                if ":" in line:
                    key, value = line.split(":", 1)
                    values[key.strip()] = value.strip()
        if "satellite" in values:
            satellites.append(values["satellite"])
    if not satellites:
        raise ValueError(f"No stats.txt files were found under {stats_dir}")
    return satellites


def select_satellites(frame, requested, stats_dir, count, seed):
    available = frame["satellite"].drop_duplicates().astype(str).tolist()
    if requested:
        missing = sorted(set(map(str, requested)) - set(available))
        if missing:
            raise ValueError(f"Unknown satellite IDs: {', '.join(missing)}")
        candidates = list(map(str, requested))
    elif stats_dir:
        candidates = stats_satellites(stats_dir)
        candidates = [satellite for satellite in candidates if satellite in available]
        if not candidates:
            raise ValueError("Saved stats contain no satellites present in the CSV")
    else:
        candidates = available
    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    return rng.choice(candidates, size=min(count, len(candidates)), replace=False).tolist()


def save_plots(output_dir, satellite, times, predicted, truth, errors, truth_source):
    safe_id = str(satellite).replace("/", "_")
    scaled_predicted = predicted / 1e6
    scaled_truth = truth / 1e6

    fig = plt.figure(figsize=(8, 6))
    axis = fig.add_subplot(111, projection="3d")
    axis.plot(*scaled_predicted.T, color="tab:red", label="PINN")
    truth_label = "Orekit J2" if truth_source == "orekit" else "CSV truth"
    axis.plot(*scaled_truth.T, color="tab:blue", linestyle=":", label=truth_label)
    axis.set_xlabel("X [Mm]")
    axis.set_ylabel("Y [Mm]")
    axis.set_zlabel("Z [Mm]")
    axis.set_title(f"Trajectory: satellite {satellite}")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"sat_{safe_id}_3d.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for index, axis in enumerate(axes):
        axis.plot(times, scaled_predicted[:, index], color="tab:red", label="PINN")
        axis.plot(times, scaled_truth[:, index], color="tab:blue", linestyle=":", label=truth_label)
        axis.set_ylabel(f"{'XYZ'[index]} [Mm]")
        axis.grid(alpha=0.3)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"Position components: satellite {satellite}")
    fig.tight_layout()
    fig.savefig(output_dir / f"sat_{safe_id}_components.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(times, errors, color="tab:red")
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Position error [km]")
    axis.set_title(
        f"Satellite {satellite}: mean {errors.mean():.3f} km, max {errors.max():.3f} km"
    )
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"sat_{safe_id}_error.png", dpi=180)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    frame = load_ephemeris(args.data)
    model, settings = load_model(args.checkpoint, model_overrides(args))
    output_dir = output_directory(args.output_dir)
    satellites = select_satellites(
        frame, args.satellites, args.stats_dir, args.count, args.seed
    )
    print(f"[Load] checkpoint={args.checkpoint} settings={settings}")
    for satellite in satellites:
        times, positions, velocities, initial_timestamp = satellite_truth(frame, satellite)
        result = predict_against_truth(
            model,
            settings,
            times,
            positions,
            velocities,
            initial_timestamp,
            horizon=args.horizon,
            points=args.points,
            truth_source=args.truth_source,
        )
        pred_times, predicted, _, truth, _, errors, truth_source, _ = result
        save_plots(
            output_dir, satellite, pred_times, predicted, truth, errors, truth_source
        )
        print(
            f"[Plot] satellite={satellite} truth={truth_source} mean={errors.mean():.3f}km "
            f"max={errors.max():.3f}km"
        )
    print(f"[Done] {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
