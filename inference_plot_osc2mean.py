import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting_common import (
    baseline_positions,
    load_ephemeris,
    load_model,
    output_directory,
    predict_against_truth,
    satellite_truth,
)


EARTH_RADIUS_KM = 6378.137


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a PINN checkpoint over an ephemeris CSV")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="plots/inference")
    parser.add_argument("--horizon", type=float)
    parser.add_argument("--points", type=int)
    parser.add_argument("--truth-source", choices=("auto", "csv", "orekit"), default="auto")
    parser.add_argument(
        "--make-propagation-plots", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--max-satellites", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
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


def orbital_properties(position, velocity):
    radius_km = np.linalg.norm(position) / 1000.0
    angular_momentum = np.cross(position, velocity)
    norm = np.linalg.norm(angular_momentum)
    inclination = np.degrees(
        np.arccos(np.clip(angular_momentum[2] / max(norm, 1e-12), -1.0, 1.0))
    )
    return radius_km - EARTH_RADIUS_KM, inclination


def binned_mean(x, y, values, x_edges, y_edges):
    result = np.full((len(y_edges) - 1, len(x_edges) - 1), np.nan)
    x_bin = np.digitize(x, x_edges) - 1
    y_bin = np.digitize(y, y_edges) - 1
    for yi in range(result.shape[0]):
        for xi in range(result.shape[1]):
            selected = (x_bin == xi) & (y_bin == yi)
            if np.any(selected):
                result[yi, xi] = np.mean(values[selected])
    return result


def save_summary_plots(results, output_dir):
    altitude = results["altitude_km"].to_numpy()
    inclination = results["inclination_deg"].to_numpy()
    mean_error = results["mean_error_km"].to_numpy()
    max_error = results["max_error_km"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, values, title in zip(
        axes, (mean_error, max_error), ("Mean position error", "Maximum position error")
    ):
        points = axis.scatter(altitude, inclination, c=values, cmap="viridis", s=45)
        axis.set_xlabel("Altitude [km]")
        axis.set_ylabel("Inclination [deg]")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        fig.colorbar(points, ax=axis, label="Error [km]")
    fig.tight_layout()
    fig.savefig(output_dir / "inference_scatter.png", dpi=180)
    plt.close(fig)

    alt_min, alt_max = float(np.min(altitude)), float(np.max(altitude))
    inc_min, inc_max = float(np.min(inclination)), float(np.max(inclination))
    alt_edges = np.linspace(alt_min - 1e-6, alt_max + 1e-6, min(12, len(results) + 1))
    inc_edges = np.linspace(inc_min - 1e-6, inc_max + 1e-6, min(12, len(results) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, values, title in zip(
        axes, (mean_error, max_error), ("Binned mean error", "Binned maximum error")
    ):
        grid = binned_mean(altitude, inclination, values, alt_edges, inc_edges)
        image = axis.imshow(
            grid,
            origin="lower",
            aspect="auto",
            extent=(alt_edges[0], alt_edges[-1], inc_edges[0], inc_edges[-1]),
            cmap="viridis",
        )
        axis.set_xlabel("Altitude [km]")
        axis.set_ylabel("Inclination [deg]")
        axis.set_title(title)
        fig.colorbar(image, ax=axis, label="Error [km]")
    fig.tight_layout()
    fig.savefig(output_dir / "inference_heatmap.png", dpi=180)
    plt.close(fig)


def save_satellite_stats(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for key, value in row.items():
            if isinstance(value, float):
                handle.write(f"{key}: {value:.6f}\n")
            else:
                handle.write(f"{key}: {value}\n")


def save_propagation_plots(
    satellite, times, predicted, truth, errors, baseline_errors, truth_source, output_dir
):
    label = "Orekit J2" if truth_source == "orekit" else "CSV truth"
    scaled_predicted = predicted / 1e6
    scaled_truth = truth / 1e6

    fig = plt.figure(figsize=(8, 6))
    axis = fig.add_subplot(111, projection="3d")
    axis.plot(*scaled_predicted.T, color="tab:red", label="PINN")
    axis.plot(*scaled_truth.T, color="tab:blue", linestyle=":", label=label)
    axis.set_xlabel("X [Mm]")
    axis.set_ylabel("Y [Mm]")
    axis.set_zlabel("Z [Mm]")
    axis.set_title(f"Trajectory: satellite {satellite}")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "01_3d.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for index, axis in enumerate(axes):
        axis.plot(times, scaled_predicted[:, index], color="tab:red", label="PINN")
        axis.plot(times, scaled_truth[:, index], color="tab:blue", linestyle=":", label=label)
        axis.set_ylabel(f"{'XYZ'[index]} [Mm]")
        axis.grid(alpha=0.3)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"Position components: satellite {satellite}")
    fig.tight_layout()
    fig.savefig(output_dir / "02_components.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(times, baseline_errors, color="tab:red", label="Analytical baseline")
    axis.plot(times, errors, color="tab:blue", label="PINN")
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Position error [km]")
    axis.set_title(f"Position error: satellite {satellite}")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "03_deviation.png", dpi=180)
    plt.close(fig)


def print_summary(results):
    mean_values = results["mean_dev_km"].to_numpy()
    max_values = results["max_dev_km"].to_numpy()
    print(f"[Summary] satellites={len(results)}")
    print(
        f"[Summary] mean deviation: min={mean_values.min():.3f}km "
        f"max={mean_values.max():.3f}km avg={mean_values.mean():.3f}km"
    )
    print(
        f"[Summary] max deviation: min={max_values.min():.3f}km "
        f"max={max_values.max():.3f}km avg={max_values.mean():.3f}km"
    )
    for label, lower, upper in (
        ("equatorial", 0.0, 20.0),
        ("mid", 20.0, 70.0),
        ("polar/retro", 70.0, 110.0),
    ):
        band = results[
            (results["inclination_deg"] >= lower)
            & (results["inclination_deg"] < upper)
        ]
        if band.empty:
            print(f"[Summary] {label}: no satellites")
        else:
            print(
                f"[Summary] {label}: n={len(band)} "
                f"mean={band['mean_dev_km'].mean():.3f}km "
                f"max={band['max_dev_km'].mean():.3f}km "
                f"gap_closed={band['gap_closed_pct'].mean():.1f}%"
            )


def main(argv=None):
    args = parse_args(argv)
    if args.max_satellites < 0:
        raise ValueError("max-satellites cannot be negative")
    frame = load_ephemeris(args.data)
    model, settings = load_model(args.checkpoint, model_overrides(args))
    output_dir = output_directory(args.output_dir)
    satellites = frame["satellite"].drop_duplicates().astype(str).tolist()
    if args.max_satellites > 0 and args.max_satellites < len(satellites):
        rng = np.random.default_rng(args.seed)
        satellites = rng.choice(
            satellites, size=args.max_satellites, replace=False
        ).tolist()

    print(f"[Load] checkpoint={args.checkpoint} settings={settings}")
    rows = []
    for satellite in satellites:
        try:
            started = time.perf_counter()
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
            pred_times, predicted, _, truth, _, errors, truth_source, timings = result
            baseline = baseline_positions(
                settings, positions[0], velocities[0], pred_times
            )
            baseline_errors = np.linalg.norm(baseline - truth, axis=1) / 1000.0
            baseline_mean = float(baseline_errors.mean())
            gap_closed = 100.0 * (1.0 - float(errors.mean()) / max(baseline_mean, 1e-12))
            altitude, inclination = orbital_properties(positions[0], velocities[0])
            row = {
                "satellite": satellite,
                "truth_source": truth_source,
                "altitude_km": altitude,
                "inclination_deg": inclination,
                "horizon_sec": float(pred_times[-1]),
                "mean_dev_km": float(errors.mean()),
                "max_dev_km": float(errors.max()),
                "final_error_km": float(errors[-1]),
                "max_dev_time_s": float(pred_times[np.argmax(errors)]),
                "baseline_mean_dev_km": baseline_mean,
                "baseline_max_dev_km": float(baseline_errors.max()),
                "gap_closed_pct": gap_closed,
                "pinn_time_sec": timings["pinn_time_sec"],
                "truth_time_sec": timings["truth_time_sec"],
                "orekit_time_sec": (
                    timings["truth_time_sec"] if truth_source == "orekit" else 0.0
                ),
                "inference_sec": time.perf_counter() - started,
            }
            rows.append(row)
            satellite_dir = output_dir / f"sat_{str(satellite).replace('/', '_')}"
            save_satellite_stats(satellite_dir / "stats.txt", row)
            if args.make_propagation_plots:
                save_propagation_plots(
                    satellite,
                    pred_times,
                    predicted,
                    truth,
                    errors,
                    baseline_errors,
                    truth_source,
                    satellite_dir,
                )
            print(
                f"[Eval] satellite={satellite} truth={truth_source} "
                f"mean={errors.mean():.3f}km max={errors.max():.3f}km "
                f"gap_closed={gap_closed:.1f}%"
            )
        except Exception as exc:
            print(f"[Error] satellite={satellite}: {exc}")

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No satellites were evaluated successfully")
    results["mean_error_km"] = results["mean_dev_km"]
    results["max_error_km"] = results["max_dev_km"]
    results.to_csv(output_dir / "inference_metrics.csv", index=False)
    save_summary_plots(results, output_dir)
    print_summary(results)
    print(f"[Done] evaluated={len(results)} output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
