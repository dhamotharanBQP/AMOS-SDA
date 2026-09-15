"""
train_opti.py -- trainer for the generalist Kepler/J2 residual PINN.

Supports, all opt-in and all recorded in config.json:
  * orbital regimes            LEO / MEO / GEO
  * orbit harmonics            sin ku, cos ku for k = 1..N
  * Fourier time embedding     sin/cos of t/T_orb at K frequencies
  * latitude features          z/r, (z/r)^2
  * gate variants              legacy / orbit / tanh
  * physics branch             J2 target, normalisation, gate weighting,
                               a_kepler(r_base) residual form
  * quantum layer (QAPINN)     classical by default; two placements available
                               variant A: --use-quantum-input
                               variant B: --use-quantum-hidden

The two headline experiments:

    run A, harmonics:   --use-orbit-features --orbit-harmonics 4
    run B, Fourier:     --use-fourier-time --fourier-n-freqs 4

Input width is never computed here; it comes from lib.input_dim(), so the
trainer, the evaluator and the checkpoint can never disagree about it.
"""

import argparse
import json
import random
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import pinn_lib_kepler as lib
from ic_loader import EphemerisDataset, ephemeris_collate_fn


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Ephemeris CSV used for training")
    p.add_argument("--output-dir", default="training_output")
    p.add_argument("--resume", help="Checkpoint to resume from")
    p.add_argument("--epochs", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--samples-per-epoch", type=int, default=12000)
    p.add_argument("--window-duration", type=float, default=80000.0)
    p.add_argument("--pde-points", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--w-data", type=float, default=1.0)
    p.add_argument("--w-pde", type=float, default=0.0)
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--regime", choices=("LEO", "MEO", "GEO"), default="LEO",
                   help="Sets V0 normalisation and the legacy gate's T_SCALED_MAX")

    p.add_argument("--use-j2-baseline", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--use-osc2mean", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--use-lat-features", action=argparse.BooleanOptionalAction, default=False,
                   help="z/r and (z/r)^2")
    p.add_argument("--use-orbit-features", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--use-orbit-phase-features", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Include sin(ku)/cos(ku); disable for static inclination/eccentricity only")
    p.add_argument("--orbit-harmonics", type=int, default=1, choices=(1, 2, 3, 4, 5, 6),
                   help="sin ku, cos ku for k=1..N. Requires --use-orbit-features")
    p.add_argument("--use-fourier-time", action=argparse.BooleanOptionalAction, default=False,
                   help="Fourier embedding of t/T_orb")
    p.add_argument("--fourier-n-freqs", type=int, default=4)
    p.add_argument("--fourier-spacing", choices=("integer", "log", "geometric"),
                   default="integer",
                   help="integer (1,2,3,..) measured best; log (1,2,4,8) skips k=3; "
                        "geometric (1..max-freq) matches the polar reference")
    p.add_argument("--fourier-max-freq", type=float, default=50.0,
                   help="Top frequency for geometric spacing. Must exceed 1")
    p.add_argument("--fourier-learnable", action=argparse.BooleanOptionalAction,
                   default=False, help="Train the Fourier frequencies")

    p.add_argument("--per-sample-gate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gate-power", type=float, default=3.0)
    p.add_argument("--gate-kind", choices=("legacy", "orbit", "tanh"), default="legacy")
    p.add_argument("--gate-tau-div", type=float, default=3.0)

    p.add_argument("--use-j2-physics", action=argparse.BooleanOptionalAction, default=False,
                   help="Include J2 in the PDE target. Without it the target is pure two-body")
    p.add_argument("--pde-norm", choices=("total", "j2"), default="total")
    p.add_argument("--pde-gate-weighted", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--pde-subtract-kepler", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=5)
    # Two mutually exclusive quantum placements, both off by default.
    p.add_argument("--use-quantum-input", action=argparse.BooleanOptionalAction,
                   default=False, help="Variant A: circuit is the input layer")
    p.add_argument("--use-quantum-hidden", action=argparse.BooleanOptionalAction,
                   default=False, help="Variant B: circuit is a hidden layer")
    p.add_argument("--quantum-hidden-position", type=int, default=1,
                   help="Which layer index the circuit replaces in variant B")
    p.add_argument("--quantum-qubits", type=int, default=9)
    p.add_argument("--quantum-depth", type=int, default=4,
                   help="BasicEntanglerLayers repetitions")
    p.add_argument("--quantum-batch-size", type=int, default=0,
                   help="0 = one broadcast call; set >0 only to bound simulator memory")
    return p.parse_args(argv)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate(args):
    if args.epochs <= 0 or args.checkpoint_every <= 0:
        raise ValueError("epochs and checkpoint-every must be positive")
    if args.batch_size <= 0 or args.samples_per_epoch <= 0:
        raise ValueError("batch-size and samples-per-epoch must be positive")
    if args.window_duration <= 0 or args.pde_points < 0:
        raise ValueError("window-duration must be positive and pde-points cannot be negative")
    if args.width <= 0 or args.depth <= 0:
        raise ValueError("width and depth must be positive")
    if args.quantum_qubits < 2 or args.quantum_depth <= 0 or args.quantum_batch_size < 0:
        raise ValueError("quantum-qubits must be at least 2, quantum-depth must be positive, "
                         "and quantum-batch-size cannot be negative")
    if args.orbit_harmonics != 1 and not args.use_orbit_features:
        raise ValueError("--orbit-harmonics > 1 requires --use-orbit-features")
    if args.orbit_harmonics != 1 and not args.use_orbit_phase_features:
        raise ValueError("--orbit-harmonics has no effect with --no-use-orbit-phase-features; "
                         "leave it at 1")
    if not args.use_orbit_phase_features and not args.use_orbit_features:
        raise ValueError("--no-use-orbit-phase-features requires --use-orbit-features")
    if args.fourier_n_freqs < 1:
        raise ValueError("--fourier-n-freqs must be at least 1")
    if args.pde_norm == "j2" and not args.use_j2_physics:
        raise ValueError("--pde-norm j2 requires --use-j2-physics: with J2 off the "
                         "reference acceleration is identically zero and the loss diverges")
    if args.pde_subtract_kepler and args.w_pde <= 0:
        raise ValueError("--pde-subtract-kepler has no effect with --w-pde 0")
    if args.use_quantum_input and args.use_quantum_hidden:
        raise ValueError("--use-quantum-input and --use-quantum-hidden are mutually "
                         "exclusive: the circuit occupies exactly one layer")
    if args.fourier_spacing == "geometric" and args.fourier_max_freq <= 1.0:
        raise ValueError("--fourier-max-freq must exceed 1: at 1.0 every frequency is "
                         "identical and the embedding collapses to one sin/cos pair")
    if args.fourier_learnable and not args.use_fourier_time:
        raise ValueError("--fourier-learnable requires --use-fourier-time")

    if args.w_pde > 0 and not args.use_j2_physics:
        print("[Warn] w-pde > 0 without --use-j2-physics: the PDE target is pure two-body, "
              "which the Kepler baseline already satisfies. Measured usable range is 3.2x "
              "versus 89000x with J2 enabled.")
    if args.w_pde > 0 and args.use_orbit_features and args.orbit_harmonics == 1:
        print("[Warn] Physics gradient measured NEGATIVE (cos -0.29 to -0.44) with a single "
              "harmonic. Raise --orbit-harmonics before enabling W_PDE.")
    if args.use_lat_features and args.use_orbit_features and args.orbit_harmonics >= 2:
        print("[Info] Latitude measured to contribute 0.000 km on top of harmonics for the "
              "POSITION target (z/r = sin i sin u). It measured non-trivial for the "
              "ODE-violation target, so it may still matter with physics on.")
    if args.use_fourier_time and args.fourier_spacing == "log":
        print("[Info] Log spacing (1,2,4,8) skips k=3 and measured worse than integer "
              "spacing (93.4 percent vs 99.1 percent). Consider --fourier-spacing integer.")
    if args.use_quantum_input or args.use_quantum_hidden:
        variant = "A (input layer)" if args.use_quantum_input else "B (hidden layer)"
        print(f"[Info] Quantum variant {variant} enabled, {args.quantum_qubits} qubits, "
              f"readout width {2 ** (args.quantum_qubits - 1)}. Uses default.qubit with "
              f"diff_method='backprop', which broadcasts over the batch. Still far slower "
              f"than the classical path; time one epoch before committing a full run.")


def configure_library(args):
    lib.set_regime(args.regime, window_duration_sec=args.window_duration)
    lib.USE_J2_BASELINE = args.use_j2_baseline
    lib.USE_OSC2MEAN = args.use_osc2mean
    lib.USE_LAT_FEATURES = args.use_lat_features
    lib.USE_ORBIT_FEATURES = args.use_orbit_features
    lib.USE_ORBIT_PHASE_FEATURES = args.use_orbit_phase_features
    lib.ORBIT_N_HARMONICS = args.orbit_harmonics
    lib.USE_FOURIER_TIME = args.use_fourier_time
    lib.FOURIER_N_FREQS = args.fourier_n_freqs
    lib.FOURIER_SPACING = args.fourier_spacing
    lib.FOURIER_MAX_FREQ = args.fourier_max_freq
    lib.FOURIER_LEARNABLE = args.fourier_learnable
    lib.GATE_POWER = args.gate_power
    lib.GATE_KIND = args.gate_kind
    lib.GATE_TAU_DIV = args.gate_tau_div
    lib.USE_J2 = args.use_j2_physics
    lib.PDE_NORM = args.pde_norm
    lib.PDE_GATE_WEIGHTED = args.pde_gate_weighted
    lib.PDE_SUBTRACT_KEPLER = args.pde_subtract_kepler
    lib.USE_QUANTUM_INPUT = args.use_quantum_input
    lib.USE_QUANTUM_HIDDEN = args.use_quantum_hidden
    lib.QUANTUM_HIDDEN_POSITION = args.quantum_hidden_position
    lib.QUANTUM_N_QUBITS = args.quantum_qubits
    lib.QUANTUM_DEPTH = args.quantum_depth
    lib.QUANTUM_BATCH_SIZE = args.quantum_batch_size or None


def model_layers(args):
    """Widths from lib.input_dim(), never recomputed locally.

    The circuit emits probs over n_qubits-1 wires, so whichever layer it
    occupies must be exactly 2**(n_qubits-1) wide. That width is substituted
    at the right index automatically; --width governs every other layer.
    """
    dim = lib.input_dim()
    hidden = [args.width] * args.depth
    if args.use_quantum_input or args.use_quantum_hidden:
        readout = 2 ** (args.quantum_qubits - 1)
        pos = 0 if args.use_quantum_input else int(args.quantum_hidden_position)
        if not 0 <= pos < len(hidden):
            raise ValueError(f"--quantum-hidden-position must be in [0, {len(hidden) - 1}] "
                             f"for --depth {args.depth}, got {pos}")
        hidden[pos] = readout
        print(f"[Info] layer index {pos + 1} set to the quantum readout width {readout}; "
              f"--width {args.width} applies to the other layers.")
    return [dim, *hidden, 3]


def training_config(args, layers):
    return {
        "data": str(Path(args.data).expanduser().resolve()),
        "epochs": args.epochs, "batch_size": args.batch_size,
        "samples_per_epoch": args.samples_per_epoch,
        "window_duration": args.window_duration, "pde_points": args.pde_points,
        "learning_rate": args.learning_rate, "w_data": args.w_data,
        "w_pde": args.w_pde, "seed": args.seed,
        "regime": args.regime,
        "use_j2_baseline": args.use_j2_baseline, "use_osc2mean": args.use_osc2mean,
        "use_lat_features": args.use_lat_features,
        "use_orbit_features": args.use_orbit_features,
        "use_orbit_phase_features": args.use_orbit_phase_features,
        "orbit_harmonics": args.orbit_harmonics,
        "use_fourier_time": args.use_fourier_time,
        "fourier_n_freqs": args.fourier_n_freqs,
        "fourier_spacing": args.fourier_spacing,
        "per_sample_gate": args.per_sample_gate, "gate_power": args.gate_power,
        "gate_kind": args.gate_kind, "gate_tau_div": args.gate_tau_div,
        "use_j2_physics": args.use_j2_physics, "pde_norm": args.pde_norm,
        "pde_gate_weighted": args.pde_gate_weighted,
        "pde_subtract_kepler": args.pde_subtract_kepler,
        "fourier_max_freq": args.fourier_max_freq,
        "fourier_learnable": args.fourier_learnable,
        "use_quantum_input": args.use_quantum_input,
        "use_quantum_hidden": args.use_quantum_hidden,
        "quantum_hidden_position": args.quantum_hidden_position,
        "quantum_qubits": args.quantum_qubits,
        "quantum_depth": args.quantum_depth,
        "quantum_batch_size": args.quantum_batch_size,
        "layers": layers, "input_dim": layers[0],
    }


def environment_tensors(device):
    sun = torch.tensor(lib.SUN_POS_ECI, dtype=lib.DTYPE, device=device).unsqueeze(0)
    moon = torch.tensor(lib.MOON_POS_ECI, dtype=lib.DTYPE, device=device).unsqueeze(0)
    t = lambda v: torch.tensor(v, dtype=lib.DTYPE, device=device)
    return (sun, moon, t(1e-12), t(lib.CD), t(lib.A_CROSS), t(lib.M_SC),
            t(0.0), t(lib.EPSILON), sun / torch.norm(sun, dim=-1, keepdim=True),
            torch.tensor([0.0, 0.0, 1.0], dtype=lib.DTYPE, device=device).unsqueeze(0))


def load_resume(path, net, optimizer, scheduler, device, current_config):
    payload = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in payload:
        net.load_state_dict(payload)
        return 0, {"total": [], "pde": [], "data": []}
    saved_config = payload.get("config", {})
    compatibility_keys = (
        "regime", "use_j2_baseline", "use_osc2mean", "use_lat_features",
        "use_orbit_features", "use_orbit_phase_features", "orbit_harmonics", "use_fourier_time",
        "fourier_n_freqs", "fourier_spacing", "fourier_max_freq",
        "fourier_learnable", "per_sample_gate", "gate_power", "gate_kind",
        "gate_tau_div", "use_j2_physics", "pde_norm", "pde_gate_weighted",
        "pde_subtract_kepler", "use_quantum_input", "use_quantum_hidden",
        "quantum_hidden_position", "quantum_qubits", "quantum_depth", "layers",
    )
    mismatches = [
        key for key in compatibility_keys
        if key in saved_config and saved_config[key] != current_config.get(key)
    ]
    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={saved_config[key]!r}, requested={current_config.get(key)!r}"
            for key in mismatches
        )
        raise ValueError(f"Resume configuration does not match the checkpoint ({details})")
    net.load_state_dict(payload["model_state_dict"])
    if payload.get("optimizer_state_dict"):
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if payload.get("scheduler_state_dict"):
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return int(payload.get("epoch", 0)), payload.get(
        "loss_history", {"total": [], "pde": [], "data": []})


def save_checkpoint(path, epoch, net, optimizer, scheduler, config, history):
    torch.save({"format_version": 2, "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config, "loss_history": history}, path)


def save_loss_plot(history, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, key, title in zip(axes, ("total", "pde", "data"), ("Total", "PDE", "Data")):
        values = history[key]
        axis.plot(range(1, len(values) + 1), values)
        axis.set_title(f"{title} loss")
        axis.set_xlabel("Epoch")
        if any(value > 0 for value in values):
            axis.set_yscale("log")
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    validate(args)
    set_seed(args.seed)
    configure_library(args)

    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    layers = model_layers(args)
    config = training_config(args, layers)
    print("[Train] resolved configuration:\n" + json.dumps(config, indent=2))

    net = lib.build_net(layers)
    # The Fourier module must be attached AFTER the net exists so its
    # frequencies land in state_dict() and in the optimizer.
    lib.attach_fourier(net)
    device = next(net.parameters()).device
    kind = ("quantum-A" if args.use_quantum_input else
            "quantum-B" if args.use_quantum_hidden else "classical")
    print(f"[Train] {lib.describe_config()}")
    print(f"[Train] net={kind} layers={layers} device={device} "
          f"params={sum(p.numel() for p in net.parameters())}")

    model = lib.GeneralistModel(net, W_DATA=args.w_data, W_PDE=args.w_pde)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.75, patience=15, min_lr=1e-10)

    start_epoch, history = 0, {"total": [], "pde": [], "data": []}
    if args.resume:
        start_epoch, history = load_resume(
            args.resume, net, optimizer, scheduler, device, config)
        if start_epoch >= args.epochs:
            raise ValueError(
                f"--epochs is the final target and must exceed checkpoint epoch "
                f"{start_epoch}; got {args.epochs}"
            )
        print(f"[Train] Resumed epoch {start_epoch} from {args.resume}")

    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    dataset = EphemerisDataset(args.data, args.samples_per_epoch, args.window_duration)
    collate = partial(ephemeris_collate_fn, K_pde=args.pde_points,
                      device=device, dtype=lib.DTYPE)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, collate_fn=collate)

    env = environment_tensors(device)
    last_epoch, interrupted = start_epoch, False
    net.train()
    print(f"[Train] epochs {start_epoch + 1}..{args.epochs}")
    try:
        for epoch in range(start_epoch + 1, args.epochs + 1):
            totals, batches = np.zeros(3, dtype=np.float64), 0
            for batch in loader:
                (t_data, x0_data, u0_data, r_truth, v_truth,
                 t_pde, x0_pde, u0_pde, tseg_data, tseg_pde) = batch
                if not args.per_sample_gate:
                    tseg_data = tseg_pde = None
                optimizer.zero_grad(set_to_none=True)
                loss, loss_pde, loss_data = model.epoch_loss(
                    t_data, x0_data, u0_data, r_truth, v_truth,
                    t_pde, x0_pde, u0_pde, *env,
                    Tseg_data_batch=tseg_data, Tseg_pde_batch=tseg_pde)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                totals += (loss.item(), loss_pde.item(), loss_data.item())
                batches += 1
            if not batches:
                raise RuntimeError(f"Epoch {epoch} produced no finite batches")
            averages = totals / batches
            scheduler.step(float(averages[0]))
            for key, value in zip(("total", "pde", "data"), averages):
                history[key].append(float(value))
            last_epoch = epoch
            if epoch % args.print_every == 0:
                print(f"[Train] epoch={epoch:04d} total={averages[0]:.3e} "
                      f"pde={averages[1]:.3e} data={averages[2]:.3e} "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}")
            if epoch % args.checkpoint_every == 0:
                save_checkpoint(checkpoint_dir / f"epoch_{epoch:04d}.pth", epoch,
                                net, optimizer, scheduler, config, history)
                save_loss_plot(history, output_dir / "loss_curves.png")
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n[Train] Interrupted after epoch {last_epoch}; saving current state")

    latest = output_dir / "latest.pth"
    save_checkpoint(latest, last_epoch, net, optimizer, scheduler, config, history)
    save_loss_plot(history, output_dir / "loss_curves.png")
    print(f"[Save] {latest}")
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
