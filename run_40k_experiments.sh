#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

discover_project_root() {
    local current="$script_dir"
    while [[ "$current" != "/" ]]; do
        if [[ -d "$current/data" ]]; then
            printf '%s\n' "$current"
            return 0
        fi
        current="$(dirname "$current")"
    done
    printf '%s\n' "$script_dir"
}

project_root="$(discover_project_root)"

absolute_file() {
    local path="$1"
    printf '%s/%s\n' "$(cd "$(dirname "$path")" && pwd)" "$(basename "$path")"
}

resolve_dataset() {
    local requested="${1:-auto}" candidate match
    local -a candidates=()

    if [[ -n "$requested" && "$requested" != "auto" ]]; then
        candidates+=(
            "$requested"
            "$project_root/$requested"
            "$project_root/data/$requested"
            "$project_root/data/LEO_739/$requested"
        )
    else
        candidates+=(
            "$project_root/data/LEO_739/propagated_40k_10stp.csv"
            "$project_root/data/LEO_739/propagated_40k.csv"
            "$project_root/data/propagated_40k_10stp.csv"
            "$project_root/propagated_40k_10stp.csv"
        )
    fi

    for candidate in "${candidates[@]}"; do
        if [[ -f "$candidate" ]]; then
            absolute_file "$candidate"
            return 0
        fi
    done

    if [[ "$requested" == "auto" && -d "$project_root/data" ]]; then
        match="$(find "$project_root/data" -type f \
            \( -name 'propagated_40k_10stp.csv' -o -name 'propagated_40k.csv' \) \
            -print -quit 2>/dev/null || true)"
        if [[ -n "$match" ]]; then
            absolute_file "$match"
            return 0
        fi
    fi

    echo "ERROR: could not locate the 40K dataset." >&2
    echo "Project root discovered from this script: $project_root" >&2
    echo "Requested dataset: $requested" >&2
    echo "Expected default: $project_root/data/LEO_739/propagated_40k_10stp.csv" >&2
    echo "Pass 'auto', a CSV filename, a tap-lab-relative path, or an absolute path." >&2
    return 1
}

usage() {
    cat <<'EOF'
Usage:
  ./run_40k_experiments.sh EXPERIMENT [DATA_40K_CSV|auto] [OUTPUT_ROOT] [EPOCHS] [EXTRA_TRAIN_ARGS...]

EXPERIMENT is 1, 2, 3, 4, or all.

Examples:
  ./run_40k_experiments.sh 1
  ./run_40k_experiments.sh 2 auto runs_40k 1200
  ./run_40k_experiments.sh 3 data/LEO_739/propagated_40k_10stp.csv runs_40k 1200
  DRY_RUN=1 ./run_40k_experiments.sh all

Optional environment variables:
  PYTHON_BIN, BATCH_SIZE, SAMPLES_PER_EPOCH, LEARNING_RATE,
  CHECKPOINT_EVERY, WIDTH, DEPTH, PDE_POINTS, W_PDE, SEED, DRY_RUN
EOF
}

if (( $# < 1 )); then
    usage >&2
    exit 2
fi

experiment="$1"
data_request="${2:-auto}"
data_file="$(resolve_dataset "$data_request")"
output_root="${3:-$project_root/runs_40k}"
epochs="${4:-2000}"
if (( $# >= 4 )); then
    shift 4
else
    shift "$#"
fi
extra_args=("$@")

case "$experiment" in
    1|2|3|4|all) ;;
    *) echo "ERROR: EXPERIMENT must be 1, 2, 3, 4, or all; got '$experiment'" >&2; exit 2 ;;
esac

if [[ "$output_root" != /* ]]; then
    output_root="$project_root/$output_root"
fi

python_bin="${PYTHON_BIN:-python3}"
if [[ "$python_bin" == */* && "$python_bin" != /* && -x "$project_root/$python_bin" ]]; then
    python_bin="$project_root/$python_bin"
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: $python_bin" >&2
    exit 2
fi

trainer_path="$script_dir/train_opti.py"
if [[ ! -f "$trainer_path" ]]; then
    echo "ERROR: trainer not found beside the launcher: $trainer_path" >&2
    exit 2
fi

if ! trainer_help="$("$python_bin" "$trainer_path" --help 2>&1)"; then
    echo "ERROR: train_opti.py could not start. Its import/dependency error follows:" >&2
    printf '%s\n' "$trainer_help" >&2
    exit 2
fi

required_options=(
    --data --output-dir --epochs --window-duration --batch-size
    --samples-per-epoch --learning-rate --checkpoint-every --width --depth
    --seed --regime --use-j2-baseline --use-osc2mean --per-sample-gate
    --gate-kind --gate-power --gate-tau-div --use-quantum-input
    --use-quantum-hidden --use-orbit-features --orbit-harmonics
    --use-lat-features --use-fourier-time --use-j2-physics --pde-points --w-pde
)
if [[ "$experiment" == 3 || "$experiment" == all ]]; then
    required_options+=(--pde-norm --pde-gate-weighted)
fi
if [[ "$experiment" == 4 || "$experiment" == all ]]; then
    required_options+=(
        --use-orbit-phase-features --fourier-n-freqs --fourier-spacing
        --fourier-learnable
    )
fi

missing_options=()
for option in "${required_options[@]}"; do
    if [[ "$trainer_help" != *"$option"* ]]; then
        missing_options+=("$option")
    fi
done
if (( ${#missing_options[@]} > 0 )); then
    echo "ERROR: launcher and trainer versions do not match." >&2
    echo "Trainer: $trainer_path" >&2
    printf 'Unsupported required options:' >&2
    printf ' %s' "${missing_options[@]}" >&2
    printf '\n' >&2
    echo "Copy/update these files together on the VM:" >&2
    echo "  run_40k_experiments.sh, train_opti.py, pinn_lib_kepler.py," >&2
    echo "  plotting_common.py, inference_plot_osc2mean.py," >&2
    echo "  fast_3d_and_error_plotting.py" >&2
    exit 2
fi

echo "[Preflight] project root: $project_root"
echo "[Preflight] trainer:      $trainer_path"
echo "[Preflight] python:       $python_bin"
echo "[Preflight] dataset:      $data_file"
echo "[Preflight] CLI compatibility and Python imports: PASS"

common_args=(
    --data "$data_file"
    --epochs "$epochs"
    --window-duration 40000
    --batch-size "${BATCH_SIZE:-32}"
    --samples-per-epoch "${SAMPLES_PER_EPOCH:-12000}"
    --learning-rate "${LEARNING_RATE:-0.0001}"
    --checkpoint-every "${CHECKPOINT_EVERY:-50}"
    --width "${WIDTH:-256}"
    --depth "${DEPTH:-5}"
    --seed "${SEED:-42}"
    --regime LEO
    --use-j2-baseline
    --use-osc2mean
    --per-sample-gate
    --gate-kind tanh
    --gate-power 2.0
    --gate-tau-div 3.0
    --no-use-quantum-input
    --no-use-quantum-hidden
)

run_one() {
    local number="$1"
    local name description expected_inputs
    local -a experiment_args

    case "$number" in
        1)
            name="exp1_h4_no_lat"
            description="Base 9 + static orbit descriptors + sin/cos(u) through harmonic 4; no latitude; J2+osc2mean baseline"
            expected_inputs=20
            experiment_args=(
                --use-orbit-features --orbit-harmonics 4
                --no-use-lat-features --no-use-fourier-time
                --no-use-j2-physics --pde-points 0 --w-pde 0.0
            )
            ;;
        2)
            name="exp2_h4_lat"
            description="Experiment 1 plus two latitude features; J2+osc2mean baseline"
            expected_inputs=22
            experiment_args=(
                --use-orbit-features --orbit-harmonics 4
                --use-lat-features --no-use-fourier-time
                --no-use-j2-physics --pde-points 0 --w-pde 0.0
            )
            ;;
        3)
            name="exp3_h3_lat_j2_physics"
            description="Base 9 + orbit harmonics through 3 + latitude + J2 physics loss; J2+osc2mean baseline"
            expected_inputs=20
            experiment_args=(
                --use-orbit-features --orbit-harmonics 3
                --use-lat-features --no-use-fourier-time
                --use-j2-physics --pde-points "${PDE_POINTS:-4}"
                --w-pde "${W_PDE:-0.01}" --pde-norm total --no-pde-gate-weighted
            )
            ;;
        4)
            name="exp4_static_orbit_fourier4"
            description="Base 9 + inclination/eccentricity only (no sin/cos u) + four Fourier phase frequencies; no latitude; J2+osc2mean baseline"
            expected_inputs=20
            experiment_args=(
                --use-orbit-features --no-use-orbit-phase-features --orbit-harmonics 1
                --no-use-lat-features
                --use-fourier-time --fourier-n-freqs 4 --fourier-spacing integer
                --no-fourier-learnable
                --no-use-j2-physics --pde-points 0 --w-pde 0.0
            )
            ;;
    esac

    local run_dir="$output_root/$name"
    local -a command=(
        "$python_bin" "$script_dir/train_opti.py"
        "${common_args[@]}"
        --output-dir "$run_dir"
        "${experiment_args[@]}"
        "${extra_args[@]}"
    )

    echo
    echo "======================================================================"
    echo "40K EXPERIMENT $number: $name"
    echo "$description"
    echo "Expected network input width: $expected_inputs"
    echo "Dataset: $data_file"
    echo "Output:  $run_dir"
    echo "Epochs:  $epochs"
    echo "Physics: use_j2_physics=$([[ "$number" == 3 ]] && echo true || echo false), w_pde=$([[ "$number" == 3 ]] && echo "${W_PDE:-0.01}" || echo 0.0), pde_points=$([[ "$number" == 3 ]] && echo "${PDE_POINTS:-4}" || echo 0)"
    printf 'Command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    echo "======================================================================"

    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        echo "DRY_RUN=1: command validated and not executed."
    else
        "${command[@]}"
    fi
}

if [[ "$experiment" == all ]]; then
    for number in 1 2 3 4; do
        run_one "$number"
    done
else
    run_one "$experiment"
fi
