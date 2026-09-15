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
    return 1
}

if (( $# < 1 )); then
    echo "Usage: ./evaluate_40k_experiments.sh EXPERIMENT [DATA_40K_CSV|auto] [OUTPUT_ROOT] [POINTS]" >&2
    echo "EXPERIMENT is 1, 2, 3, 4, or all. POINTS defaults to 4001." >&2
    exit 2
fi

experiment="$1"
data_file="$(resolve_dataset "${2:-auto}")"
output_root="${3:-$project_root/runs_40k}"
points="${4:-4001}"

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
export PYTHON_BIN="$python_bin"

inference_path="$script_dir/inference_plot_osc2mean.py"
if [[ ! -f "$inference_path" ]]; then
    echo "ERROR: inference program not found: $inference_path" >&2
    exit 2
fi
if ! inference_help="$("$python_bin" "$inference_path" --help 2>&1)"; then
    echo "ERROR: inference_plot_osc2mean.py could not start:" >&2
    printf '%s\n' "$inference_help" >&2
    exit 2
fi
if [[ "$experiment" == 4 || "$experiment" == all ]]; then
    if [[ "$inference_help" != *"--use-orbit-phase-features"* ]]; then
        echo "ERROR: inference files are older than the experiment-4 checkpoint format." >&2
        echo "Update inference_plot_osc2mean.py, fast_3d_and_error_plotting.py," >&2
        echo "plotting_common.py, and pinn_lib_kepler.py together." >&2
        exit 2
    fi
fi

echo "[Preflight] project root: $project_root"
echo "[Preflight] python:       $python_bin"
echo "[Preflight] dataset:      $data_file"
echo "[Preflight] inference compatibility and Python imports: PASS"

run_name() {
    case "$1" in
        1) echo "exp1_h4_no_lat" ;;
        2) echo "exp2_h4_lat" ;;
        3) echo "exp3_h3_lat_j2_physics" ;;
        4) echo "exp4_static_orbit_fourier4" ;;
    esac
}

evaluate_one() {
    local number="$1" name checkpoint eval_dir
    name="$(run_name "$number")"
    checkpoint="$output_root/$name/latest.pth"
    eval_dir="$output_root/$name/evaluation_40k"
    if [[ ! -f "$checkpoint" ]]; then
        echo "ERROR: checkpoint not found: $checkpoint" >&2
        return 1
    fi

    echo
    echo "======================================================================"
    echo "EVALUATING EXPERIMENT $number: $name"
    echo "Checkpoint: $checkpoint"
    echo "Dataset:    $data_file"
    echo "Output:     $eval_dir"
    echo "Horizon:    40000 s; points: $points; truth: CSV"
    echo "Checkpoint metadata will restore all training feature flags."
    echo "======================================================================"

    "$script_dir/plot_checkpoint.sh" \
        "$checkpoint" "$data_file" "$eval_dir" 40000 csv "$points"
}

if [[ "$experiment" == all ]]; then
    for number in 1 2 3 4; do
        evaluate_one "$number"
    done
else
    evaluate_one "$experiment"
fi
