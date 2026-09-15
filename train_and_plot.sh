#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"
data_file="${1:?Usage: train_and_plot.sh DATA_CSV [RUN_DIR] [EPOCHS] [TRAIN_OPTIONS...]}"
run_dir="${2:-$script_dir/training_output}"
epochs="${3:-1200}"
if (( $# >= 3 )); then
    shift 3
else
    shift "$#"
fi

plot_options=(--truth-source "${TRUTH_SOURCE:-auto}")
if [[ -n "${PLOT_HORIZON:-}" ]]; then
    plot_options+=(--horizon "$PLOT_HORIZON")
fi
if [[ -n "${PLOT_POINTS:-}" ]]; then
    plot_options+=(--points "$PLOT_POINTS")
fi

"$python_bin" "$script_dir/train_opti.py" \
    --data "$data_file" \
    --output-dir "$run_dir" \
    --epochs "$epochs" \
    "$@"

"$python_bin" "$script_dir/inference_plot_osc2mean.py" \
    --checkpoint "$run_dir/latest.pth" \
    --data "$data_file" \
    --output-dir "$run_dir/plots/inference" \
    "${plot_options[@]}"

"$python_bin" "$script_dir/fast_3d_and_error_plotting.py" \
    --checkpoint "$run_dir/latest.pth" \
    --data "$data_file" \
    --output-dir "$run_dir/plots/trajectories" \
    --stats-dir "$run_dir/plots/inference" \
    "${plot_options[@]}" \
    --count "${PLOT_SATELLITES:-6}"
