#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"
checkpoint="${1:?Usage: plot_checkpoint.sh CHECKPOINT DATA_CSV [OUTPUT_DIR] [HORIZON] [TRUTH_SOURCE] [POINTS]}"
data_file="${2:?Usage: plot_checkpoint.sh CHECKPOINT DATA_CSV [OUTPUT_DIR] [HORIZON] [TRUTH_SOURCE] [POINTS]}"
output_dir="${3:-$script_dir/checkpoint_plots}"
horizon="${4:-}"
truth_source="${5:-auto}"
points="${6:-}"
truth_options=(--truth-source "$truth_source")
if [[ -n "$horizon" ]]; then
    truth_options+=(--horizon "$horizon")
fi
if [[ -n "$points" ]]; then
    truth_options+=(--points "$points")
fi

"$python_bin" "$script_dir/inference_plot_osc2mean.py" \
    --checkpoint "$checkpoint" \
    --data "$data_file" \
    --output-dir "$output_dir/inference" \
    "${truth_options[@]}"

"$python_bin" "$script_dir/fast_3d_and_error_plotting.py" \
    --checkpoint "$checkpoint" \
    --data "$data_file" \
    --output-dir "$output_dir/trajectories" \
    --stats-dir "$output_dir/inference" \
    "${truth_options[@]}" \
    --count "${PLOT_SATELLITES:-6}"
