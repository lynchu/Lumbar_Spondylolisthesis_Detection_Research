#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_SH="${CONDA_SH:-/home/cclynn/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-spondy}"
if [[ -f "$CONDA_SH" ]]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
fi

CONFIG_DIR="${CONFIG_DIR:-presence/configs/local_pair_diagnostic}"
EXP_DIR="${EXP_DIR:-presence/experiments/local_pair_diagnostic_stage}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-presence/scripts/local_pair_diagnostic/train_local_pair_diagnostic_cv.py}"
EVAL_SCRIPT="${EVAL_SCRIPT:-presence/scripts/local_pair_diagnostic/analyze_local_pair_diagnostic_cv_ensemble_test.py}"
LOCATION_SET_SCRIPT="${LOCATION_SET_SCRIPT:-presence/scripts/local_pair_diagnostic/analyze_location_set_accuracy.py}"
ANALYSIS_DIR_NAME="${ANALYSIS_DIR_NAME:-test_analysis}"
RESUME_CONFIG_DIR="${RESUME_CONFIG_DIR:-${TMPDIR:-/tmp}/local_pair_diagnostic_resume_configs}"
mkdir -p "$RESUME_CONFIG_DIR"

RUNS=(
  "location_only|th02|0.2|resnet50_local_pair_location_only_balacc_half_sacrum_cv5_th02"
  "location_only|th03|0.3|resnet50_local_pair_location_only_balacc_half_sacrum_cv5_th03"
  "location_only|th04|0.4|resnet50_local_pair_location_only_balacc_half_sacrum_cv5_th04"
  "slip_aux|th02|0.2|resnet50_local_pair_slip_aux_balacc_half_sacrum_cv5_th02"
  "slip_aux|th03|0.3|resnet50_local_pair_slip_aux_balacc_half_sacrum_cv5_th03"
  "slip_aux|th04|0.4|resnet50_local_pair_slip_aux_balacc_half_sacrum_cv5_th04"
  "slip_ordinal|th02|0.2|resnet50_local_pair_slip_ordinal_balacc_half_sacrum_cv5_th02"
  "slip_ordinal|th03|0.3|resnet50_local_pair_slip_ordinal_balacc_half_sacrum_cv5_th03"
  "slip_ordinal|th04|0.4|resnet50_local_pair_slip_ordinal_balacc_half_sacrum_cv5_th04"
)

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] Missing required file: $path" >&2
    exit 1
  fi
}

make_resume_config() {
  local config="$1"
  local cv_dir="$2"
  local resume_config="$3"
  python - "$config" "$cv_dir" "$resume_config" <<'INNER_PY'
import sys
from pathlib import Path

config = Path(sys.argv[1])
cv_dir = Path(sys.argv[2])
out = Path(sys.argv[3])
num_folds = 5

if (cv_dir / "cv_summary.json").exists():
    print("SKIP cv_summary.json exists")
    raise SystemExit(0)

complete = [(cv_dir / f"fold_{i}" / "fold_complete.json").exists() for i in range(1, num_folds + 1)]
if any(complete):
    start_fold = next((i for i, done in enumerate(complete, start=1) if not done), num_folds + 1)
else:
    last_started = 0
    for i in range(1, num_folds + 1):
        fold_dir = cv_dir / f"fold_{i}"
        if (fold_dir / "last.pt").exists() or (fold_dir / "best_metrics.json").exists():
            last_started = i
    start_fold = max(last_started, 1)

while start_fold > 1 and not (cv_dir / f"fold_{start_fold - 1}" / "best_metrics.json").exists():
    start_fold -= 1

run_folds = [] if start_fold > num_folds else list(range(start_fold, num_folds + 1))
skip_prefixes = ("run_folds:", "resume_from_last:", "save_dir:")
lines = [line for line in config.read_text().splitlines() if not line.startswith(skip_prefixes)]
lines.append(f"save_dir: ./{cv_dir.as_posix()}")
lines.append(f"run_folds: {run_folds}")
lines.append("resume_from_last: true")
out.write_text("\n".join(lines).rstrip() + "\n")
print("RUN run_folds=" + ",".join(str(x) for x in run_folds))
INNER_PY
}

run_training() {
  local branch="$1"
  local threshold="$2"
  local exp_name="$3"
  local config="$CONFIG_DIR/${exp_name}.yaml"
  local cv_dir="$EXP_DIR/$exp_name"
  local resume_config="$RESUME_CONFIG_DIR/${exp_name}.resume.yaml"

  require_file "$config"
  echo "================================================================================"
  echo "[TRAIN/RESUME] branch=$branch threshold=$threshold config=$config"
  echo "[TRAIN/RESUME] output=$cv_dir"
  echo "================================================================================"

  status="$(make_resume_config "$config" "$cv_dir" "$resume_config")"
  echo "[INFO] $status"
  if [[ "$status" == SKIP* ]]; then
    return 0
  fi
  python "$TRAIN_SCRIPT" --config "$resume_config"
}

run_evaluation() {
  local branch="$1"
  local th_name="$2"
  local threshold="$3"
  local exp_name="$4"
  local config="$CONFIG_DIR/${exp_name}.yaml"
  local cv_dir="$EXP_DIR/$exp_name"
  local outdir="$cv_dir/$ANALYSIS_DIR_NAME"

  require_file "$config"
  echo "================================================================================"
  echo "[EVAL] branch=$branch threshold=$threshold cv_dir=$cv_dir"
  echo "[EVAL] outdir=$outdir"
  echo "================================================================================"
  python "$EVAL_SCRIPT" \
    --config "$config" \
    --cv_dir "$cv_dir" \
    --outdir "$outdir" \
    --threshold "$threshold"
}

require_file "$TRAIN_SCRIPT"
require_file "$EVAL_SCRIPT"
require_file "$LOCATION_SET_SCRIPT"

for item in "${RUNS[@]}"; do
  IFS='|' read -r branch th_name threshold exp_name <<< "$item"
  run_training "$branch" "$threshold" "$exp_name"
done

for item in "${RUNS[@]}"; do
  IFS='|' read -r branch th_name threshold exp_name <<< "$item"
  run_evaluation "$branch" "$th_name" "$threshold" "$exp_name"
done

python "$LOCATION_SET_SCRIPT" \
  --exp_dir "$EXP_DIR" \
  --analysis_dir_name "$ANALYSIS_DIR_NAME" \
  --out_csv "$EXP_DIR/location_set_accuracy_summary.csv"
