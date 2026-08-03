#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-spondy}"
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

SCRIPT="./presence/scripts/stage4_case_fusion/analyze_case_fusion.py"

GLOBAL_DIR="./presence/experiments/global_screening_stage2/resnet50_lumbarCropHalfSacrum_geo_th03_b32_bce_balacc_cv5/test_analysis"
LOCAL_DIR="./presence/experiments/local_pair_diagnostic_previous/resnet50_local_pair_slip_aux_balacc_half_sacrum_cv5_th02/test_analysis"
OUTDIR="./presence/experiments/stage4_case_fusion/global_geo_th03__local_slip_aux_th02"

python "${SCRIPT}" \
  --global-case-csv "${GLOBAL_DIR}/basic_cv_ensemble_case_predictions.csv" \
  --local-case-csv "${LOCAL_DIR}/local_pair_diagnostic_case_predictions.csv" \
  --local-pair-csv "${LOCAL_DIR}/local_pair_diagnostic_pair_predictions.csv" \
  --outdir "${OUTDIR}" \
  --local-threshold 0.2 \
  --thresholds 0.1 0.2 0.3 0.4 0.5
