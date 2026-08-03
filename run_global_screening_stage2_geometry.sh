#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-spondy}"
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

CONFIG_DIR="./presence/configs/global_screening_stage"
EXPERIMENT_ROOT="./presence/experiments/global_screening_stage"
TRAIN_SCRIPT="./presence/scripts/lumbar_crop_resnet50/train_lumbar_crop_cv.py"
ANALYZE_SCRIPT="./presence/scripts/lumbar_crop_resnet50/analyze_lumbar_crop_basic_cv_ensemble_test.py"
GEOMETRY_SCRIPT="./presence/scripts/lumbar_crop_resnet50/generate_case_geometry_features.py"

N_FOLDS="${N_FOLDS:-5}"
CKPT_NAME="${CKPT_NAME:-best.pt}"
ENSEMBLE_MODE="${ENSEMBLE_MODE:-probs}"

BASELINE_DIR="${BASELINE_DIR:-./presence/experiments/lumbarCropHalfSacrum_noimplant_resnet50_threshold/resnet50_basic_lumbarCropHalfSacrum_aug_th04_w256_b32_bce_balacc_cv5}"
BASELINE_METRICS="${BASELINE_DIR}/test_analysis/basic_cv_ensemble_overall_metrics.csv"

IMG384_CONFIG="${CONFIG_DIR}/resnet50_basic_lumbarCropHalfSacrum_aug_th04_w384_b32_bce_balacc_cv5.yaml"
GEO_TH03_CONFIG="${CONFIG_DIR}/resnet50_lumbarCropHalfSacrum_geo_th03_b32_bce_balacc_cv5.yaml"
GEO_TH04_CONFIG="${CONFIG_DIR}/resnet50_lumbarCropHalfSacrum_geo_th04_b32_bce_balacc_cv5.yaml"

mkdir -p "${EXPERIMENT_ROOT}"

config_value() {
  python - "$1" "$2" <<'INNER_PY'
from pathlib import Path
import sys
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
value = cfg
for key in sys.argv[2].split('.'):
    value = value[key]
print(value)
INNER_PY
}

train_if_needed() {
  local cfg="$1"
  local save_dir
  save_dir=$(config_value "${cfg}" "save_dir")
  if [[ -f "${save_dir}/cv_summary.json" ]]; then
    echo "[SKIP TRAIN] ${save_dir} already has cv_summary.json"
  else
    echo "[TRAIN] ${cfg}"
    python "${TRAIN_SCRIPT}" --config "${cfg}"
  fi
}

analyze_if_needed() {
  local cfg="$1"
  local threshold="$2"
  local save_dir
  save_dir=$(config_value "${cfg}" "save_dir")
  local metrics="${save_dir}/test_analysis/basic_cv_ensemble_overall_metrics.csv"
  if [[ -f "${metrics}" ]]; then
    echo "[SKIP ANALYZE] ${metrics} already exists"
  else
    echo "[ANALYZE] ${cfg} threshold=${threshold}"
    mkdir -p "${save_dir}/test_analysis"
    python "${ANALYZE_SCRIPT}" \
      --config "${cfg}" \
      --cv_dir "${save_dir}" \
      --outdir "${save_dir}/test_analysis" \
      --n_folds "${N_FOLDS}" \
      --ckpt_name "${CKPT_NAME}" \
      --ensemble_mode "${ENSEMBLE_MODE}" \
      --threshold "${threshold}"
  fi
}

update_geo_image_size() {
  local selected_size="$1"
  python - "${selected_size}" "${GEO_TH03_CONFIG}" "${GEO_TH04_CONFIG}" <<'INNER_PY'
from pathlib import Path
import sys
import yaml
selected = sys.argv[1]
if selected == "384x384":
    image_size = [384, 384]
elif selected == "512x256":
    image_size = [512, 256]
else:
    raise ValueError(f"Unsupported selected image size: {selected}")
for cfg_path in sys.argv[2:]:
    p = Path(cfg_path)
    cfg = yaml.safe_load(p.read_text())
    cfg.setdefault("augmentation", {})["image_size"] = image_size
    cfg["selected_image_size_source"] = selected
    p.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(f"[CONFIG] {p}: augmentation.image_size={image_size}")
INNER_PY
}

choose_image_size() {
  local img384_metrics="$1"
  if [[ -n "${SELECTED_IMAGE_SIZE:-}" ]]; then
    echo "${SELECTED_IMAGE_SIZE}"
    return
  fi
  if [[ ! -f "${BASELINE_METRICS}" ]]; then
    echo "[WARN] Missing baseline metrics: ${BASELINE_METRICS}; defaulting to 384x384 after training it." >&2
    echo "384x384"
    return
  fi
  python - "${BASELINE_METRICS}" "${img384_metrics}" <<'INNER_PY'
import sys
import pandas as pd
base = pd.read_csv(sys.argv[1]).iloc[0]
new = pd.read_csv(sys.argv[2]).iloc[0]

def score(r):
    gap = abs(float(r['sensitivity']) - float(r['specificity']))
    return 0.45 * float(r['balanced_acc']) + 0.45 * float(r['accuracy']) - 0.10 * gap
base_score = score(base)
new_score = score(new)
print(f"[COMPARE] baseline 512x256: acc={base['accuracy']:.4f}, bal_acc={base['balanced_acc']:.4f}, sens={base['sensitivity']:.4f}, spec={base['specificity']:.4f}, score={base_score:.4f}", file=sys.stderr)
print(f"[COMPARE] new 384x384:   acc={new['accuracy']:.4f}, bal_acc={new['balanced_acc']:.4f}, sens={new['sensitivity']:.4f}, spec={new['specificity']:.4f}, score={new_score:.4f}", file=sys.stderr)
print('384x384' if new_score > base_score else '512x256')
INNER_PY
}

echo "[STEP] Refresh case-level geometry features"
python "${GEOMETRY_SCRIPT}" --dataset-root ./dataset_half_sacrum --splits train valid test

echo "[STEP] Train/evaluate image-only 384x384 threshold 0.4"
train_if_needed "${IMG384_CONFIG}"
analyze_if_needed "${IMG384_CONFIG}" "0.4"

IMG384_SAVE_DIR=$(config_value "${IMG384_CONFIG}" "save_dir")
IMG384_METRICS="${IMG384_SAVE_DIR}/test_analysis/basic_cv_ensemble_overall_metrics.csv"
SELECTED_SIZE=$(choose_image_size "${IMG384_METRICS}")
echo "[SELECTED_IMAGE_SIZE] ${SELECTED_SIZE}"
update_geo_image_size "${SELECTED_SIZE}"

echo "[STEP] Train/evaluate geometry fusion threshold 0.3"
train_if_needed "${GEO_TH03_CONFIG}"
analyze_if_needed "${GEO_TH03_CONFIG}" "0.3"

echo "[STEP] Train/evaluate geometry fusion threshold 0.4"
train_if_needed "${GEO_TH04_CONFIG}"
analyze_if_needed "${GEO_TH04_CONFIG}" "0.4"

echo "[DONE] Stage2 global screening experiments complete."
