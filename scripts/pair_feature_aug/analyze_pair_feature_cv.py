# python analyze_pair_feature_model_cv.py --checkpoint_dir ./presence/experiments/pair_feature_mlp_cv5 --feat_csv ./dataset/test/pair_crop_features.csv \
#   --diag_csv ./dataset/test/lumbar_diagnosis.csv --outdir ./presence/experiments/pair_feature_mlp_cv5 --split_name test --image_prob_method max --ensemble_method mean

#!/usr/bin/env python3
"""
Analyze pair-feature model checkpoints on a split such as test.

Supports either:
1) single checkpoint:
   --checkpoint ./exp/fold_0/best.pt
2) 5-fold CV ensemble checkpoint directory:
   --checkpoint_dir ./exp
   # discovers ./exp/fold_*/best.pt
3) explicit comma-separated checkpoint list:
   --checkpoint_list ./exp/fold_0/best.pt,./exp/fold_1/best.pt,...

All outputs are written to:
    <outdir>/test_analysis/

For CV checkpoints, pair probabilities are ensembled by mean by default.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional, Sequence, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


PAIR_NAMES = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
PAIR_TO_ID = {p: i for i, p in enumerate(PAIR_NAMES)}
UNK_PAIR_ID = len(PAIR_NAMES)
EPS = 1e-8


def normalize_pair_name(a: str, b: str) -> Optional[str]:
    a = str(a).strip().upper().replace(" ", "")
    b = str(b).strip().upper().replace(" ", "")
    order = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "S1": 6}
    if a not in order or b not in order:
        return None
    ia, ib = order[a], order[b]
    if abs(ia - ib) != 1:
        return None
    upper, lower = (a, b) if ia < ib else (b, a)
    pair = f"{upper}-{lower}"
    return pair if pair in PAIR_TO_ID else None


def location_to_pair_set(location) -> set:
    if location is None or pd.isna(location):
        return set()
    s = str(location).strip().upper().replace(" ", "")
    if not s or s == "NAN":
        return set()

    pair_set = set()
    for chunk in re.split(r";+", s):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.fullmatch(r"(L[1-5]|S1)[,-](L[1-5]|S1)", chunk)
        if m:
            pair = normalize_pair_name(m.group(1), m.group(2))
            if pair is not None:
                pair_set.add(pair)
    if pair_set:
        return pair_set

    levels = []
    seen = set()
    order = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "S1": 6}
    for tok in re.split(r"[;,]+", s):
        tok = tok.strip()
        if tok in order and tok not in seen:
            levels.append(tok)
            seen.add(tok)
    levels.sort(key=lambda x: order[x])
    for x, y in zip(levels[:-1], levels[1:]):
        pair = normalize_pair_name(x, y)
        if pair is not None:
            pair_set.add(pair)
    return pair_set


def load_pair_table(feat_csv: str, diag_csv: str, feature_cols: Sequence[str]) -> pd.DataFrame:
    feat = pd.read_csv(feat_csv)
    diag = pd.read_csv(diag_csv)
    if "image_id" not in feat.columns or "pair_name" not in feat.columns:
        raise ValueError(f"{feat_csv} must contain image_id and pair_name")
    if "image_id" not in diag.columns:
        raise ValueError(f"{diag_csv} must contain image_id")

    for c in ["location", "spondylolisthesis_label", "grade", "implant"]:
        if c not in diag.columns:
            diag[c] = np.nan
    diag = diag[["image_id", "location", "spondylolisthesis_label", "grade", "implant"]].copy()

    df = feat.merge(diag, on="image_id", how="left")
    df["gt_pair_set"] = df["location"].apply(location_to_pair_set)
    df["pair_label"] = df.apply(lambda r: int(str(r["pair_name"]) in r["gt_pair_set"]), axis=1)
    df["study_label"] = pd.to_numeric(df["spondylolisthesis_label"], errors="coerce")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{feat_csv} missing selected feature columns: {missing}")
    df = df[df["pair_name"].isin(PAIR_NAMES)].copy()
    return df.reset_index(drop=True)


class LoadedPreprocessor:
    def __init__(self, state: dict):
        self.feature_cols = state["feature_cols"]
        self.median_ = state["median"]
        self.mean_ = state["mean"]
        self.std_ = state["std"]
        self.lo_ = state["lo"]
        self.hi_ = state["hi"]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        arrs = []
        for c in self.feature_cols:
            s = pd.to_numeric(df[c], errors="coerce").astype(float)
            s = s.fillna(float(self.median_[c])).clip(float(self.lo_[c]), float(self.hi_[c]))
            s = (s - float(self.mean_[c])) / max(float(self.std_[c]), EPS)
            arrs.append(s.to_numpy(dtype=np.float32))
        return np.stack(arrs, axis=1).astype(np.float32)


class PairFeatureDataset(Dataset):
    def __init__(self, df: pd.DataFrame, x_num: np.ndarray):
        self.df = df.reset_index(drop=True).copy()
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.pair_id = torch.tensor([PAIR_TO_ID.get(str(p), UNK_PAIR_ID) for p in self.df["pair_name"]], dtype=torch.long)
        self.y = torch.tensor(self.df["pair_label"].to_numpy(dtype=np.float32), dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {"x_num": self.x_num[idx], "pair_id": self.pair_id[idx], "y": self.y[idx]}


class PairFeatureMLP(nn.Module):
    def __init__(self, n_num: int, n_pair: int = len(PAIR_NAMES) + 1, pair_emb_dim: int = 4, hidden: int = 64, dropout: float = 0.20):
        super().__init__()
        self.pair_emb = nn.Embedding(n_pair, pair_emb_dim)
        in_dim = n_num + pair_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1),
        )

    def forward(self, x_num, pair_id):
        return self.net(torch.cat([x_num, self.pair_emb(pair_id)], dim=1)).squeeze(1)


class NumericFeatureTokenizer(nn.Module):
    def __init__(self, n_num: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_num, d_token))
        self.bias = nn.Parameter(torch.empty(n_num, d_token))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x_num):
        return x_num.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class PairFeatureFTTransformer(nn.Module):
    def __init__(self, n_num: int, n_pair: int = len(PAIR_NAMES) + 1, d_token: int = 32, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.num_tok = NumericFeatureTokenizer(n_num, d_token)
        self.pair_emb = nn.Embedding(n_pair, d_token)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 4, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.Linear(d_token, d_token), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_token, 1))
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, x_num, pair_id):
        bsz = x_num.shape[0]
        tokens = torch.cat([self.cls.expand(bsz, -1, -1), self.pair_emb(pair_id).unsqueeze(1), self.num_tok(x_num)], dim=1)
        z = self.encoder(tokens)
        return self.head(z[:, 0]).squeeze(1)


def build_model_from_checkpoint(ckpt: dict, n_num: int) -> nn.Module:
    args = ckpt.get("args", {})
    model_name = args.get("model", "fttransformer")
    dropout = float(args.get("dropout", 0.15))
    if model_name == "mlp":
        return PairFeatureMLP(n_num=n_num, hidden=int(args.get("hidden", 64)), dropout=dropout)
    return PairFeatureFTTransformer(
        n_num=n_num,
        d_token=int(args.get("d_token", 32)),
        n_heads=int(args.get("n_heads", 4)),
        n_layers=int(args.get("n_layers", 2)),
        dropout=dropout,
    )


def binary_metrics(y_true, y_prob, threshold=0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
        out["ap"] = float(average_precision_score(y_true, y_prob))
    else:
        out["auroc"] = float("nan")
        out["ap"] = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out["tn"], out["fp"], out["fn"], out["tp"] = [int(x) for x in cm.ravel()]
    return out


def find_best_threshold(y_true, y_prob, metric="balanced_acc") -> Tuple[float, dict]:
    best_t, best_m, best = 0.5, -1.0, None
    for t in np.linspace(0.05, 0.95, 181):
        m = binary_metrics(y_true, y_prob, threshold=float(t))
        val = m.get(metric, np.nan)
        if np.isfinite(val) and val > best_m:
            best_t, best_m, best = float(t), float(val), m
    return best_t, best


def predict_one_checkpoint(ckpt_path: Path, raw_df: Optional[pd.DataFrame], feat_csv: str, diag_csv: str, batch_size: int, device) -> Tuple[pd.DataFrame, np.ndarray, float, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    feature_cols = ckpt.get("feature_cols") or ckpt["preprocessor"]["feature_cols"]
    if raw_df is None:
        df = load_pair_table(feat_csv, diag_csv, feature_cols)
    else:
        df = raw_df.copy()
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Input dataframe missing feature columns required by {ckpt_path}: {missing}")

    include_l5s1 = bool(ckpt.get("args", {}).get("include_l5s1", True))
    if not include_l5s1:
        df = df[df["pair_name"] != "L5-S1"].copy().reset_index(drop=True)

    preprocessor = LoadedPreprocessor(ckpt["preprocessor"])
    x_num = preprocessor.transform(df)
    loader = DataLoader(PairFeatureDataset(df, x_num), batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model_from_checkpoint(ckpt, n_num=len(feature_cols)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    probs = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["x_num"].to(device), batch["pair_id"].to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.concatenate(probs)
    threshold = float(ckpt.get("best_valid_threshold", 0.5))
    return df, probs, threshold, ckpt


def discover_checkpoints(args) -> List[Path]:
    paths: List[Path] = []
    if args.checkpoint:
        paths.append(Path(args.checkpoint))
    if args.checkpoint_list:
        paths += [Path(x.strip()) for x in args.checkpoint_list.split(",") if x.strip()]
    if args.checkpoint_dir:
        root = Path(args.checkpoint_dir)
        found = sorted(root.glob("fold_*/best.pt"), key=lambda p: str(p))
        if not found:
            found = sorted(root.glob("fold_*/best_pair_feature_model.pt"), key=lambda p: str(p))
        if not found and (root / "best.pt").exists():
            found = [root / "best.pt"]
        if not found and (root / "best_pair_feature_model.pt").exists():
            found = [root / "best_pair_feature_model.pt"]
        paths += found
    # de-duplicate while preserving order
    uniq = []
    seen = set()
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            uniq.append(p)
            seen.add(rp)
    if not uniq:
        raise ValueError("Provide --checkpoint, --checkpoint_list, or --checkpoint_dir containing fold_*/best.pt")
    for p in uniq:
        if not p.exists():
            raise FileNotFoundError(p)
    return uniq


def add_pair_prediction_columns(df: pd.DataFrame, probs: np.ndarray, threshold: float) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    out["pair_prob"] = probs
    out["pair_pred"] = (out["pair_prob"] >= threshold).astype(int)
    out["pair_correct"] = (out["pair_pred"] == out["pair_label"]).astype(int)
    out["pair_confusion_type"] = np.select(
        [
            (out["pair_label"] == 1) & (out["pair_pred"] == 1),
            (out["pair_label"] == 0) & (out["pair_pred"] == 0),
            (out["pair_label"] == 0) & (out["pair_pred"] == 1),
            (out["pair_label"] == 1) & (out["pair_pred"] == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="NA",
    )
    return out


def build_image_predictions(pair_pred_df: pd.DataFrame, threshold: float, image_prob_method: str = "noisy_or") -> pd.DataFrame:
    rows = []
    for image_id, g in pair_pred_df.groupby("image_id", sort=False):
        probs = np.clip(g["pair_prob"].to_numpy(dtype=float), 0.0, 1.0)
        noisy_or_prob = 1.0 - float(np.prod(1.0 - probs))
        max_prob = float(np.max(probs)) if len(probs) else 0.0
        mean_prob = float(np.mean(probs)) if len(probs) else 0.0
        if image_prob_method == "max":
            image_prob = max_prob
        elif image_prob_method == "mean":
            image_prob = mean_prob
        else:
            image_prob = noisy_or_prob

        study_vals = pd.to_numeric(g["study_label"], errors="coerce").dropna().unique()
        y = int(study_vals[0]) if len(study_vals) else int(g["pair_label"].max())
        pred = int(image_prob >= threshold)
        rows.append({
            "image_id": image_id,
            "image_label": y,
            "image_prob": image_prob,
            "image_pred": pred,
            "image_correct": int(pred == y),
            "image_confusion_type": "TP" if y == 1 and pred == 1 else "TN" if y == 0 and pred == 0 else "FP" if y == 0 and pred == 1 else "FN",
            "image_prob_method": image_prob_method,
            "noisy_or_prob": noisy_or_prob,
            "max_pair_prob": max_prob,
            "mean_pair_prob": mean_prob,
            "n_pairs": int(len(g)),
            "n_pair_pred_positive": int(g["pair_pred"].sum()),
            "n_pair_label_positive": int(g["pair_label"].sum()),
            "pred_positive_pairs": ";".join(g.loc[g["pair_pred"] == 1, "pair_name"].astype(str).tolist()),
            "gt_positive_pairs": ";".join(g.loc[g["pair_label"] == 1, "pair_name"].astype(str).tolist()),
            "location": g["location"].iloc[0] if "location" in g.columns else np.nan,
            "grade": g["grade"].iloc[0] if "grade" in g.columns else np.nan,
            "implant": g["implant"].iloc[0] if "implant" in g.columns else np.nan,
        })
    return pd.DataFrame(rows)


def write_summary_report(path: Path, split_name: str, checkpoint_paths: Sequence[Path], threshold: float, pair_metrics: dict, image_metrics: dict, pair_df: pd.DataFrame, image_df: pd.DataFrame, image_prob_method: str, ensemble_method: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"split: {split_name}\n")
        f.write(f"n_checkpoints: {len(checkpoint_paths)}\n")
        f.write(f"ensemble_method: {ensemble_method}\n")
        f.write(f"threshold: {threshold}\n")
        f.write(f"image_prob_method: {image_prob_method}\n")
        f.write("checkpoints:\n")
        for p in checkpoint_paths:
            f.write(f"  - {p}\n")
        f.write("\n=== Pair-level metrics ===\n")
        for k, v in pair_metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nPair-level confusion counts:\n")
        for key in ["TP", "TN", "FP", "FN"]:
            f.write(f"{key}: {int((pair_df['pair_confusion_type'] == key).sum())}\n")
        f.write("\n=== Image-level metrics ===\n")
        for k, v in image_metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nImage-level confusion counts:\n")
        for key in ["TP", "TN", "FP", "FN"]:
            f.write(f"{key}: {int((image_df['image_confusion_type'] == key).sum())}\n")
        f.write("\n=== Pair-level confusion by pair_name ===\n")
        f.write(pd.crosstab(pair_df["pair_name"], pair_df["pair_confusion_type"], dropna=False).to_string())
        f.write("\n\n=== Useful diagnostics ===\n")
        f.write(f"n_pair_rows: {len(pair_df)}\n")
        f.write(f"n_images: {image_df['image_id'].nunique()}\n")
        f.write(f"pair_positive_rate: {float(pair_df['pair_label'].mean())}\n")
        f.write(f"image_positive_rate: {float(image_df['image_label'].mean())}\n")
        f.write(f"mean_n_pair_pred_positive_per_image: {float(image_df['n_pair_pred_positive'].mean())}\n")
        f.write(f"num_images_with_any_pair_pred_positive: {int((image_df['n_pair_pred_positive'] > 0).sum())}\n")


def main():
    args = parse_args()
    ckpt_paths = discover_checkpoints(args)
    root_outdir = Path(args.outdir) if args.outdir else ckpt_paths[0].parent
    outdir = root_outdir / "test_analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Load the first checkpoint only to know feature columns and construct the base table.
    first_ckpt = torch.load(ckpt_paths[0], map_location=device)
    feature_cols = first_ckpt.get("feature_cols") or first_ckpt["preprocessor"]["feature_cols"]
    base_df = load_pair_table(args.feat_csv, args.diag_csv, feature_cols)

    probs_by_ckpt = []
    thresholds = []
    fold_cols = []
    ckpt_summaries = []
    for i, ckpt_path in enumerate(ckpt_paths):
        df_i, probs_i, threshold_i, ckpt_i = predict_one_checkpoint(
            ckpt_path, raw_df=base_df, feat_csv=args.feat_csv, diag_csv=args.diag_csv,
            batch_size=args.batch_size, device=device,
        )
        if len(df_i) != len(base_df) or not (df_i["image_id"].astype(str).values == base_df["image_id"].astype(str).values).all() or not (df_i["pair_name"].astype(str).values == base_df["pair_name"].astype(str).values).all():
            raise RuntimeError(f"Checkpoint {ckpt_path} produced a different row set/order. Check include_l5s1 consistency.")
        probs_by_ckpt.append(probs_i)
        thresholds.append(threshold_i)
        fold_name = f"fold{ckpt_i.get('fold', i)}"
        fold_cols.append(f"pair_prob_{fold_name}")
        ckpt_summaries.append({
            "checkpoint": str(ckpt_path),
            "fold": ckpt_i.get("fold", None),
            "best_valid_threshold": threshold_i,
            "best_epoch": ckpt_i.get("best_epoch", None),
            "best_monitor_score": ckpt_i.get("best_monitor_score", None),
        })

    prob_mat = np.vstack(probs_by_ckpt)  # [n_ckpt, n_rows]
    if args.ensemble_method == "median":
        probs = np.median(prob_mat, axis=0)
    else:
        probs = np.mean(prob_mat, axis=0)

    if args.threshold is not None:
        threshold = float(args.threshold)
    elif args.threshold_policy == "mean_checkpoint_threshold":
        threshold = float(np.mean(thresholds))
    elif args.threshold_policy == "median_checkpoint_threshold":
        threshold = float(np.median(thresholds))
    else:
        threshold = 0.5

    if args.tune_threshold_on_this_split:
        threshold, _ = find_best_threshold(base_df["pair_label"].values, probs, metric=args.threshold_metric)

    pair_pred_df = add_pair_prediction_columns(base_df, probs, threshold=threshold)
    for col, p in zip(fold_cols, probs_by_ckpt):
        pair_pred_df[col] = p
    pair_pred_df["pair_prob_std_across_checkpoints"] = np.std(prob_mat, axis=0)

    pair_metrics = binary_metrics(pair_pred_df["pair_label"].values, pair_pred_df["pair_prob"].values, threshold=threshold)
    image_pred_df = build_image_predictions(pair_pred_df, threshold=threshold, image_prob_method=args.image_prob_method)
    image_metrics = binary_metrics(image_pred_df["image_label"].values, image_pred_df["image_prob"].values, threshold=threshold)

    split = args.split_name
    pair_pred_df.to_csv(outdir / f"{split}_pair_predictions.csv", index=False)
    pair_correct_cols = [
        "image_id", "pair_name", "pair_label", "pair_prob", "pair_pred", "pair_correct", "pair_confusion_type",
        "pair_prob_std_across_checkpoints", "location", "grade", "implant",
    ] + fold_cols
    pair_pred_df[pair_correct_cols].to_csv(outdir / f"{split}_pair_correctness.csv", index=False)

    image_pred_df.to_csv(outdir / f"{split}_image_predictions.csv", index=False)
    image_pred_df[[
        "image_id", "image_label", "image_prob", "image_pred", "image_correct", "image_confusion_type",
        "noisy_or_prob", "max_pair_prob", "mean_pair_prob", "n_pairs", "n_pair_pred_positive",
        "pred_positive_pairs", "gt_positive_pairs", "location", "grade", "implant",
    ]].to_csv(outdir / f"{split}_image_correctness.csv", index=False)

    pd.crosstab(pair_pred_df["pair_name"], pair_pred_df["pair_confusion_type"], dropna=False).to_csv(outdir / f"{split}_pair_confusion_by_pair_name.csv")
    pd.crosstab(image_pred_df["image_confusion_type"], image_pred_df["n_pair_pred_positive"], dropna=False).to_csv(outdir / f"{split}_image_confusion_by_num_pred_positive_pairs.csv")

    with open(outdir / f"{split}_pair_metrics.json", "w", encoding="utf-8") as f:
        json.dump(pair_metrics, f, indent=2)
    with open(outdir / f"{split}_image_metrics.json", "w", encoding="utf-8") as f:
        json.dump(image_metrics, f, indent=2)

    summary = {
        "split": split,
        "n_checkpoints": len(ckpt_paths),
        "checkpoints": ckpt_summaries,
        "feature_cols": feature_cols,
        "threshold": threshold,
        "threshold_policy": args.threshold_policy if args.threshold is None else "manual",
        "ensemble_method": args.ensemble_method,
        "image_prob_method": args.image_prob_method,
        "pair_metrics": pair_metrics,
        "image_metrics": image_metrics,
        "n_pair_rows": int(len(pair_pred_df)),
        "n_images": int(image_pred_df["image_id"].nunique()),
    }
    with open(outdir / f"{split}_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_summary_report(
        outdir / f"{split}_summary_report.txt",
        split_name=split,
        checkpoint_paths=ckpt_paths,
        threshold=threshold,
        pair_metrics=pair_metrics,
        image_metrics=image_metrics,
        pair_df=pair_pred_df,
        image_df=image_pred_df,
        image_prob_method=args.image_prob_method,
        ensemble_method=args.ensemble_method,
    )
    print(json.dumps(summary, indent=2))



def _cfg_get(cfg, *keys, default=None):
    for k in keys:
        if k in cfg and cfg[k] is not None:
            return cfg[k]
    return default


def apply_config_defaults(args):
    cfg = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            import yaml
            cfg = yaml.safe_load(f) or {}

    args.checkpoint_dir = args.checkpoint_dir or _cfg_get(cfg, "checkpoint_dir", "save_dir", "outdir")
    args.feat_csv = args.feat_csv or _cfg_get(cfg, "test_feat_csv", "test_feature_csv")
    args.diag_csv = args.diag_csv or _cfg_get(cfg, "test_diag_csv", "test_csv")
    args.outdir = args.outdir or _cfg_get(cfg, "outdir", "save_dir")
    args.split_name = args.split_name or _cfg_get(cfg, "split_name", default="test")
    args.threshold_policy = args.threshold_policy or _cfg_get(cfg, "threshold_policy", default="mean_checkpoint_threshold")
    args.threshold_metric = args.threshold_metric or _cfg_get(cfg, "threshold_metric", default="balanced_acc")
    args.image_prob_method = args.image_prob_method or _cfg_get(cfg, "image_prob_method", default="max")
    args.ensemble_method = args.ensemble_method or _cfg_get(cfg, "ensemble_method", default="mean")
    args.batch_size = int(args.batch_size if args.batch_size is not None else _cfg_get(cfg, "analysis_batch_size", "batch_size", default=256))
    args.cpu = bool(args.cpu or _cfg_get(cfg, "cpu", default=False))

    missing = []
    if not (args.checkpoint or args.checkpoint_list or args.checkpoint_dir):
        missing.append("checkpoint/checkpoint_list/checkpoint_dir")
    for k in ["feat_csv", "diag_csv"]:
        if not getattr(args, k):
            missing.append(k)
    if missing:
        raise ValueError(f"Missing required analysis settings: {missing}. Provide them by CLI or --config.")
    return args


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="Optional YAML config; CLI args override config values.")
    p.add_argument("--checkpoint", type=str, default=None, help="Single checkpoint path, e.g. fold_0/best.pt")
    p.add_argument("--checkpoint_dir", type=str, default=None, help="Experiment root containing fold_*/best.pt")
    p.add_argument("--checkpoint_list", type=str, default=None, help="Comma-separated checkpoint paths")
    p.add_argument("--feat_csv", type=str, default=None, help="pair_crop_features.csv for the split to analyze")
    p.add_argument("--diag_csv", type=str, default=None, help="lumbar_diagnosis.csv for the split to analyze")
    p.add_argument("--outdir", type=str, default=None, help="Experiment root. Outputs go to <outdir>/test_analysis.")
    p.add_argument("--split_name", type=str, default=None)
    p.add_argument("--threshold", type=float, default=None, help="Manual threshold override")
    p.add_argument("--threshold_policy", type=str, default=None, choices=["mean_checkpoint_threshold", "median_checkpoint_threshold", "fixed_0.5"])
    p.add_argument("--tune_threshold_on_this_split", action="store_true", help="Debug only; do not use for final test reporting")
    p.add_argument("--threshold_metric", type=str, default=None, choices=["balanced_acc", "f1", "accuracy"])
    p.add_argument("--image_prob_method", type=str, default=None, choices=["noisy_or", "max", "mean"])
    p.add_argument("--ensemble_method", type=str, default=None, choices=["mean", "median"])
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    return apply_config_defaults(p.parse_args())


if __name__ == "__main__":
    main()
