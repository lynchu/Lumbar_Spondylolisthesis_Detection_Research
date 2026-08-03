#!/usr/bin/env python3
"""
5-fold cross-validation trainer for pair-level tabular DNN lumbar spondylolisthesis prediction.

Behavior requested:
- Load original train + valid pair_crop_features.csv and lumbar_diagnosis.csv.
- Rebuild pair_label from lumbar_diagnosis.csv location.
- Concatenate train and valid into one combined dataset.
- Split combined images into 5 folds by image_id, stratified by image-level label.
- For each fold, train on 4 folds and validate on 1 fold.
- For each fold, save ONLY: <outdir>/fold_<k>/best.pt
- Do not save validation metrics, train history, validation predictions, or CV summaries.

Example:
python ./presence/scripts/pair_feature/train_pair_feature_cv.py  
 --train_feat_csv ./dataset/train/pair_crop_features.csv   
 --valid_feat_csv ./dataset/valid/pair_crop_features.csv   
 --train_diag_csv ./dataset/train/lumbar_diagnosis.csv  
 --valid_diag_csv ./dataset/valid/lumbar_diagnosis.csv   
 --outdir ./presence/experiments/fttransformer_11_pair_feature_cv5   
 --model fttransformer   --epochs 200 --batch_size 128 --lr 5e-4 --weight_decay 5e-5 --dropout 0.20 --patience 30
 
"""

import argparse
import random
import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset


SELECTED_FEATURES = [
    # core absolute horizontal slip
    "abs_dx_slip_px",
    "abs_left_edge_dx_px",
    "abs_right_edge_dx_px",
    "abs_dx_slip_ratio_wavg",
    "abs_left_edge_dx_ratio_wavg",
    "abs_right_edge_dx_ratio_wavg",

    # signed horizontal slip, useful with pair_name embedding
    "dx_slip_px",
    "left_edge_dx_px",
    "right_edge_dx_px",

    # weak vertical/context signal
    "dy_slip_px",
    "abs_dy_slip_px",

    # local scale/context
    # "pair_interface_width",
    # "pair_reference_width",
]
OPTIONAL_EXTENDED_FEATURES = ["abs_dy_slip_px", "dy_slip_px"]
PAIR_NAMES = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
PAIR_TO_ID = {p: i for i, p in enumerate(PAIR_NAMES)}
UNK_PAIR_ID = len(PAIR_NAMES)
EPS = 1e-8


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
    """Parse 'L3,L4; L4,L5;', 'L5-S1', or old level-list 'L3; L4'."""
    if location is None or pd.isna(location):
        return set()
    s = str(location).strip().upper().replace(" ", "")
    if not s or s == "NAN":
        return set()

    pair_set = set()
    # Explicit adjacent-pair chunks.
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

    # Old level-list format.
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


def load_pair_table(feat_csv: str, diag_csv: str, feature_cols: Sequence[str], source_split: str) -> pd.DataFrame:
    feat = pd.read_csv(feat_csv, dtype={"image_id": str})
    diag = pd.read_csv(diag_csv, dtype={"image_id": str})

    feat["image_id"] = feat["image_id"].astype(str).str.strip()
    diag["image_id"] = diag["image_id"].astype(str).str.strip()

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
    df["source_split"] = source_split

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{feat_csv} missing selected feature columns: {missing}")

    df = df[df["pair_name"].isin(PAIR_NAMES)].copy()

    if df["study_label"].isna().any():
        bad = df.loc[df["study_label"].isna(), "image_id"].drop_duplicates().head(5).tolist()
        raise ValueError(f"Missing spondylolisthesis_label for some images, examples={bad}")

    return df


class TabularPreprocessor:
    def __init__(self, feature_cols: Sequence[str], clip_quantile: float = 0.005):
        self.feature_cols = list(feature_cols)
        self.clip_quantile = float(clip_quantile)
        self.median_: Dict[str, float] = {}
        self.mean_: Dict[str, float] = {}
        self.std_: Dict[str, float] = {}
        self.lo_: Dict[str, float] = {}
        self.hi_: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        for c in self.feature_cols:
            s = pd.to_numeric(df[c], errors="coerce").astype(float)
            med = float(s.median()) if s.notna().any() else 0.0
            s = s.fillna(med)
            lo = float(s.quantile(self.clip_quantile))
            hi = float(s.quantile(1.0 - self.clip_quantile))
            s = s.clip(lo, hi)
            mean = float(s.mean())
            std = float(s.std())
            if not np.isfinite(std) or std < EPS:
                std = 1.0
            self.median_[c] = med
            self.lo_[c] = lo
            self.hi_[c] = hi
            self.mean_[c] = mean
            self.std_[c] = std
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        arrs = []
        for c in self.feature_cols:
            s = pd.to_numeric(df[c], errors="coerce").astype(float)
            s = s.fillna(self.median_[c]).clip(self.lo_[c], self.hi_[c])
            s = (s - self.mean_[c]) / self.std_[c]
            arrs.append(s.to_numpy(dtype=np.float32))
        return np.stack(arrs, axis=1).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "feature_cols": self.feature_cols,
            "clip_quantile": self.clip_quantile,
            "median": self.median_,
            "mean": self.mean_,
            "std": self.std_,
            "lo": self.lo_,
            "hi": self.hi_,
        }


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


def build_model(args, n_num: int) -> nn.Module:
    if args.model == "mlp":
        return PairFeatureMLP(n_num=n_num, hidden=args.hidden, dropout=args.dropout)
    return PairFeatureFTTransformer(n_num=n_num, d_token=args.d_token, n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout)


def binary_metrics(y_true, y_prob, threshold=0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(cm[0]), "fp": int(cm[1]), "fn": int(cm[2]), "tp": int(cm[3]),
    }
    if len(np.unique(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
        out["ap"] = float(average_precision_score(y_true, y_prob))
    else:
        out["auroc"] = float("nan")
        out["ap"] = float("nan")
    return out


def find_best_threshold(y_true, y_prob, metric="balanced_acc") -> Tuple[float, dict]:
    best_t, best_v, best_m = 0.5, -1.0, None
    for t in np.linspace(0.05, 0.95, 181):
        m = binary_metrics(y_true, y_prob, threshold=float(t))
        v = m.get(metric, np.nan)
        if np.isfinite(v) and v > best_v:
            best_t, best_v, best_m = float(t), float(v), m
    return best_t, best_m


def predict(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["x_num"].to(device), batch["pair_id"].to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            ys.append(batch["y"].cpu().numpy())
    return np.concatenate(ys), np.concatenate(probs)


def image_level_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for image_id, g in df.groupby("image_id", sort=False):
        study_vals = pd.to_numeric(g["study_label"], errors="coerce").dropna().unique()
        y = int(study_vals[0]) if len(study_vals) else int(g["pair_label"].max())
        rows.append({"image_id": image_id, "study_label": y})
    return pd.DataFrame(rows)


def make_cv_splits(combined_df: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    image_df = image_level_table(combined_df).sort_values("image_id").reset_index(drop=True)
    y = image_df["study_label"].astype(int).to_numpy()
    min_class = int(pd.Series(y).value_counts().min())
    if min_class < n_splits:
        raise ValueError(f"Cannot run {n_splits}-fold CV: smallest image-level class has only {min_class} images")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    image_df["fold"] = -1
    for fold_idx, (_, valid_idx) in enumerate(skf.split(image_df["image_id"].to_numpy(), y)):
        image_df.loc[valid_idx, "fold"] = fold_idx
    return image_df[["image_id", "fold"]]


def train_one_fold(args, train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_cols: Sequence[str], fold_idx: int) -> None:
    seed_everything(args.seed + fold_idx)
    fold_dir = Path(args.outdir) / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = fold_dir / "best.pt"

    prep = TabularPreprocessor(feature_cols, clip_quantile=args.clip_quantile).fit(train_df)
    train_loader = DataLoader(PairFeatureDataset(train_df, prep.transform(train_df)), batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    valid_loader = DataLoader(PairFeatureDataset(valid_df, prep.transform(valid_df)), batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_model(args, n_num=len(feature_cols)).to(device)

    y_train = train_df["pair_label"].to_numpy(dtype=np.float32)
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_score = -1.0
    best_state = None
    best_threshold = 0.5
    best_epoch = -1
    bad = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            logits = model(batch["x_num"].to(device), batch["pair_id"].to(device))
            y = batch["y"].to(device)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()

        yv, pv = predict(model, valid_loader, device)
        m05 = binary_metrics(yv, pv, threshold=0.5)
        threshold, mbt = find_best_threshold(yv, pv, metric=args.threshold_metric)
        monitor = m05[args.monitor_metric]
        if monitor > best_score:
            best_score = float(monitor)
            best_threshold = float(threshold)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if epoch == 1 or epoch % args.log_every == 0:
            print(
                f"fold={fold_idx} epoch={epoch:03d} loss={np.mean(losses):.4f} "
                f"valid_{args.monitor_metric}={monitor:.4f} valid_ap={m05['ap']:.4f} "
                f"valid_bal@0.5={m05['balanced_acc']:.4f} best_t_now={threshold:.3f}"
            )
        if bad >= args.patience:
            print(f"[EARLY STOP] fold={fold_idx} epoch={epoch}, best_valid_{args.monitor_metric}={best_score:.4f}")
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Save only best.pt for this fold. No metrics/history/predictions are written.
    torch.save(
        {
            "model_state": best_state,
            "args": vars(args),
            "preprocessor": prep.to_dict(),
            "pair_names": PAIR_NAMES,
            "feature_cols": list(feature_cols),
            "best_valid_threshold": float(best_threshold),
            "fold": int(fold_idx),
            "best_epoch": int(best_epoch),
            "monitor_metric": args.monitor_metric,
            "best_monitor_score": float(best_score),
        },
        ckpt_path,
    )
    print(f"[OK] fold={fold_idx} saved {ckpt_path}")


def run_cv(args):
    seed_everything(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    feature_cols = SELECTED_FEATURES.copy()
    if args.use_extended_features:
        feature_cols += OPTIONAL_EXTENDED_FEATURES
    if args.feature_cols:
        feature_cols = [x.strip() for x in args.feature_cols.split(",") if x.strip()]

    train_df = load_pair_table(args.train_feat_csv, args.train_diag_csv, feature_cols, source_split="original_train")
    valid_df = load_pair_table(args.valid_feat_csv, args.valid_diag_csv, feature_cols, source_split="original_valid")
    combined_df = pd.concat([train_df, valid_df], ignore_index=True)
    if not args.include_l5s1:
        combined_df = combined_df[combined_df["pair_name"] != "L5-S1"].copy()
    combined_df = combined_df.reset_index(drop=True)

    folds = make_cv_splits(combined_df, n_splits=args.n_splits, seed=args.seed)
    combined_df = combined_df.merge(folds, on="image_id", how="left")
    if combined_df["fold"].isna().any():
        raise RuntimeError("Some rows were not assigned to a CV fold")
    combined_df["fold"] = combined_df["fold"].astype(int)

    print(f"[INFO] combined_rows={len(combined_df)} combined_images={combined_df['image_id'].nunique()} n_splits={args.n_splits}")
    print(f"[INFO] output pattern: {outdir}/fold_<k>/best.pt")

    for fold_idx in range(args.n_splits):
        print(f"\n========== CV fold {fold_idx + 1}/{args.n_splits} ==========")
        fold_train = combined_df[combined_df["fold"] != fold_idx].copy()
        fold_valid = combined_df[combined_df["fold"] == fold_idx].copy()
        print(
            f"[INFO] fold={fold_idx} train_rows={len(fold_train)} valid_rows={len(fold_valid)} "
            f"train_images={fold_train['image_id'].nunique()} valid_images={fold_valid['image_id'].nunique()}"
        )
        train_one_fold(args, fold_train, fold_valid, feature_cols, fold_idx)

    print(f"[OK] CV training complete. Only best.pt files were saved under {outdir}/fold_*/")



def _cfg_get(cfg, *keys, default=None):
    for k in keys:
        if k in cfg and cfg[k] is not None:
            return cfg[k]
    return default


def _bool_from_cfg_or_arg(arg_value, cfg, *keys, default=True):
    if arg_value is not None:
        return bool(arg_value)
    for k in keys:
        if k in cfg and cfg[k] is not None:
            return bool(cfg[k])
    return bool(default)


def apply_config_defaults(args):
    cfg = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            import yaml
            cfg = yaml.safe_load(f) or {}

    args.train_feat_csv = args.train_feat_csv or _cfg_get(cfg, "train_feat_csv", "train_feature_csv")
    args.valid_feat_csv = args.valid_feat_csv or _cfg_get(cfg, "valid_feat_csv", "val_feat_csv", "valid_feature_csv", "val_feature_csv")
    args.train_diag_csv = args.train_diag_csv or _cfg_get(cfg, "train_diag_csv", "train_csv")
    args.valid_diag_csv = args.valid_diag_csv or _cfg_get(cfg, "valid_diag_csv", "val_diag_csv", "valid_csv", "val_csv")
    args.outdir = args.outdir or _cfg_get(cfg, "outdir", "save_dir")

    args.n_splits = int(args.n_splits if args.n_splits is not None else _cfg_get(cfg, "n_splits", "num_folds", default=5))
    args.model = args.model or _cfg_get(cfg, "model", default="fttransformer")

    if args.feature_cols is None:
        feature_cols = _cfg_get(cfg, "feature_cols", "feature_names")
        if isinstance(feature_cols, (list, tuple)):
            args.feature_cols = ",".join(str(x) for x in feature_cols)
        elif isinstance(feature_cols, str):
            args.feature_cols = feature_cols

    args.use_extended_features = bool(args.use_extended_features or _cfg_get(cfg, "use_extended_features", default=False))
    # Prefer explicit include_l5s1. Otherwise map include_s1_pair_features to the same decision.
    if args.include_l5s1 is None:
        args.include_l5s1 = bool(_cfg_get(cfg, "include_l5s1", "include_s1_pair_features", default=True))

    for name, keys, cast, default in [
        ("clip_quantile", ("clip_quantile",), float, 0.005),
        ("epochs", ("epochs",), int, 200),
        ("batch_size", ("batch_size",), int, 128),
        ("lr", ("lr",), float, 1e-3),
        ("weight_decay", ("weight_decay",), float, 1e-4),
        ("dropout", ("dropout",), float, 0.15),
        ("hidden", ("hidden",), int, 64),
        ("d_token", ("d_token",), int, 32),
        ("n_heads", ("n_heads",), int, 4),
        ("n_layers", ("n_layers",), int, 2),
        ("grad_clip", ("grad_clip",), float, 1.0),
        ("patience", ("patience", "early_stop_patience"), int, 30),
        ("log_every", ("log_every",), int, 10),
        ("seed", ("seed",), int, 42),
    ]:
        cur = getattr(args, name)
        if cur is None:
            setattr(args, name, cast(_cfg_get(cfg, *keys, default=default)))

    args.monitor_metric = args.monitor_metric or _cfg_get(cfg, "monitor_metric", default="ap")
    args.threshold_metric = args.threshold_metric or _cfg_get(cfg, "threshold_metric", default="balanced_acc")
    args.cpu = bool(args.cpu or _cfg_get(cfg, "cpu", default=False))

    required = ["train_feat_csv", "valid_feat_csv", "train_diag_csv", "valid_diag_csv", "outdir"]
    missing = [k for k in required if not getattr(args, k)]
    if missing:
        raise ValueError(f"Missing required train settings: {missing}. Provide them by CLI or --config.")
    return args


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="Optional YAML config; CLI args override config values.")
    p.add_argument("--train_feat_csv", type=str, default=None)
    p.add_argument("--valid_feat_csv", type=str, default=None)
    p.add_argument("--train_diag_csv", type=str, default=None)
    p.add_argument("--valid_diag_csv", type=str, default=None)
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--n_splits", type=int, default=None)
    p.add_argument("--model", type=str, default=None, choices=["fttransformer", "mlp"])
    p.add_argument("--feature_cols", type=str, default=None, help="comma-separated custom feature columns")
    p.add_argument("--use_extended_features", action="store_true")
    p.add_argument("--include_l5s1", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--clip_quantile", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--d_token", type=int, default=None)
    p.add_argument("--n_heads", type=int, default=None)
    p.add_argument("--n_layers", type=int, default=None)
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--monitor_metric", type=str, default=None, choices=["ap", "auroc", "balanced_acc", "f1"])
    p.add_argument("--threshold_metric", type=str, default=None, choices=["balanced_acc", "f1", "accuracy"])
    p.add_argument("--log_every", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    return apply_config_defaults(p.parse_args())


if __name__ == "__main__":
    run_cv(parse_args())
