#!/usr/bin/env python3
"""Analyze Stage 2 local pair diagnostic CV ensemble on the held-out test split."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from data.resnet50.dataset_local_pair_diagnostic import PAIR_NAMES, DEFAULT_GEOMETRY_FEATURES, LocalPairDiagnosticDataset, local_pair_collate_fn
from models.local_pair_diagnostic import LocalPairDiagnosticModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--cv_dir', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--ckpt_name', default='best.pt')
    p.add_argument('--threshold', type=float, default=None)
    return p.parse_args()


def safe_binary_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= float(threshold)).astype(int)
    out = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_acc': float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) == 2 else np.nan,
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try: out['auroc'] = float(roc_auc_score(y_true, y_prob))
    except Exception: out['auroc'] = np.nan
    try: out['ap'] = float(average_precision_score(y_true, y_prob))
    except Exception: out['ap'] = np.nan
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        out.update({'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp), 'specificity': float(tn / max(tn + fp, 1)), 'sensitivity': float(tp / max(tp + fn, 1))})
    return out


def load_model_from_ckpt(ckpt_path, cfg, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    feature_names = ckpt.get('geometry_feature_names', cfg.get('geometry_feature_names', DEFAULT_GEOMETRY_FEATURES))
    model = LocalPairDiagnosticModel(
        geometry_dim=len(feature_names),
        backbone_name=str(cfg.get('backbone_name', 'resnet18')),
        backbone_pretrained=False,
        token_dim=int(cfg.get('token_dim', 256)),
        geo_hidden_dim=int(cfg.get('geo_hidden_dim', 64)),
        transformer_layers=int(cfg.get('transformer_layers', 2)),
        transformer_heads=int(cfg.get('transformer_heads', 4)),
        dropout=float(cfg.get('dropout', 0.1)),
    ).to(device)
    state = ckpt.get('model', ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt)))
    model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=True)
    geo_mean = torch.tensor(ckpt['geometry_mean'], dtype=torch.float32, device=device).view(1, 1, -1)
    geo_std = torch.tensor(ckpt['geometry_std'], dtype=torch.float32, device=device).view(1, 1, -1)
    return model, geo_mean, geo_std, feature_names


@torch.no_grad()
def predict_fold(model, loader, device, geo_mean, geo_std):
    model.eval()
    case_rows = []
    pair_rows = []
    for batch in loader:
        images = batch['pair_images'].to(device, non_blocking=True)
        geom = torch.nan_to_num(batch['geometry_features'].to(device, non_blocking=True).float(), nan=0.0, posinf=0.0, neginf=0.0)
        geom = (geom - geo_mean) / geo_std
        out = model(images, geom)
        probs = out['location_probs'].detach().cpu().numpy()
        slip = out['slip_percentage'].detach().cpu().numpy()
        y_pair = batch['pair_labels'].cpu().numpy()
        y_case = batch['study_label'].cpu().numpy()
        slip_target = batch['slip_targets'].cpu().numpy()
        slip_mask = batch['slip_loss_mask'].cpu().numpy()
        slip_outlier = batch['slip_outlier'].cpu().numpy()
        for b, image_id in enumerate(batch['image_ids']):
            mil = float(1.0 - np.prod(1.0 - np.clip(probs[b], 1e-6, 1.0 - 1e-6)))
            top_idx = int(np.argmax(probs[b]))
            case_rows.append({'image_id': image_id, 'y_true': int(y_case[b]), 'y_prob_pair_mil': mil, 'top_pair_name': PAIR_NAMES[top_idx], 'top_pair_prob': float(probs[b, top_idx]), 'top_pair_slip_pred': float(slip[b, top_idx]), 'location_raw': batch['location_raw'][b]})
            for p, pair_name in enumerate(PAIR_NAMES):
                pair_rows.append({'image_id': image_id, 'pair_name': pair_name, 'pair_idx': p, 'study_y_true': int(y_case[b]), 'pair_y_true': int(y_pair[b, p]), 'pair_prob': float(probs[b, p]), 'slip_pred': float(slip[b, p]), 'slip_target': float(slip_target[b, p]), 'slip_loss_mask': int(slip_mask[b, p] > 0.5), 'slip_outlier': int(slip_outlier[b, p] > 0.5)})
    return pd.DataFrame(case_rows), pd.DataFrame(pair_rows)


def aggregate_pair_predictions(pair_dfs):
    key_cols = ['image_id', 'pair_name', 'pair_idx', 'study_y_true', 'pair_y_true', 'slip_target', 'slip_loss_mask', 'slip_outlier']
    base = pair_dfs[0][key_cols].copy()
    prob_stack = np.stack([df['pair_prob'].to_numpy(float) for df in pair_dfs], axis=0)
    slip_stack = np.stack([df['slip_pred'].to_numpy(float) for df in pair_dfs], axis=0)
    base['pair_prob'] = prob_stack.mean(axis=0)
    base['slip_pred'] = slip_stack.mean(axis=0)
    return base


def slip_to_meyerding_grade(values):
    values = np.asarray(values, dtype=float)
    grades = np.zeros(values.shape, dtype=int)
    valid = np.isfinite(values) & (values > 0)
    grades[valid & (values <= 25.0)] = 1
    grades[valid & (values > 25.0) & (values <= 50.0)] = 2
    grades[valid & (values > 50.0) & (values <= 75.0)] = 3
    grades[valid & (values > 75.0)] = 4
    return grades


def add_slip_grade_columns(pair_df, threshold):
    out = pair_df.copy()
    out['slip_grade_target'] = 0
    out['slip_grade_pred_regression'] = slip_to_meyerding_grade(out['slip_pred'].to_numpy(float))
    pos_mask = out['pair_y_true'].astype(int) == 1
    out.loc[pos_mask, 'slip_grade_target'] = slip_to_meyerding_grade(out.loc[pos_mask, 'slip_target'].to_numpy(float))
    out['slip_grade_pred_joint'] = 0
    pred_pos_mask = out['pair_prob'].to_numpy(float) >= float(threshold)
    out.loc[pred_pos_mask, 'slip_grade_pred_joint'] = out.loc[pred_pos_mask, 'slip_grade_pred_regression']
    return out


def grade_agreement_metrics(y_true, y_pred, prefix):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    out = {f'{prefix}_n': int(len(y_true))}
    if len(y_true) == 0:
        out.update({
            f'{prefix}_acc': np.nan,
            f'{prefix}_within1_acc': np.nan,
            f'{prefix}_mae_grade': np.nan,
            f'{prefix}_quadratic_kappa': np.nan,
        })
        return out
    out[f'{prefix}_acc'] = float(np.mean(y_true == y_pred))
    out[f'{prefix}_within1_acc'] = float(np.mean(np.abs(y_true - y_pred) <= 1))
    out[f'{prefix}_mae_grade'] = float(np.mean(np.abs(y_true - y_pred)))
    try:
        out[f'{prefix}_quadratic_kappa'] = float(cohen_kappa_score(y_true, y_pred, weights='quadratic', labels=[0, 1, 2, 3, 4]))
    except Exception:
        out[f'{prefix}_quadratic_kappa'] = np.nan
    return out


def location_set_metrics(pair_df, threshold):
    exact_all = []
    exact_pos = []
    pred_set_hit_pos = []
    top1_hit_pos = []
    pred_counts = []
    true_counts = []

    for _, group in pair_df.groupby('image_id', sort=False):
        group = group.sort_values('pair_idx')
        true_mask = group['pair_y_true'].to_numpy(int) == 1
        probs = group['pair_prob'].to_numpy(float)
        pred_mask = probs >= float(threshold)

        true_set = set(np.flatnonzero(true_mask).tolist())
        pred_set = set(np.flatnonzero(pred_mask).tolist())
        top_idx = int(np.argmax(probs))
        exact = pred_set == true_set

        exact_all.append(exact)
        pred_counts.append(len(pred_set))
        true_counts.append(len(true_set))
        if true_set:
            exact_pos.append(exact)
            pred_set_hit_pos.append(len(pred_set & true_set) > 0)
            top1_hit_pos.append(top_idx in true_set)

    return {
        'location_exact_all': float(np.mean(exact_all)) if exact_all else np.nan,
        'location_exact_pos': float(np.mean(exact_pos)) if exact_pos else np.nan,
        'location_pred_set_hit_pos': float(np.mean(pred_set_hit_pos)) if pred_set_hit_pos else np.nan,
        'location_top1_hit_pos': float(np.mean(top1_hit_pos)) if top1_hit_pos else np.nan,
        'location_mean_predicted_positive_pairs_per_case': float(np.mean(pred_counts)) if pred_counts else np.nan,
        'location_mean_true_positive_pairs_per_case': float(np.mean(true_counts)) if true_counts else np.nan,
    }


def build_case_from_pair(pair_df):
    rows = []
    for image_id, g in pair_df.groupby('image_id', sort=False):
        probs = g.sort_values('pair_idx')['pair_prob'].to_numpy(float)
        mil = float(1.0 - np.prod(1.0 - np.clip(probs, 1e-6, 1.0 - 1e-6)))
        top = g.loc[g['pair_prob'].idxmax()]
        rows.append({'image_id': image_id, 'y_true': int(g['study_y_true'].iloc[0]), 'y_prob_pair_mil': mil, 'top_pair_name': top['pair_name'], 'top_pair_prob': float(top['pair_prob']), 'top_pair_slip_pred': float(top['slip_pred'])})
    return pd.DataFrame(rows)


def summarize(case_df, pair_df, threshold):
    out = {f'case_mil_{k}': v for k, v in safe_binary_metrics(case_df['y_true'], case_df['y_prob_pair_mil'], threshold).items()}
    out.update({f'pair_{k}': v for k, v in safe_binary_metrics(pair_df['pair_y_true'], pair_df['pair_prob'], threshold).items()})
    for pair_name in PAIR_NAMES:
        g = pair_df[pair_df['pair_name'] == pair_name]
        m = safe_binary_metrics(g['pair_y_true'], g['pair_prob'], threshold)
        out[f'{pair_name}_ap'] = m.get('ap', np.nan)
        out[f'{pair_name}_sensitivity'] = m.get('sensitivity', np.nan)
        out[f'{pair_name}_specificity'] = m.get('specificity', np.nan)
        out[f'{pair_name}_f1'] = m.get('f1', np.nan)
    correct = total = 0
    for _, g in pair_df.groupby('image_id', sort=False):
        if int(g['pair_y_true'].max()) != 1:
            continue
        top = g.loc[g['pair_prob'].idxmax()]
        correct += int(top['pair_y_true'] == 1)
        total += 1
    out['top1_positive_location_acc'] = float(correct / max(total, 1)) if total else np.nan
    out.update(location_set_metrics(pair_df, threshold))
    pair_df = add_slip_grade_columns(pair_df, threshold)
    slip_eval = pair_df[pair_df['slip_loss_mask'] == 1]
    out['slip_mae_positive_nonoutlier'] = float(np.mean(np.abs(slip_eval['slip_pred'] - slip_eval['slip_target']))) if len(slip_eval) else np.nan
    out['n_slip_eval_pairs'] = int(len(slip_eval))
    out.update(grade_agreement_metrics(
        slip_eval['slip_grade_target'],
        slip_eval['slip_grade_pred_regression'],
        'slip_grade_positive_nonoutlier',
    ))
    out.update(grade_agreement_metrics(
        pair_df['slip_grade_target'],
        pair_df['slip_grade_pred_joint'],
        'joint_pair_grade_all_pairs',
    ))
    return out


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8')) or {}
    threshold = float(args.threshold if args.threshold is not None else cfg.get('threshold', 0.5))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feature_names = cfg.get('geometry_feature_names', DEFAULT_GEOMETRY_FEATURES)
    test_ds = LocalPairDiagnosticDataset(cfg['test_csv'], cfg['test_pair_image_dir'], cfg['test_feature_csv'], split='test', aug_cfg=cfg.get('augmentation', {}), geometry_feature_names=feature_names, require_all_pairs=bool(cfg.get('require_all_pairs', True)), slip_column=cfg.get('slip_column', 'estimated_slip_percentage'), slip_raw_column=cfg.get('slip_raw_column', 'estimated_slip_percentage_raw'), slip_outlier_column=cfg.get('slip_outlier_column', 'estimated_slip_percentage_outlier'), exclude_slip_outliers=bool(cfg.get('exclude_slip_outliers', True)))
    loader = DataLoader(test_ds, batch_size=int(cfg.get('batch_size', 16)), shuffle=False, num_workers=int(cfg.get('num_workers', 4)), collate_fn=local_pair_collate_fn, pin_memory=True)
    pair_dfs = []
    used = []
    for fold in range(1, args.n_folds + 1):
        ckpt_path = Path(args.cv_dir) / f'fold_{fold}' / args.ckpt_name
        if not ckpt_path.exists():
            print(f'[WARN] missing checkpoint: {ckpt_path}')
            continue
        model, geo_mean, geo_std, _ = load_model_from_ckpt(ckpt_path, cfg, device)
        _, pair_df = predict_fold(model, loader, device, geo_mean, geo_std)
        pair_df['fold'] = fold
        pair_dfs.append(pair_df.drop(columns=['fold']))
        used.append(str(ckpt_path))
        print(f'[INFO] loaded fold {fold}: {ckpt_path}')
    if not pair_dfs:
        raise RuntimeError('No fold checkpoints loaded.')
    pair_df = aggregate_pair_predictions(pair_dfs)
    case_df = build_case_from_pair(pair_df)
    metrics = summarize(case_df, pair_df, threshold)
    pair_df = add_slip_grade_columns(pair_df, threshold)
    pair_df.to_csv(Path(args.outdir) / 'local_pair_diagnostic_pair_predictions.csv', index=False)
    case_df.to_csv(Path(args.outdir) / 'local_pair_diagnostic_case_predictions.csv', index=False)
    pd.DataFrame([metrics]).to_csv(Path(args.outdir) / 'local_pair_diagnostic_metrics.csv', index=False)
    with open(Path(args.outdir) / 'local_pair_diagnostic_summary.txt', 'w', encoding='utf-8') as f:
        f.write('Branch: local_pair_diagnostic_cv_ensemble\n')
        f.write(f'threshold: {threshold}\n')
        f.write(f'used_checkpoints: {used}\n\n')
        for k, v in metrics.items():
            f.write(f'{k}: {v}\n')
    print(metrics)


if __name__ == '__main__':
    main()
