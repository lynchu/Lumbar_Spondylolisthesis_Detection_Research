import json
import os
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_metric,
    scheduler=None,
    scaler=None,
    early_stopping=None,
    extra_state: Optional[Dict[str, Any]] = None,
):
    """
    Save a resumable training checkpoint.

    Required fields keep backward compatibility with the old version:
      - epoch
      - model
      - optimizer
      - best_metric

    Extra states are optional, but recommended for true resume training.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "best_metric": best_metric,
    }

    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    if early_stopping is not None:
        ckpt["early_stopping"] = {
            "patience": early_stopping.patience,
            "mode": early_stopping.mode,
            "min_delta": early_stopping.min_delta,
            "best_score": early_stopping.best_score,
            "num_bad_epochs": early_stopping.num_bad_epochs,
            "should_stop": early_stopping.should_stop,
        }
    if extra_state is not None:
        ckpt["extra_state"] = extra_state

    torch.save(ckpt, path)



def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    early_stopping=None,
    map_location="cpu",
):
    """
    Load a checkpoint and optionally restore optimizer / scheduler / scaler /
    early stopping states.
    """
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    if early_stopping is not None and "early_stopping" in ckpt:
        es = ckpt["early_stopping"]
        early_stopping.best_score = es.get("best_score")
        early_stopping.num_bad_epochs = es.get("num_bad_epochs", 0)
        early_stopping.should_stop = es.get("should_stop", False)

    return ckpt



def save_cv_summary(exp_dir: str, summary: Dict[str, Any], summary_filename: str = "cv_summary.json"):
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, summary_filename), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)



def _has_any_file(directory: str, filenames) -> bool:
    return any(os.path.isfile(os.path.join(directory, name)) for name in filenames)



def inspect_cv_run(
    exp_dir: str,
    n_folds: int = 5,
    fold_prefix: str = "fold_",
    summary_filename: str = "cv_summary.json",
    resume_ckpt_names = ("last.pt", "last.ckpt", "resume.pt"),
    fold_done_files = ("best.pt", "metrics.json", "fold_result.json"),
    fold_start_index: int = 1,
):
    """
    Inspect an experiment directory and decide whether to:
      1) stop because CV is already complete,
      2) resume an interrupted fold,
      3) start the next missing fold,
      4) start a brand-new run.

    Returns a dict with:
      - status: one of [already_completed, resume_fold, start_fold, new_run]
      - fold_idx: fold index to train next (or None)
      - fold_dir: resolved fold directory path (or None)
      - message: human-readable message
      - completed_folds / interrupted_folds
    """
    summary_path = os.path.join(exp_dir, summary_filename)
    if os.path.isfile(summary_path):
        return {
            "status": "already_completed",
            "fold_idx": None,
            "fold_dir": None,
            "resume_checkpoint": None,
            "completed_folds": list(range(fold_start_index, fold_start_index + n_folds)),
            "interrupted_folds": [],
            "message": f"Experiment directory already exists and {summary_filename} is present: {exp_dir}",
        }

    if not os.path.isdir(exp_dir):
        return {
            "status": "new_run",
            "fold_idx": fold_start_index,
            "fold_dir": os.path.join(exp_dir, f"{fold_prefix}{fold_start_index}"),
            "resume_checkpoint": None,
            "completed_folds": [],
            "interrupted_folds": [],
            "message": f"No existing CV directory found. Start a new run at fold {fold_start_index}.",
        }

    completed_folds = []
    interrupted_folds = []

    for fold_idx in range(fold_start_index, fold_start_index + n_folds):
        fold_dir = os.path.join(exp_dir, f"{fold_prefix}{fold_idx}")
        if not os.path.isdir(fold_dir):
            continue

        done = _has_any_file(fold_dir, fold_done_files)
        resume_ckpt = None
        for name in resume_ckpt_names:
            candidate = os.path.join(fold_dir, name)
            if os.path.isfile(candidate):
                resume_ckpt = candidate
                break

        if done:
            completed_folds.append(fold_idx)
        elif resume_ckpt is not None:
            interrupted_folds.append((fold_idx, resume_ckpt))

    if interrupted_folds:
        fold_idx, resume_checkpoint = interrupted_folds[0]
        return {
            "status": "resume_fold",
            "fold_idx": fold_idx,
            "fold_dir": os.path.join(exp_dir, f"{fold_prefix}{fold_idx}"),
            "resume_checkpoint": resume_checkpoint,
            "completed_folds": completed_folds,
            "interrupted_folds": [x[0] for x in interrupted_folds],
            "message": f"Resume interrupted training from {fold_prefix}{fold_idx}.",
        }

    for fold_idx in range(fold_start_index, fold_start_index + n_folds):
        if fold_idx in completed_folds:
            continue
        fold_dir = os.path.join(exp_dir, f"{fold_prefix}{fold_idx}")
        return {
            "status": "start_fold",
            "fold_idx": fold_idx,
            "fold_dir": fold_dir,
            "resume_checkpoint": None,
            "completed_folds": completed_folds,
            "interrupted_folds": [],
            "message": f"Start unfinished fold {fold_idx}.",
        }

    # Safety fallback: all folds exist but no summary file.
    return {
        "status": "already_completed",
        "fold_idx": None,
        "fold_dir": None,
        "resume_checkpoint": None,
        "completed_folds": completed_folds,
        "interrupted_folds": [],
        "message": f"All fold directories are already present under {exp_dir}. Consider writing {summary_filename}.",
    }