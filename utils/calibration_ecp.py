import numpy as np
import torch


def binary_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def binary_metrics_from_prob(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    conf = binary_confusion(y_true, y_pred)
    tn, fp, fn, tp = conf["tn"], conf["fp"], conf["fn"], conf["tp"]

    acc = (tp + tn) / max(len(y_true), 1)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    bal_acc = 0.5 * (sens + spec)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-12)

    out = {
        **conf,
        "accuracy": acc,
        "balanced_acc": bal_acc,
        "precision": prec,
        "recall": sens,
        "specificity": spec,
        "f1": f1,
    }

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        out["auroc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
        out["ap"] = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    except Exception:
        out["auroc"] = float("nan")
        out["ap"] = float("nan")
    return out


class TemperatureScaler(torch.nn.Module):
    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.log_temp = torch.nn.Parameter(torch.log(torch.tensor(float(init_temp))))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temp).clamp_min(1e-4)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


@torch.no_grad()
def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return np.asarray(logits, dtype=float) / max(float(temperature), 1e-6)


def fit_temperature_scaler(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200, lr: float = 1e-2) -> float:
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)

    scaler = TemperatureScaler(init_temp=1.0)
    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        scaled = scaler(logits_t)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scaled, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(scaler.temperature.detach().cpu().item())


def sweep_threshold(y_true: np.ndarray, y_prob: np.ndarray, num_thresholds: int = 181, metric: str = "balanced_acc") -> tuple[float, dict]:
    best_thr = 0.5
    best_metrics = None
    best_score = -1e18

    for thr in np.linspace(0.05, 0.95, num_thresholds):
        m = binary_metrics_from_prob(y_true, y_prob, threshold=float(thr))
        score = m[metric]
        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_metrics = m

    return best_thr, best_metrics
