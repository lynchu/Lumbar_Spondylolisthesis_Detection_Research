import math


class EarlyStopping:
    """
    Stop training when monitored metric does not improve for `patience` epochs.

    mode:
        - "max": larger is better (e.g. AUROC)
        - "min": smaller is better (e.g. loss)
    """
    def __init__(self, patience=10, mode="max", min_delta=0.0):
        assert mode in ["max", "min"]
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best_score = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def _is_improvement(self, current, best):
        if self.mode == "max":
            return current > best + self.min_delta
        return current < best - self.min_delta

    def step(self, current_score):
        if current_score is None or (isinstance(current_score, float) and math.isnan(current_score)):
            self.num_bad_epochs += 1
        elif self.best_score is None:
            self.best_score = current_score
            self.num_bad_epochs = 0
        elif self._is_improvement(current_score, self.best_score):
            self.best_score = current_score
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self.should_stop = True

        return self.should_stop

    def state_dict(self):
        return {
            "patience": self.patience,
            "mode": self.mode,
            "min_delta": self.min_delta,
            "best_score": self.best_score,
            "num_bad_epochs": self.num_bad_epochs,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state):
        self.patience = state.get("patience", self.patience)
        self.mode = state.get("mode", self.mode)
        self.min_delta = state.get("min_delta", self.min_delta)
        self.best_score = state.get("best_score", self.best_score)
        self.num_bad_epochs = state.get("num_bad_epochs", self.num_bad_epochs)
        self.should_stop = state.get("should_stop", self.should_stop)