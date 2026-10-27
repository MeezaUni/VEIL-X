"""L2-regularised logistic regression by Newton's method (IRLS), numpy only.

Minimises ``mean(log-loss) + (1 / (2 C n)) * ||w||^2`` (intercept unpenalised), the same
objective as scikit-learn's ``LogisticRegression(C=C)`` divided by n. Features are standardised
with statistics from the training data only; the fitted object stores them so that validation,
calibration and test data are transformed identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]


def _sigmoid(z: F64) -> F64:
    out: F64 = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def log_loss(p: F64, y: NDArray[np.int64], eps: float = 1e-12) -> F64:
    """Per-sample binary log-loss (natural log)."""
    p = np.clip(p, eps, 1 - eps)
    return np.asarray(-(y * np.log(p) + (1 - y) * np.log(1 - p)), dtype=np.float64)


@dataclass
class LogisticModel:
    feature_names: tuple[str, ...]
    C: float
    mean: F64 = field(default_factory=lambda: np.zeros(0))
    scale: F64 = field(default_factory=lambda: np.ones(0))
    coef: F64 = field(default_factory=lambda: np.zeros(0))
    intercept: float = 0.0
    n_iter: int = 0
    converged: bool = False

    def _z(self, x: F64) -> F64:
        return np.asarray((x - self.mean) / self.scale, dtype=np.float64)

    def fit(
        self, x: F64, y: NDArray[np.int64], *, max_iter: int = 100, tol: float = 1e-9
    ) -> LogisticModel:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names) or x.shape[0] != y.size:
            raise ValueError("x must be (n, n_features) matching feature_names and y")
        if not np.isfinite(x).all():
            raise FloatingPointError("non-finite feature values")
        self.mean = x.mean(0)
        self.scale = np.where(x.std(0) > 1e-12, x.std(0), 1.0)
        z = np.hstack([np.ones((x.shape[0], 1)), self._z(x)])
        n, d = z.shape
        lam = np.full(d, 1.0 / (self.C * n))
        lam[0] = 0.0
        w = np.zeros(d)
        for it in range(1, max_iter + 1):
            p = _sigmoid(z @ w)
            grad = z.T @ (p - y) / n + lam * w
            h = (z * (p * (1 - p))[:, None]).T @ z / n + np.diag(lam) + 1e-12 * np.eye(d)
            step = np.linalg.solve(h, grad)
            w -= step
            self.n_iter = it
            if float(np.max(np.abs(step))) < tol:
                self.converged = True
                break
        self.intercept, self.coef = float(w[0]), w[1:]
        return self

    def decision(self, x: F64) -> F64:
        return np.asarray(self._z(np.asarray(x, dtype=np.float64)) @ self.coef + self.intercept)

    def predict_proba(self, x: F64) -> F64:
        return _sigmoid(self.decision(x))

    def to_dict(self) -> dict[str, object]:
        return {
            "features": list(self.feature_names),
            "C": self.C,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
            "n_iter": self.n_iter,
            "converged": self.converged,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> LogisticModel:
        m = cls(tuple(d["features"]), float(d["C"]))  # type: ignore[arg-type]
        m.mean = np.asarray(d["mean"], dtype=np.float64)
        m.scale = np.asarray(d["scale"], dtype=np.float64)
        m.coef = np.asarray(d["coef"], dtype=np.float64)
        m.intercept = float(d["intercept"])  # type: ignore[arg-type]
        m.n_iter = int(d["n_iter"])  # type: ignore[call-overload]
        m.converged = bool(d["converged"])
        return m


def select_c(
    x_tr: F64,
    y_tr: NDArray[np.int64],
    x_va: F64,
    y_va: NDArray[np.int64],
    names: tuple[str, ...],
    grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
) -> tuple[LogisticModel, dict[float, float]]:
    """Fit on train for every C; choose the C with the lowest validation mean log-loss."""
    scores: dict[float, float] = {}
    best: LogisticModel | None = None
    for c in grid:
        m = LogisticModel(names, c).fit(x_tr, y_tr)
        scores[c] = float(log_loss(m.predict_proba(x_va), y_va).mean())
        if best is None or scores[c] < scores[best.C]:
            best = m
    assert best is not None
    return best, scores
