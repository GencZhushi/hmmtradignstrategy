"""Gaussian HMM regime detector with BIC model selection and filtered inference.

Key design rules (Spec A1):

- Training runs across ``n_components in [3..7]`` and selects the lowest-BIC model.
- **Inference is filtered (forward algorithm only)** — no ``predict`` / Viterbi.
- Regime *labels* are assigned by ascending mean return for readability, but the
  *strategy layer sorts by volatility*, independent of labels.
- All artifacts (model, metadata, label mapping) are persisted for reproducibility
  and rollback.
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - exercised indirectly
    from hmmlearn.hmm import GaussianHMM
except ImportError as exc:  # pragma: no cover - dependency missing in runtime env
    GaussianHMM = None  # type: ignore[assignment]
    _HMMLEARN_IMPORT_ERROR = exc
else:
    _HMMLEARN_IMPORT_ERROR = None

LOG = logging.getLogger(__name__)

REGIME_LABEL_MAP: dict[int, tuple[str, ...]] = {
    3: ("BEAR", "NEUTRAL", "BULL"),
    4: ("CRASH", "BEAR", "BULL", "EUPHORIA"),
    5: ("CRASH", "BEAR", "NEUTRAL", "BULL", "EUPHORIA"),
    6: ("CRASH", "STRONG_BEAR", "WEAK_BEAR", "WEAK_BULL", "STRONG_BULL", "EUPHORIA"),
    7: ("CRASH", "STRONG_BEAR", "WEAK_BEAR", "NEUTRAL", "WEAK_BULL", "STRONG_BULL", "EUPHORIA"),
}


@dataclass
class RegimeInfo:
    """Metadata for a single regime state (persisted with the model)."""

    regime_id: int
    regime_name: str
    expected_return: float
    expected_volatility: float
    vol_rank: float
    label_return_rank: int
    max_leverage_allowed: float = 1.0
    max_position_size_pct: float = 0.15
    min_confidence_to_act: float = 0.55


@dataclass
class RegimeState:
    """Live regime state returned to the orchestrator / API / agent."""

    regime_id: int
    regime_name: str
    probability: float
    state_probabilities: list[float]
    timestamp: pd.Timestamp
    is_confirmed: bool
    consecutive_bars: int
    flicker_rate: float


@dataclass
class ModelMetadata:
    """Persisted alongside the pickled model for audit/rollback."""

    model_version: str
    trained_at: str
    n_states: int
    n_samples: int
    feature_columns: list[str]
    bic_scores: dict[int, float]
    selected_bic: float
    log_likelihood: float
    dataset_hash: str
    random_state: int | None
    regimes: list[dict]
    notes: str = ""


@dataclass
class VolatilityRegimeHMM:
    """Gaussian HMM wrapper enforcing filtered inference."""

    n_candidates: Sequence[int] = (3, 4, 5, 6, 7)
    covariance_type: str = "full"
    n_init: int = 10
    random_state: int | None = 42
    max_iter: int = 200
    _fitted_model: object | None = field(default=None, init=False, repr=False)
    _feature_columns: list[str] = field(default_factory=list, init=False)
    _regimes: list[RegimeInfo] = field(default_factory=list, init=False)
    _metadata: ModelMetadata | None = field(default=None, init=False)

    # ------------------------------------------------------------------ training
    def fit(self, features: pd.DataFrame) -> "VolatilityRegimeHMM":
        if _HMMLEARN_IMPORT_ERROR is not None:  # pragma: no cover
            raise RuntimeError(f"hmmlearn is not available: {_HMMLEARN_IMPORT_ERROR}")
        if features.empty:
            raise ValueError("Feature frame is empty; cannot train HMM")
        data = features.to_numpy(dtype=float)
        if np.isnan(data).any():
            raise ValueError("Feature frame contains NaN; drop warmup rows first")

        best = self.select_model_bic(data)
        self._fitted_model = best["model"]
        self._feature_columns = list(features.columns)
        self._regimes = self._build_regime_infos(best["model"], data)
        self._metadata = ModelMetadata(
            model_version=f"hmm-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
            trained_at=datetime.now(timezone.utc).isoformat(),
            n_states=int(best["n_states"]),
            n_samples=int(len(data)),
            feature_columns=self._feature_columns,
            bic_scores={int(k): float(v) for k, v in best["bic_scores"].items()},
            selected_bic=float(best["bic"]),
            log_likelihood=float(best["log_likelihood"]),
            dataset_hash=_hash_dataset(features),
            random_state=self.random_state,
            regimes=[asdict(r) for r in self._regimes],
        )
        LOG.info(
            "HMM trained: n_states=%d BIC=%.3f LL=%.3f",
            best["n_states"],
            best["bic"],
            best["log_likelihood"],
        )
        return self

    def select_model_bic(self, data: np.ndarray) -> dict[str, object]:
        """Train each candidate with multiple random restarts and pick lowest BIC."""
        bic_scores: dict[int, float] = {}
        best_per_candidate: dict[int, tuple[float, float, object]] = {}
        for n_states in self.n_candidates:
            best_ll: float | None = None
            best_bic: float | None = None
            best_model: object | None = None
            for init_idx in range(self.n_init):
                seed = None if self.random_state is None else int(self.random_state) + int(n_states) * 101 + init_idx
                try:
                    model = GaussianHMM(
                        n_components=int(n_states),
                        covariance_type=self.covariance_type,
                        n_iter=self.max_iter,
                        random_state=seed,
                        tol=1e-4,
                    )
                    model.fit(data)
                except Exception as exc:  # pragma: no cover - numerical guard
                    LOG.debug("HMM init %d for n=%d failed: %s", init_idx, n_states, exc)
                    continue
                ll = float(model.score(data))
                n_params = _gaussian_hmm_n_params(int(n_states), int(data.shape[1]), self.covariance_type)
                bic = -2.0 * ll + n_params * float(np.log(len(data)))
                if best_bic is None or bic < best_bic:
                    best_bic = bic
                    best_ll = ll
                    best_model = model
            if best_bic is None or best_model is None:
                raise RuntimeError(f"HMM training failed for n_states={n_states}")
            bic_scores[int(n_states)] = float(best_bic)
            best_per_candidate[int(n_states)] = (float(best_bic), float(best_ll or 0.0), best_model)

        chosen_n = min(bic_scores, key=lambda k: bic_scores[k])
        chosen_bic, chosen_ll, chosen_model = best_per_candidate[chosen_n]
        LOG.info(
            "BIC candidates=%s selected=%d BIC=%.3f",
            {k: round(v, 3) for k, v in bic_scores.items()},
            chosen_n,
            chosen_bic,
        )
        return {
            "n_states": chosen_n,
            "bic": chosen_bic,
            "log_likelihood": chosen_ll,
            "bic_scores": bic_scores,
            "model": chosen_model,
        }

    # ---------------------------------------------------------------- inference
    def filtered_probabilities(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return P(state_t | obs_1..t) computed via the forward algorithm only."""
        self._ensure_fitted()
        if list(features.columns) != self._feature_columns:
            raise ValueError("Feature columns do not match those used for training")
        data = features.to_numpy(dtype=float)
        if np.isnan(data).any():
            raise ValueError("Feature frame contains NaN; drop warmup rows first")
        log_alpha = self._forward_log_alpha(data)
        proba = np.exp(log_alpha - log_alpha.max(axis=1, keepdims=True))
        proba /= proba.sum(axis=1, keepdims=True)
        columns = [f"state_{i}" for i in range(proba.shape[1])]
        return pd.DataFrame(proba, index=features.index, columns=columns)

    def predict_filtered_state(self, features: pd.DataFrame) -> np.ndarray:
        proba = self.filtered_probabilities(features)
        return proba.to_numpy().argmax(axis=1)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError("Use filtered_probabilities() instead")

    # ----------------------------------------------------------- persistence
    def save_model(self, directory: Path | str) -> Path:
        self._ensure_fitted()
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        version = self._metadata.model_version  # type: ignore[union-attr]
        artifact_dir = root / version
        artifact_dir.mkdir(parents=True, exist_ok=True)
        with (artifact_dir / "model.pkl").open("wb") as fh:
            pickle.dump(self._fitted_model, fh)
        with (artifact_dir / "metadata.json").open("w", encoding="utf-8") as fh:
            json.dump(asdict(self._metadata), fh, indent=2)  # type: ignore[arg-type]
        LOG.info("Saved HMM artifact to %s", artifact_dir)
        return artifact_dir

    def load_model(self, directory: Path | str) -> "VolatilityRegimeHMM":
        path = Path(directory)
        with (path / "model.pkl").open("rb") as fh:
            self._fitted_model = pickle.load(fh)
        with (path / "metadata.json").open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        self._metadata = ModelMetadata(**meta)
        self._feature_columns = list(self._metadata.feature_columns)
        self._regimes = [RegimeInfo(**r) for r in self._metadata.regimes]
        return self

    # ----------------------------------------------------------- accessors
    @property
    def metadata(self) -> ModelMetadata:
        self._ensure_fitted()
        return self._metadata  # type: ignore[return-value]

    @property
    def regimes(self) -> list[RegimeInfo]:
        self._ensure_fitted()
        return list(self._regimes)

    def assign_regime_labels(self) -> dict[int, str]:
        return {r.regime_id: r.regime_name for r in self.regimes}

    # -------------------------------------------------------- internals
    def _ensure_fitted(self) -> None:
        if self._fitted_model is None or self._metadata is None:
            raise RuntimeError("HMM has not been fitted yet")

    def _forward_log_alpha(self, data: np.ndarray) -> np.ndarray:
        model = self._fitted_model
        startprob = np.log(np.asarray(model.startprob_) + 1e-300)  # type: ignore[attr-defined]
        transmat = np.log(np.asarray(model.transmat_) + 1e-300)  # type: ignore[attr-defined]
        emission = model._compute_log_likelihood(data)  # type: ignore[attr-defined]
        n_samples, n_states = emission.shape
        log_alpha = np.empty((n_samples, n_states))
        log_alpha[0] = startprob + emission[0]
        for t in range(1, n_samples):
            log_alpha[t] = (
                _logsumexp_axis(log_alpha[t - 1][:, None] + transmat, axis=0)
                + emission[t]
            )
        return log_alpha

    def _build_regime_infos(self, model: object, training_data: np.ndarray) -> list[RegimeInfo]:
        means = np.asarray(model.means_)  # type: ignore[attr-defined]
        covars = np.asarray(model.covars_)  # type: ignore[attr-defined]
        returns_idx = self._feature_columns.index("ret_1") if "ret_1" in self._feature_columns else 0
        vol_idx = (
            self._feature_columns.index("realized_vol_20")
            if "realized_vol_20" in self._feature_columns
            else returns_idx
        )
        expected_returns = means[:, returns_idx]
        if covars.ndim == 3:  # full
            diag = np.array([np.diag(covars[i]) for i in range(covars.shape[0])])
        elif covars.ndim == 2:  # diag or spherical-like
            diag = covars
        else:  # pragma: no cover
            diag = np.repeat(covars[:, None], means.shape[1], axis=1)
        expected_vol = np.sqrt(np.abs(diag[:, vol_idx]))

        return_order = np.argsort(expected_returns)
        vol_order = np.argsort(expected_vol)
        n_states = means.shape[0]
        labels = REGIME_LABEL_MAP.get(n_states)
        if labels is None:
            labels = tuple(f"REGIME_{i}" for i in range(n_states))

        infos: list[RegimeInfo] = []
        for state_id in range(n_states):
            return_rank = int(np.where(return_order == state_id)[0][0])
            vol_rank_idx = int(np.where(vol_order == state_id)[0][0])
            vol_rank = vol_rank_idx / max(n_states - 1, 1)
            max_lev = 1.25 if vol_rank <= 0.34 else 1.0
            max_pos = 0.15
            infos.append(
                RegimeInfo(
                    regime_id=state_id,
                    regime_name=labels[return_rank],
                    expected_return=float(expected_returns[state_id]),
                    expected_volatility=float(expected_vol[state_id]),
                    vol_rank=float(vol_rank),
                    label_return_rank=return_rank,
                    max_leverage_allowed=max_lev,
                    max_position_size_pct=max_pos,
                )
            )
        _ = training_data  # training_data retained for future diagnostics
        return infos


def _gaussian_hmm_n_params(n_states: int, n_features: int, covariance_type: str) -> int:
    if covariance_type == "full":
        covar_params = n_states * n_features * (n_features + 1) // 2
    elif covariance_type == "diag":
        covar_params = n_states * n_features
    elif covariance_type == "tied":
        covar_params = n_features * (n_features + 1) // 2
    else:
        covar_params = n_states  # spherical or fallback
    mean_params = n_states * n_features
    startprob_params = n_states - 1
    transmat_params = n_states * (n_states - 1)
    return mean_params + covar_params + startprob_params + transmat_params


def _logsumexp_axis(arr: np.ndarray, axis: int) -> np.ndarray:
    arr_max = arr.max(axis=axis, keepdims=True)
    arr_max = np.where(np.isfinite(arr_max), arr_max, 0.0)
    out = np.log(np.sum(np.exp(arr - arr_max), axis=axis, keepdims=True)) + arr_max
    return np.squeeze(out, axis=axis)


def _hash_dataset(features: pd.DataFrame) -> str:
    payload = {
        "columns": list(features.columns),
        "index_hash": hashlib.sha1(
            pd.util.hash_pandas_object(features.index, index=False).values.tobytes()
        ).hexdigest(),
        "values_hash": hashlib.sha1(
            np.ascontiguousarray(features.to_numpy(dtype=float)).tobytes()
        ).hexdigest(),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest


# ----------------------------------------------------------- CLI hook
def train_cli(cfg) -> int:  # pragma: no cover - convenience only
    """Thin CLI helper used by ``main.py --train-only``."""
    from data.market_data import build_provider, MarketDataManager
    from data.feature_engineering import FeatureEngine
    from core.model_registry import ModelRegistry
    from monitoring.application import build_data_config

    symbols: Iterable[str] = cfg.get("broker.symbols", ["SPY"])  # type: ignore[assignment]
    min_train_bars = int(cfg.get("hmm.min_train_bars", 504))
    state_dir = Path(cfg.get("platform.state_dir", "state"))
    if not state_dir.is_absolute() and cfg.source_path is not None:
        state_dir = Path(cfg.source_path).parent.parent / state_dir
    provider = build_provider(build_data_config(cfg, state_dir))
    manager = MarketDataManager(provider=provider)
    engine = FeatureEngine(zscore_window=int(cfg.get("hmm.zscore_window", 252)))
    frames = []
    for sym in symbols:
        try:
            bars = manager.fetch_historical_daily_bars(sym, lookback_bars=min_train_bars + 260)
        except Exception as exc:
            LOG.warning("Skipping %s during training: %s", sym, exc)
            continue
        feats = engine.build_daily_features(bars)
        if len(feats) >= min_train_bars:
            frames.append(feats)
    if not frames:
        LOG.error("No features available to train HMM")
        return 1
    training = pd.concat(frames).sort_index().tail(min_train_bars)
    model = VolatilityRegimeHMM(
        n_candidates=tuple(cfg.get("hmm.n_candidates", (3, 4, 5, 6, 7))),
        n_init=int(cfg.get("hmm.n_init", 10)),
        covariance_type=str(cfg.get("hmm.covariance_type", "full")),
    )
    model.fit(training)
    registry = ModelRegistry(Path(cfg.get("governance.model_registry_path", "state/models")))
    artifact = model.save_model(registry.root)
    registry.register(artifact_dir=artifact, metadata=model.metadata)
    LOG.info("Trained and registered model %s", model.metadata.model_version)
    return 0
