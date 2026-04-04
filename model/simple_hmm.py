import math

import numpy as np

from .logging_utils import get_logger

logger = get_logger(__name__)


def _stable_inverse(matrix: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix)


# ⚡ Bolt: Vectorized to compute over all samples (N) at once using einsum.
# This prevents redundant covariance inversions and loop overhead per sample.
def _gaussian_log_prob(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    dim = mean.shape[0]
    diff = X - mean
    inv = _stable_inverse(cov)
    log_det = np.log(np.linalg.det(cov) + 1e-9)
    quad = np.einsum("ni,ij,nj->n", diff, inv, diff)
    return -0.5 * (quad + log_det + dim * math.log(2 * math.pi))


class GaussianHMM:
    """
    Minimal fallback implementation mirroring the hmmlearn GaussianHMM API.
    This is **not** a full HMM; it approximates behavior via clustering so that
    the rest of the application can function offline when hmmlearn is unavailable.
    """

    def __init__(
        self,
        n_components: int = 4,
        covariance_type: str = "diag",
        n_iter: int = 100,
        random_state: int | None = None,
    ):
        if covariance_type not in {"diag", "full"}:
            raise ValueError("Only 'diag' and 'full' covariances are supported.")
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = max(10, n_iter)
        self.random_state = np.random.default_rng(random_state)

    def _init_means(self, X: np.ndarray) -> np.ndarray:
        if len(X) < self.n_components:
            raise ValueError("Number of observations must exceed number of components.")
        choices = self.random_state.choice(len(X), size=self.n_components, replace=False)
        return X[choices].copy()

    def _assign_clusters(self, X: np.ndarray, means: np.ndarray) -> np.ndarray:
        # ⚡ Bolt: Replaced memory-intensive explicit broadcasting
        # with expanded squared distance formula.
        # This uses matrix multiplication to calculate distances without
        # creating intermediate (N, K, D) arrays.
        distances_sq = -2 * np.dot(X, means.T) + np.sum(means**2, axis=1)
        return np.argmin(distances_sq, axis=1)

    def _estimate_transitions(self, labels: np.ndarray) -> np.ndarray:
        trans = np.ones((self.n_components, self.n_components))  # add-one smoothing
        for prev, nxt in zip(labels[:-1], labels[1:], strict=False):
            trans[prev, nxt] += 1
        trans /= trans.sum(axis=1, keepdims=True)
        return trans

    def _cluster_cov(self, cluster: np.ndarray) -> np.ndarray:
        if cluster.shape[0] <= 1:
            return np.eye(cluster.shape[1])
        cov = np.cov(cluster, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[cov]])
        if self.covariance_type == "diag":
            cov = np.diag(np.diag(cov))
        return cov + np.eye(cov.shape[0]) * 1e-6

    def fit(self, X: np.ndarray) -> "GaussianHMM":
        X = np.asarray(X, dtype=float)
        logger.debug("Fallback GaussianHMM fitting on %s samples", len(X))
        means = self._init_means(X)
        for _ in range(self.n_iter // 5):
            labels = self._assign_clusters(X, means)
            for idx in range(self.n_components):
                mask = labels == idx
                if mask.any():
                    means[idx] = X[mask].mean(axis=0)
        labels = self._assign_clusters(X, means)
        covars = np.stack([self._cluster_cov(X[labels == idx]) for idx in range(self.n_components)])

        self.means_ = means
        self.covars_ = covars
        self.transmat_ = self._estimate_transitions(labels)
        start_counts = np.bincount(labels, minlength=self.n_components)
        self.startprob_ = start_counts / start_counts.sum()
        self._fitted = True
        logger.info("Fallback GaussianHMM fitted with %s states", self.n_components)
        return self

    def _check_fitted(self):
        if not getattr(self, "_fitted", False):
            raise RuntimeError("GaussianHMM has not been fitted yet.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X)
        states = self._assign_clusters(X, self.means_)
        logger.debug("Fallback GaussianHMM predicted %s states", len(states))
        return states

    def _compute_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        log_probs = np.zeros((len(X), self.n_components))
        for idx in range(self.n_components):
            cov = self.covars_[idx]
            mean = self.means_[idx]
            # ⚡ Bolt: Pass the whole array X at once
            log_probs[:, idx] = _gaussian_log_prob(X, mean, cov)
        return log_probs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_probs = self._compute_log_likelihood(X)
        probs = np.exp(log_probs - log_probs.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def score(self, X: np.ndarray) -> float:
        log_probs = self._compute_log_likelihood(X)
        per_sample = np.logaddexp.reduce(log_probs, axis=1)
        return float(per_sample.sum())

    def score_samples(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        log_prob_matrix = self._compute_log_likelihood(X)
        posteriors = self.predict_proba(X)
        total_log_prob = float(np.logaddexp.reduce(log_prob_matrix, axis=1).sum())
        return total_log_prob, posteriors
