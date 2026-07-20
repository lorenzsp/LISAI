"""Toy spectral sources: 1-D mixtures of narrow Gaussian peaks.

    x(f) = sum_k A_k * exp(-(f - mu_k)^2 / (2 sigma^2)) + n(f)

Each source has theta_k = (mu_k, A_k); the peak width sigma is fixed and
known. These are *toy spectral sources*, not LISA waveforms — the link to
LISA is only that both require inferring an unordered catalogue from a
superposition.
"""

import numpy as np

N_BINS = 128
SIGMA = 0.025           # fixed peak width, in units of the [0, 1] frequency axis
MU_RANGE = (0.10, 0.90)
A_RANGE = (0.5, 1.5)
NOISE_STD = 0.10

FREQ_AXIS = np.linspace(0.0, 1.0, N_BINS).astype(np.float32)


def render(theta):
    """Render a noiseless mixture from a catalogue theta of shape (K, 2)."""
    mu, amp = theta[:, 0:1], theta[:, 1:2]
    return (amp * np.exp(-((FREQ_AXIS[None] - mu) ** 2) / (2 * SIGMA**2))).sum(0)


def sample_catalogue(rng, K=2, sep_range=None):
    """Draw a catalogue of K sources. If sep_range=(lo, hi) is given, the
    pairwise |mu_i - mu_j| separations are constrained to that interval
    (used for the well-separated / overlapping / coincident regimes)."""
    for _ in range(500):
        mu = rng.uniform(*MU_RANGE, size=K)
        if K < 2 or sep_range is None:
            break
        seps = np.abs(mu[:, None] - mu[None, :])[np.triu_indices(K, 1)]
        if seps.min() >= sep_range[0] and seps.max() <= sep_range[1]:
            break
    amp = rng.uniform(*A_RANGE, size=K)
    return np.stack([mu, amp], axis=-1).astype(np.float32)


def make_mixture(rng, K=2, sep_range=None, noise_std=NOISE_STD):
    theta = sample_catalogue(rng, K=K, sep_range=sep_range)
    x = render(theta) + rng.normal(0.0, noise_std, N_BINS)
    return x.astype(np.float32), theta


def make_dataset(n, K=2, seed=0, sep_range=None, noise_std=NOISE_STD,
                 shuffle_targets=True):
    """Generate a dataset of n mixtures.

    shuffle_targets randomly permutes each target catalogue. The sources are
    drawn i.i.d., so their order carries no information either way — the
    explicit shuffle just makes that impossible to forget.
    """
    rng = np.random.default_rng(seed)
    X = np.empty((n, N_BINS), np.float32)
    T = np.empty((n, K, 2), np.float32)
    for i in range(n):
        x, theta = make_mixture(rng, K=K, sep_range=sep_range, noise_std=noise_std)
        if shuffle_targets:
            theta = theta[rng.permutation(K)]
        X[i], T[i] = x, theta
    return X, T
