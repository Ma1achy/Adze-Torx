"""Independent finite-distribution statistics for G2.1, without RNG or autodiff."""

from itertools import product
import math

import numpy as np


FIXTURES = ("uniform", "asymmetric", "peaked")
FAMILY_SIZE = 42


def tolerance(reference):
    return 2e-5 + 2e-5 * np.abs(reference)


def bernstein_radius(variance, maximum_deviation, n, *, family=FAMILY_SIZE, alpha=0.01):
    if n <= 0 or family <= 0 or not 0 < alpha < 1:
        raise ValueError("invalid sample/family/alpha")
    v, bound = np.asarray(variance), np.asarray(maximum_deviation)
    if np.any(v < 0) or np.any(bound < 0) or not np.all(np.isfinite(v + bound)):
        raise ValueError("invalid exact moments")
    t = math.log(2 * family / alpha)
    a = bound * t / (3 * n)
    return a + np.sqrt(a * a + 2 * v * t / n)


def wilson(successes, total):
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("invalid binomial support")
    z = 1.959963984540054
    p = successes / total
    d = 1 + z * z / total
    center = (p + z * z / (2 * total)) / d
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / d
    return [float(max(0, center - half)), float(min(1, center + half))]


def actions():
    return np.asarray(list(product(range(2), range(2), range(3), range(3), range(3))), np.int32)


def route(row):
    run, result = 0, []
    for edge in (*row[:2], 1):
        run += 1
        cut = int(edge == 1 or run == 2)
        result.append(cut)
        if cut:
            run = 0
    return np.asarray(result)


def softmax(value):
    x = np.asarray(value, np.float64)
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def finite_table(boundary_logits, extent_logits, theta):
    """Independently implement the exact frozen global-byte/query fixture.

    No production routes/loss/gradient helpers are called. Every emitted slot
    is aligned to its global byte position; absent bytes cost log(256).
    """
    raw = actions()
    pb, pl = softmax(boundary_logits), softmax(extent_logits)
    probability = np.prod(pb[np.arange(2)[None], raw[:, :2]], 1) * np.prod(
        pl[np.arange(3)[None], raw[:, 2:]], 1
    )
    weights = np.asarray([[0.4, -0.2], [0.7, 0.3], [-0.5, 0.9]], np.float64)
    own_bytes = np.asarray([[5, 6], [7, 8], [9, 10]])
    truth = [5, 6, 7, 9, 10]
    losses, direct = [], []
    for row in raw:
        b = route(row)
        effect = np.asarray([b[0], b[0] - b[1], b[1]])
        values = theta * weights + 0.6 * effect[:, None]
        emitted = [(i, j) for i in range(3) for j in range(int(row[2 + i]))]
        overlap = min(len(emitted), 5)
        costs, derivatives = [], []
        for j, (site, slot) in enumerate(emitted[:overlap]):
            v, w = values[site, slot], weights[site, slot]
            match = float(own_bytes[site, slot] == truth[j])
            costs.append(np.logaddexp(math.log(255), v) - match * v)
            derivatives.append((math.exp(v) / (255 + math.exp(v)) - match) * w)
        loss = (sum(costs) + math.log(256) * abs(len(emitted) - 5)) / 5
        deriv = sum(derivatives) / 5
        if overlap > 1:
            loss += costs[1]
            deriv += derivatives[1]
        else:
            loss += math.log(256)
        factor = 1 + row[2:].mean()
        losses.append(loss + 0.5 * theta**2 * factor)
        direct.append(deriv + theta * factor)
    losses = np.asarray(losses)
    scores = np.concatenate(
        (
            (np.eye(2)[raw[:, :2]] - pb).reshape(108, 4),
            (np.eye(3)[raw[:, 2:]] - pl).reshape(108, 9),
        ),
        axis=1,
    )
    gradients = np.concatenate((losses[:, None] * scores, np.asarray(direct)[:, None]), 1)
    mean = probability @ gradients
    centered = gradients - mean
    covariance = np.einsum("a,ai,aj->ij", probability, centered, centered)
    variance = np.diag(covariance).copy()
    maximum = np.abs(centered).max(0)
    assert abs(probability.sum() - 1) < 1e-12
    assert np.max(np.abs(probability @ scores)) < 1e-12
    return {
        "actions": raw,
        "probability": probability,
        "loss": losses,
        "direct": np.asarray(direct),
        "score": scores,
        "gradient": gradients,
        "mean": mean,
        "covariance": covariance,
        "variance": variance,
        "maximum": maximum,
    }


def pack_parameters(p):
    return np.concatenate(
        (
            np.asarray(p.boundary_logits).reshape(-1),
            np.asarray(p.extent_logits).reshape(-1),
            np.asarray(p.direct_theta).reshape(-1),
        )
    ).astype(np.float64)


def from_vector(value):
    return finite_table(value[:4].reshape(2, 2), value[4:13].reshape(3, 3), value[13])


def finite_difference(value, step=1e-4):
    result = []
    for i in range(14):
        delta = np.zeros(14)
        delta[i] = step
        a, b = from_vector(value + delta), from_vector(value - delta)
        result.append((a["probability"] @ a["loss"] - b["probability"] @ b["loss"]) / (2 * step))
    return np.asarray(result)


def component_labels():
    return (
        [f"boundary[{i},{j}]" for i in range(2) for j in range(2)]
        + [f"extent[{i},{j}]" for i in range(3) for j in range(3)]
        + ["direct_theta"]
    )
