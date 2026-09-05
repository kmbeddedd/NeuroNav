import numpy as np
from scipy.stats import shapiro


ERROR_COLS = [
    "x_error (m)",
    "y_error (m)",
    "z_error (m)",
    "satclockerror (m)",
]

ALPHA = 0.05


def evaluate_residuals(residuals):
    """
    Evaluate one residual vector using the competition-oriented
    statistics:

    - Shapiro-Wilk W
    - Shapiro-Wilk p-value
    - H0 decision at alpha=0.05
    - residual mean
    - residual standard deviation

    Returns a dictionary.
    """

    residuals = np.asarray(
        residuals,
        dtype=float
    )

    residuals = residuals[
        np.isfinite(residuals)
    ]

    n = len(residuals)

    if n < 3:
        return {
            "n": n,
            "W": np.nan,
            "p": np.nan,
            "decision": "INSUFFICIENT",
            "mean": np.mean(residuals)
            if n else np.nan,
            "std": np.std(residuals)
            if n else np.nan,
        }

    W, p = shapiro(residuals)

    return {
        "n": n,
        "W": float(W),
        "p": float(p),
        "decision": (
            "FAIL_TO_REJECT"
            if p >= ALPHA
            else "REJECT"
        ),
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
    }


def evaluate_all_targets(residual_matrix):
    """
    Evaluate the four error components.

    residual_matrix shape:
        (n_samples, 4)

    Column order:
        X, Y, Z, clock
    """

    residual_matrix = np.asarray(
        residual_matrix,
        dtype=float
    )

    if residual_matrix.ndim != 2:
        raise ValueError(
            "residual_matrix must be 2-dimensional"
        )

    if residual_matrix.shape[1] != 4:
        raise ValueError(
            "Expected four target columns: X, Y, Z, clock"
        )

    results = {}

    for i, col in enumerate(ERROR_COLS):
        results[col] = evaluate_residuals(
            residual_matrix[:, i]
        )

    # Equal-weight average of W across the four parameters.
    W_values = [
        results[col]["W"]
        for col in ERROR_COLS
        if np.isfinite(results[col]["W"])
    ]

    results["average_W"] = (
        float(np.mean(W_values))
        if W_values
        else np.nan
    )

    return results


def print_evaluation(results):

    print(
        "\nCompetition-style residual evaluation"
    )
    print("=" * 80)

    for col in ERROR_COLS:

        r = results[col]

        print(
            f"{col:<20} "
            f"n={r['n']:3d} | "
            f"W={r['W']:.6f} | "
            f"p={r['p']:.6f} | "
            f"{r['decision']:<15} | "
            f"mean={r['mean']: .6f} | "
            f"std={r['std']:.6f}"
        )

    print(
        f"\nAverage W = "
        f"{results['average_W']:.6f}"
    )


if __name__ == "__main__":

    # Small self-test so we know the evaluator works.
    rng = np.random.default_rng(42)

    residuals = rng.normal(
        size=(50, 4)
    )

    results = evaluate_all_targets(
        residuals
    )

    print_evaluation(results)
