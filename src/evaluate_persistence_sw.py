import pandas as pd
from scipy.stats import shapiro
from pathlib import Path

RESIDUAL_FILE = Path(
    "results/baseline/persistence_residuals.csv"
)


def main():

    df = pd.read_csv(RESIDUAL_FILE)

    print("\nPERSISTENCE BASELINE — SHAPIRO-WILK")
    print("=" * 90)

    results = []

    for (dataset, validation_day, variable), group in df.groupby(
        ["dataset", "validation_day", "variable"]
    ):

        residuals = group["residual"].dropna().to_numpy()
        n = len(residuals)

        if n < 3:
            W = float("nan")
            p = float("nan")
            decision = "INSUFFICIENT"
        else:
            W, p = shapiro(residuals)
            decision = (
                "FAIL_TO_REJECT"
                if p >= 0.05
                else "REJECT"
            )

        mean = residuals.mean()
        std = residuals.std()

        print(
            f"{dataset:5s} | "
            f"{validation_day} | "
            f"{variable:<20s} | "
            f"n={n:3d} | "
            f"W={W:.6f} | "
            f"p={p:.6f} | "
            f"{decision:<15s} | "
            f"mean={mean: .6f} | "
            f"std={std:.6f}"
        )

        results.append({
            "dataset": dataset,
            "validation_day": validation_day,
            "variable": variable,
            "n": n,
            "shapiro_W": W,
            "shapiro_p": p,
            "decision": decision,
            "residual_mean": mean,
            "residual_std": std,
        })

    result_df = pd.DataFrame(results)

    output_dir = Path("results/baseline")
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(
        output_dir / "persistence_shapiro_results.csv",
        index=False
    )

    print("\n")
    print("=" * 90)
    print("AVERAGE W BY DATASET AND VARIABLE")
    print("=" * 90)

    summary = (
        result_df
        .groupby(["dataset", "variable"])["shapiro_W"]
        .mean()
        .unstack()
    )

    print(summary.round(6).to_string())

    print("\nSaved:")
    print(
        "results/baseline/persistence_shapiro_results.csv"
    )


if __name__ == "__main__":
    main()
