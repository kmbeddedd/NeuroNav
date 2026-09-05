import pandas as pd
import numpy as np
from pathlib import Path

from evaluation import (
    evaluate_all_targets,
    print_evaluation,
    ERROR_COLS,
)


RESIDUAL_FILE = Path(
    "results/baseline/persistence_residuals.csv"
)


def main():

    df = pd.read_csv(RESIDUAL_FILE)

    print("\nPERSISTENCE BASELINE — CENTRAL EVALUATION")
    print("=" * 90)

    fold_results = []

    for (dataset, validation_day), group in df.groupby(
        ["dataset", "validation_day"]
    ):

        pivot = (
            group
            .pivot(
                index="sample_index",
                columns="variable",
                values="residual"
            )
            .reindex(columns=ERROR_COLS)
        )

        residual_matrix = pivot.to_numpy(
            dtype=float
        )

        results = evaluate_all_targets(
            residual_matrix
        )

        print(
            f"\n{dataset} | "
            f"Validation day: {validation_day}"
        )

        print_evaluation(results)

        fold_results.append({
            "dataset": dataset,
            "validation_day": validation_day,
            "average_W": results["average_W"],
        })

    fold_df = pd.DataFrame(fold_results)

    print("\n")
    print("=" * 90)
    print("BASELINE FOLD SUMMARY")
    print("=" * 90)

    print(
        fold_df.to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 90)
    print("OVERALL BASELINE")
    print("=" * 90)

    print(
        f"Mean fold Average W: "
        f"{fold_df['average_W'].mean():.6f}"
    )

    print(
        f"Median fold Average W: "
        f"{fold_df['average_W'].median():.6f}"
    )

    output_dir = Path("results/baseline")
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    fold_df.to_csv(
        output_dir /
        "persistence_fold_scores.csv",
        index=False
    )

    print("\nSaved:")
    print(
        "results/baseline/"
        "persistence_fold_scores.csv"
    )


if __name__ == "__main__":
    main()
