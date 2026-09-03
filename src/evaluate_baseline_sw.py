import pandas as pd
from scipy.stats import shapiro
from pathlib import Path

RESULT_FILE = Path(
    "results/baseline/persistence_results.csv"
)


def main():

    df = pd.read_csv(RESULT_FILE)

    print("\nPERSISTENCE BASELINE — SHAPIRO-WILK EVALUATION")
    print("=" * 80)

    results = []

    for (dataset, validation_day, variable), group in df.groupby(
        ["dataset", "validation_day", "variable"]
    ):

        n = int(group["n"].iloc[0])

        # The residual statistics currently stored in the baseline
        # are enough for mean/std, but not enough for Shapiro-Wilk.
        # Therefore this script identifies the required evaluation
        # structure and reports what is currently available.
        print(
            f"{dataset:5s} | "
            f"{validation_day} | "
            f"{variable:<20s} | "
            f"n={n:3d} | "
            f"mean={group['residual_mean'].iloc[0]: .6f} | "
            f"std={group['residual_std'].iloc[0]: .6f}"
        )

    print("\nNOTE")
    print("-" * 80)
    print(
        "The current persistence_results.csv stores only summary "
        "statistics, not individual residuals."
    )
    print(
        "Shapiro-Wilk requires the full residual vector."
    )
    print(
        "Therefore the baseline script must next be modified to "
        "save residuals for exact competition-metric evaluation."
    )


if __name__ == "__main__":
    main()
