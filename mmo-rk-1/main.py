from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/matplotlib").resolve()))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_PATH = Path("data/CO2_Emissions_Canada.csv")
RESULTS_DIR = Path("results")
TARGET_COLUMN = "CO2 Emissions(g/km)"
CATEGORY_COLUMN = "Make"
ENCODED_COLUMN = "Make_mean_encoded"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    data = pd.read_csv(DATA_PATH)

    mean_encoding = (
        data.groupby(CATEGORY_COLUMN)[TARGET_COLUMN]
        .mean()
        .sort_values(ascending=False)
        .rename(ENCODED_COLUMN)
    )

    encoded_data = data.copy()
    encoded_data[ENCODED_COLUMN] = encoded_data[CATEGORY_COLUMN].map(mean_encoding)

    encoded_data.to_csv(RESULTS_DIR / "co2_emissions_with_mean_encoding.csv", index=False)
    mean_encoding.reset_index().to_csv(RESULTS_DIR / "mean_encoding_by_make.csv", index=False)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=encoded_data,
        x="Engine Size(L)",
        y=TARGET_COLUMN,
        hue="Fuel Type",
        alpha=0.65,
        s=35,
    )
    plt.title("CO2 emissions and engine size")
    plt.xlabel("Engine Size (L)")
    plt.ylabel("CO2 Emissions (g/km)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "scatter_engine_size_co2.png", dpi=200)
    plt.close()

    summary = [
        "Target / mean encoding",
        f"Dataset rows: {len(data)}",
        f"Encoded categorical column: {CATEGORY_COLUMN}",
        f"Target column: {TARGET_COLUMN}",
        f"New column: {ENCODED_COLUMN}",
        "",
        "First encoded values:",
        encoded_data[[CATEGORY_COLUMN, TARGET_COLUMN, ENCODED_COLUMN]].head(10).to_string(index=False),
        "",
        "Files:",
        "results/co2_emissions_with_mean_encoding.csv",
        "results/mean_encoding_by_make.csv",
        "results/scatter_engine_size_co2.png",
    ]
    (RESULTS_DIR / "summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print("\n".join(summary))


if __name__ == "__main__":
    main()
