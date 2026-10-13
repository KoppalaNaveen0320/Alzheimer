import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

roi_files = glob.glob(
    "processed/roi_features/*.csv"
)

all_regions = []

for file in roi_files:

    df = pd.read_csv(file)

    all_regions.append(
        df[["Region", "Mean"]]
    )

combined = pd.concat(
    all_regions
)

importance = (
    combined
    .groupby("Region")["Mean"]
    .mean()
    .sort_values(
        ascending=False
    )
)

top_regions = importance.head(15)

print("\nTop Important Brain Regions\n")
print(top_regions)

os.makedirs(
    "outputs/heatmaps",
    exist_ok=True
)

plt.figure(
    figsize=(10,6)
)

top_regions.plot(
    kind="barh"
)

plt.title(
    "Top Brain Regions"
)

plt.xlabel(
    "Average ROI Intensity"
)

plt.tight_layout()

plt.savefig(
    "outputs/heatmaps/roi_importance.png"
)

print(
    "\nSaved:"
)

print(
    "outputs/heatmaps/roi_importance.png"
)