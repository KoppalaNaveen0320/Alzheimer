import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -------------------------
# Output Folder
# -------------------------
OUTPUT_DIR = "outputs/heatmaps"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# ROI Files
# -------------------------
roi_files = glob.glob("processed/roi_features/*.csv")

if len(roi_files) == 0:
    print("No ROI Feature Files Found")
    exit()

print(f"Found {len(roi_files)} ROI Files")

# -------------------------
# Features To Analyze
# -------------------------
FEATURE_COLUMNS = [
    "Mean",
    "Std",
    "Min",
    "Max",
    "Median",
    "Volume",
    "NonZeroRatio"
]

# -------------------------
# Load ROI Data
# -------------------------
all_features = []
for file in roi_files:
    try:
        df = pd.read_csv(file)
        # కేవలం అందుబాటులో ఉన్న కాలమ్స్ మాత్రమే ఫిల్టర్ అవుతాయి
        available_cols = [col for col in FEATURE_COLUMNS if col in df.columns]
        
        if available_cols:
            all_features.append(df[available_cols])
            
    except Exception as e:
        print(f"Error Loading: {file}")
        print(e)

# -------------------------
# Combine Data
# -------------------------
if len(all_features) == 0:
    print("No valid ROI data loaded. Exiting...")
    exit()

combined_df = pd.concat(all_features, ignore_index=True)
print("\nCombined Shape:")
print(combined_df.shape)

# -------------------------
# Correlation Matrix
# -------------------------
corr_matrix = combined_df.corr(numeric_only=True)
print("\nCorrelation Matrix:")
print(corr_matrix)

# -------------------------
# Heatmap
# -------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    square=True
)
plt.title("ROI Feature Correlation Analysis")
plt.tight_layout()

# ఇక్కడ మీ స్ట్రీమ్‌లిట్ యాప్ వెతికే 'attention_map.png' లొకేషన్‌కు సేవ్ అవుతుంది
save_path = os.path.join(OUTPUT_DIR, "attention_map.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

# -------------------------
# Feature Importance
# -------------------------
importance = corr_matrix.abs().sum().sort_values(ascending=False)
print("\nFeature Importance Ranking:")
print(importance)

importance.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"))

# -------------------------
# Bar Chart
# -------------------------
plt.figure(figsize=(8, 5))
importance.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("ROI Feature Importance")
plt.ylabel("Correlation Score")
plt.xticks(rotation=45)
plt.tight_layout()

chart_save_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
plt.savefig(chart_save_path, dpi=300)
plt.close()

# -------------------------
# Finished
# -------------------------
print("\nHeatmap Saved:")
print(save_path)

print("\nFeature Importance Saved:")
print(chart_save_path)