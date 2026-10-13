import pandas as pd
import glob
import os

labels_df = pd.read_csv(
    "processed/labels.csv"
)

dataset_rows = []

roi_files = glob.glob(
    "processed/roi_features/*.csv"
)

for file_path in roi_files:

    filename = os.path.basename(file_path)

    start = filename.find("OAS1_")
    end = filename.find("_mpr")

    if start == -1 or end == -1:
        continue

    subject_id = filename[start:end]

    match = labels_df[
        labels_df["ID"] == subject_id
    ]

    if len(match) == 0:
        continue

    label = int(match.iloc[0]["Label"])

    dataset_rows.append({
        "Subject_ID": subject_id,
        "ROI_File": file_path,
        "Label": label
    })

dataset_df = pd.DataFrame(dataset_rows)

dataset_df.to_csv(
    "processed/master_dataset.csv",
    index=False
)

print(dataset_df.head())

print("\nTotal Samples:")
print(len(dataset_df))