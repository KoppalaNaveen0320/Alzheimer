import pandas as pd
import numpy as np
import glob
import os

INPUT_DIR = "processed/roi_features"
OUTPUT_DIR = "processed/connectivity_matrices"

os.makedirs(OUTPUT_DIR, exist_ok=True)

roi_files = glob.glob(
    os.path.join(INPUT_DIR, "*.csv")
)

print(f"Found {len(roi_files)} ROI files")

for idx, roi_file in enumerate(roi_files):

    try:

        df = pd.read_csv(roi_file)

        features = df["Mean"].values

        matrix = np.outer(features, features)

        filename = (
            os.path.basename(roi_file)
            .replace(".csv", ".npy")
        )

        save_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        np.save(save_path, matrix)

        print(
            f"[{idx+1}/{len(roi_files)}] Saved"
        )

    except Exception as e:

        print("Error:", roi_file)
        print(e)

print("\nConnectivity Matrix Generation Completed")
