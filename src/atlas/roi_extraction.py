import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd

from nilearn import datasets
from nilearn.image import resample_to_img

# ---------------------------------
# Harvard Oxford Atlas
# ---------------------------------
print("Loading Harvard Oxford Atlas...")

atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
atlas_img = nib.load(atlas.maps)
atlas_data = atlas_img.get_fdata()
labels = atlas.labels

print(f"Atlas Regions: {len(labels)-1}")

# ---------------------------------
# Output Folder
# ---------------------------------
OUTPUT_DIR = "processed/roi_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------
# MRI Files
# ---------------------------------
MRI_FOLDER = "processed/normalized_mri"
mri_files = glob.glob(os.path.join(MRI_FOLDER, "*.nii.gz"))

print(f"Found {len(mri_files)} MRI Files")

# ---------------------------------
# Process MRI Files
# ---------------------------------
for idx, mri_path in enumerate(mri_files):
    try:
        print(f"\n[{idx+1}/{len(mri_files)}]")
        print(os.path.basename(mri_path))

        # -------------------------
        # Load MRI
        # -------------------------
        mri_img = nib.load(mri_path)

        # -------------------------
        # Resample Atlas
        # -------------------------
        atlas_resampled = resample_to_img(
            atlas_img,
            mri_img,
            interpolation="nearest"
        )

        atlas_resampled_data = atlas_resampled.get_fdata()
        mri_data = mri_img.get_fdata()

        roi_rows = []

        # -------------------------
        # ROI Loop
        # -------------------------
        for region_id in range(1, len(labels)):
            region_name = labels[region_id]
            mask = (atlas_resampled_data == region_id)

            if np.sum(mask) == 0:
                roi_rows.append([region_name, 0, 0, 0, 0, 0, 0, 0])
                continue

            values = mri_data[mask]
            values = values[values > 0]  # Background (0) ని తీసివేయడం

            if len(values) == 0:
                roi_rows.append([region_name, 0, 0, 0, 0, 0, 0, 0])
                continue

            # Statistical Feature Calculation
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            median_val = np.median(values)
            volume_val = len(values)
            nonzero_ratio = len(values) / np.sum(mask)

            roi_rows.append([
                region_name,
                mean_val,
                std_val,
                min_val,
                max_val,
                median_val,
                volume_val,
                nonzero_ratio
            ])

        # -------------------------
        # Save to DataFrame
        # -------------------------
        roi_df = pd.DataFrame(
            roi_rows,
            columns=[
                "Region",
                "Mean",
                "Std",
                "Min",
                "Max",
                "Median",
                "Volume",
                "NonZeroRatio"
            ]
        )

        filename = os.path.basename(mri_path).replace(".nii.gz", ".csv")
        save_path = os.path.join(OUTPUT_DIR, filename)
        
        roi_df.to_csv(save_path, index=False)
        print(f"Saved -> {save_path}")

    except Exception as e:
        print("\nERROR occurred while processing:")
        print(mri_path)
        print(f"Exception details: {e}")

# ---------------------------------
# Finished
# ---------------------------------
print("\nROI Extraction Completed")
print(f"Output Folder: {OUTPUT_DIR}")