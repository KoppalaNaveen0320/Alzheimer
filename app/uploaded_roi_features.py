import os
import re

import numpy as np
import pandas as pd
from PIL import Image


REGION_COUNT = 48


def sanitize_filename(filename):
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", filename)
    return safe_name.strip("._") or "uploaded_mri"


def save_uploaded_file(uploaded_file, upload_dir="processed/uploads"):
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(
        upload_dir,
        sanitize_filename(uploaded_file.name)
    )

    uploaded_file.seek(0)

    with open(file_path, "wb") as output_file:
        output_file.write(uploaded_file.getbuffer())

    uploaded_file.seek(0)
    return file_path


def _normalize_array(values):
    values = values.astype(np.float32)
    finite_values = values[np.isfinite(values)]

    if finite_values.size == 0:
        return np.zeros_like(values, dtype=np.float32)

    low = np.percentile(finite_values, 1)
    high = np.percentile(finite_values, 99)

    if high <= low:
        low = float(np.min(finite_values))
        high = float(np.max(finite_values))

    if high <= low:
        return np.zeros_like(values, dtype=np.float32)

    values = np.clip(values, low, high)
    return (values - low) / (high - low)


def _crop_foreground(image_array):
    mask = image_array > max(0.03, float(np.percentile(image_array, 60)) * 0.35)

    if not np.any(mask):
        return image_array, np.ones_like(image_array, dtype=bool)

    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1

    return (
        image_array[y_min:y_max, x_min:x_max],
        mask[y_min:y_max, x_min:x_max]
    )


def _features_from_regions(region_values, region_prefix):
    rows = []

    for index, values in enumerate(region_values, start=1):
        values = values[np.isfinite(values)]
        values = values[values > 0]

        if values.size == 0:
            rows.append([
                f"{region_prefix} {index:02d}",
                0.0,
                0.0,
                0.0,
                0.0,
                0
            ])
            continue

        rows.append([
            f"{region_prefix} {index:02d}",
            float(np.mean(values)),
            float(np.std(values)),
            float(np.min(values)),
            float(np.max(values)),
            int(values.size)
        ])

    return pd.DataFrame(
        rows,
        columns=[
            "Region",
            "Mean",
            "Std",
            "Min",
            "Max",
            "Volume"
        ]
    )


def extract_image_roi_features(image):
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    grayscale = image.convert("L")
    image_array = np.asarray(grayscale, dtype=np.float32)
    image_array = _normalize_array(image_array)
    image_array, mask = _crop_foreground(image_array)

    height, width = image_array.shape
    row_edges = np.linspace(0, height, 7, dtype=int)
    col_edges = np.linspace(0, width, 9, dtype=int)

    region_values = []

    for row in range(6):
        for col in range(8):
            y0, y1 = row_edges[row], row_edges[row + 1]
            x0, x1 = col_edges[col], col_edges[col + 1]

            cell = image_array[y0:y1, x0:x1]
            cell_mask = mask[y0:y1, x0:x1]
            region_values.append(cell[cell_mask])

    return _features_from_regions(region_values, "Image ROI")


def _extract_grid_volume_features(volume, region_prefix="Volume ROI"):
    volume = _normalize_array(volume)
    mask = volume > max(0.03, float(np.percentile(volume, 60)) * 0.35)

    if not np.any(mask):
        mask = np.ones_like(volume, dtype=bool)

    depth, height, width = volume.shape
    z_edges = np.linspace(0, depth, 4, dtype=int)
    y_edges = np.linspace(0, height, 5, dtype=int)
    x_edges = np.linspace(0, width, 5, dtype=int)

    region_values = []

    for z in range(3):
        for y in range(4):
            for x in range(4):
                z0, z1 = z_edges[z], z_edges[z + 1]
                y0, y1 = y_edges[y], y_edges[y + 1]
                x0, x1 = x_edges[x], x_edges[x + 1]

                cell = volume[z0:z1, y0:y1, x0:x1]
                cell_mask = mask[z0:z1, y0:y1, x0:x1]
                region_values.append(cell[cell_mask])

    return _features_from_regions(region_values, region_prefix)


def extract_nifti_roi_features(mri_path):
    import nibabel as nib

    mri_img = nib.load(mri_path)

    try:
        from nilearn import datasets
        from nilearn.image import resample_to_img

        atlas = datasets.fetch_atlas_harvard_oxford(
            "cort-maxprob-thr25-2mm"
        )
        atlas_img = nib.load(atlas.maps)
        atlas_resampled = resample_to_img(
            atlas_img,
            mri_img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True
        )

        atlas_data = atlas_resampled.get_fdata()
        mri_data = _normalize_array(mri_img.get_fdata())
        rows = []

        for region_id, region_name in enumerate(atlas.labels[1:], start=1):
            region_mask = atlas_data == region_id
            values = mri_data[region_mask]
            values = values[values > 0]

            if values.size == 0:
                rows.append([region_name, 0.0, 0.0, 0.0, 0.0, 0])
                continue

            rows.append([
                region_name,
                float(np.mean(values)),
                float(np.std(values)),
                float(np.min(values)),
                float(np.max(values)),
                int(values.size)
            ])

        return pd.DataFrame(
            rows,
            columns=[
                "Region",
                "Mean",
                "Std",
                "Min",
                "Max",
                "Volume"
            ]
        )

    except Exception:
        return _extract_grid_volume_features(
            mri_img.get_fdata(),
            region_prefix="Volume ROI"
        )


def save_roi_features(roi_df, uploaded_path, output_dir="processed/uploads"):
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(uploaded_path)
    base_name = re.sub(r"\.nii\.gz$", "", base_name, flags=re.IGNORECASE)
    base_name = os.path.splitext(base_name)[0]
    roi_path = os.path.join(output_dir, f"{base_name}_roi_features.csv")

    roi_df.to_csv(roi_path, index=False)
    return roi_path
