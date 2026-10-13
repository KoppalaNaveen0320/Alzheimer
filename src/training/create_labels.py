import pandas as pd
import os

# --------------------------------
# Load OASIS Metadata
# --------------------------------

csv_path = (
    "data/raw/oasis1/"
    "oasis_cross-sectional.csv"
)

df = pd.read_csv(
    csv_path
)

# --------------------------------
# Remove Missing CDR
# --------------------------------

df = df[
    df["CDR"].notna()
]

# --------------------------------
# Multi-Class Label Mapping
# --------------------------------

def convert_label(cdr):

    if cdr == 0:
        return 0      # Normal

    elif cdr == 0.5:
        return 1      # Very Mild

    elif cdr == 1:
        return 2      # Mild

    elif cdr == 2:
        return 3      # Moderate

    else:
        return 4      # Severe

# --------------------------------
# Create Labels
# --------------------------------

df["Label"] = df[
    "CDR"
].apply(
    convert_label
)

# --------------------------------
# Save Labels
# --------------------------------

labels = df[
    [
        "ID",
        "CDR",
        "Label"
    ]
]

os.makedirs(
    "processed",
    exist_ok=True
)

labels.to_csv(
    "processed/labels.csv",
    index=False
)

# --------------------------------
# Display Statistics
# --------------------------------

print(
    "\nFirst 10 Labels\n"
)

print(
    labels.head(10)
)

print(
    "\nCDR Distribution\n"
)

print(
    df["CDR"]
    .value_counts()
    .sort_index()
)

print(
    "\nLabel Distribution\n"
)

print(
    labels["Label"]
    .value_counts()
    .sort_index()
)

print(
    "\nTotal Samples:"
)

print(
    len(labels)
)

print(
    "\nSaved:"
)

print(
    "processed/labels.csv"
)