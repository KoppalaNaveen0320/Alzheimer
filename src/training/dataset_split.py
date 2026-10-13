import pandas as pd
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv("processed/master_dataset.csv")

print("\nClass Distribution:")
print(df["Label"].value_counts())

# First split (stratified)
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["Label"]
)

# Second split. Stratify when every temporary class has enough samples.
temp_counts = temp_df["Label"].value_counts()
second_stratify = temp_df["Label"] if temp_counts.min() >= 2 else None

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=second_stratify
)

os.makedirs("processed/splits", exist_ok=True)

train_df.to_csv("processed/splits/train.csv", index=False)
val_df.to_csv("processed/splits/val.csv", index=False)
test_df.to_csv("processed/splits/test.csv", index=False)

print("\nDataset Split Completed")
print("Train      :", len(train_df))
print("Validation :", len(val_df))
print("Test       :", len(test_df))

print("\nTrain Distribution:")
print(train_df["Label"].value_counts())

print("\nValidation Distribution:")
print(val_df["Label"].value_counts())

print("\nTest Distribution:")
print(test_df["Label"].value_counts())
