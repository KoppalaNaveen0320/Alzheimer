import pandas as pd

train_df = pd.read_csv(
    "processed/splits/train.csv"
)

val_df = pd.read_csv(
    "processed/splits/val.csv"
)

test_df = pd.read_csv(
    "processed/splits/test.csv"
)

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nTrain Distribution:")
print(train_df["Label"].value_counts())

print("\nValidation Distribution:")
print(val_df["Label"].value_counts())

print("\nTest Distribution:")
print(test_df["Label"].value_counts())