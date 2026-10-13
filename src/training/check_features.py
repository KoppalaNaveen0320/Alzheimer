import pandas as pd

df = pd.read_csv(
    "processed/roi_features/mwrc1OAS1_0001_MR1_mpr_anon_fslswapdim_bet.csv"
)

print(df.head())

print("\nMean:")
print(df["Mean"].mean())

print("\nStd:")
print(df["Mean"].std())
