import pandas as pd

df = pd.read_csv(
    "processed/roi_features/mwrc1OAS1_0001_MR1_mpr_anon_fslswapdim_bet.csv"
)

df = df.sort_values(
    by="Mean",
    ascending=False
)

print(
    "\nMost Active Regions\n"
)

print(
    df.head(20)
)