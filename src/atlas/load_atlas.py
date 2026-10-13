from nilearn import datasets

atlas = datasets.fetch_atlas_harvard_oxford(
    "cort-maxprob-thr25-2mm"
)

print("Atlas Path:")
print(atlas.maps)

print("\nTotal Regions:")
print(len(atlas.labels))

print("\nFirst 20 Regions:")

for region in atlas.labels[:20]:
    print(region)