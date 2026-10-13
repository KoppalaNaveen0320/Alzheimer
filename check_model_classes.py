import torch

model_path = "outputs/models/best_gat_model.pth"

state_dict = torch.load(
    model_path,
    map_location="cpu"
)

print("Last Layer Keys:\n")

for key in state_dict.keys():
    if "classifier" in key:
        print(key, state_dict[key].shape)