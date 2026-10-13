import torch

graphs = torch.load(
    "processed/graphs/graph_dataset.pt",
    weights_only=False
)

print("=" * 50)
print("Graphs Loaded:", len(graphs))
print("=" * 50)

g = graphs[0]

print("\nGraph Object:")
print(g)

print("\nNode Shape:")
print(g.x.shape)

print("\nEdge Shape:")
print(g.edge_index.shape)

print("\nLabel:")
print(g.y)

print("\nFirst 5 Node Features:")
print(g.x[:5])

print("\nFeature Dimension:")
print(g.x.shape[1])

print("\nNumber of Nodes:")
print(g.x.shape[0])

print("\nNumber of Edges:")
print(g.edge_index.shape[1])

print("\nGraph Dataset Verification Complete")