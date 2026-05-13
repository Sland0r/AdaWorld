import torch
d = torch.load('latent_actions_dump_2/adaworld/0194e1a2-aa74-7bf1-9d3a-caefac405d75/latent_actions.pt', map_location='cpu')
print(list(d.keys()))
