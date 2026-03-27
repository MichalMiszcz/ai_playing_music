import torch

x = torch.ones(4, 4, 3)
print(x)

start_token = torch.zeros(4, 1, 3).to(x.device)
print(start_token)

input_seq = torch.cat([x[:, :-1, :]], dim=2)
print(input_seq)