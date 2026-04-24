import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

zeros = torch.zeros(5, 2).to(device)
ones = torch.ones(3, 2).to(device)

print(zeros)
print(ones)

concatenated = torch.cat([zeros, ones], dim=0).to(device)
print(concatenated)