import torch
def directions(file_name):
    return torch.tensor([float(line.strip()) for line in open(file_name, 'r')][1:]).reshape(-1, 3).T