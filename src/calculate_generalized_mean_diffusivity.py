import torch


def calculate_generalized_mean_diffusivity(diffusivity, volumes):
    # Ensure volumes is a PyTorch tensor and reshape it for broadcasting
    volumes_tensor = volumes.view(-1, 1, 1)

    # Multiply diffusivity tensor by volumes tensor with broadcasting
    weighted_diffusivity = diffusivity * volumes_tensor

    # Sum over the last dimension to get sum(diffusivity * volumes) for each matrix
    sum_weighted_diffusivity = torch.sum(weighted_diffusivity, dim=0)

    # Calculate the trace of the resulting matrix (sum of diagonal elements)
    trace_sum_weighted_diffusivity = torch.trace(sum_weighted_diffusivity)

    # Compute mean diffusivity
    mean_diffusivity = trace_sum_weighted_diffusivity / (3 * torch.sum(volumes_tensor))

    return mean_diffusivity
