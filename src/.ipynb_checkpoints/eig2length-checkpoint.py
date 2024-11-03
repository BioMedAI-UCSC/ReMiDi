import torch


def eig2length(eigenvalues, diffusivity):
    """
    Convert Laplace eigenvalues into length scale.
    """
    # Ensure eigenvalues are not less than 0 by comparing with a tensor of zeros
    # of the same shape and taking the maximum element-wise.
    safe_eigenvalues = torch.max(
        eigenvalues, eigenvalues.new_full(eigenvalues.size(), 0)
    )

    length_scales = torch.pi * torch.sqrt(
        diffusivity / torch.max(safe_eigenvalues, eigenvalues)
    )

    return length_scales
