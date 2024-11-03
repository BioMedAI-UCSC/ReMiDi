import torch


def length2eig(length_scales, diffusivity):
    """
    Convert length scales into Laplace eigenvalues
    """
    eigenvalues = (diffusivity * (torch.pi**2)) / (length_scales**2)

    return eigenvalues
