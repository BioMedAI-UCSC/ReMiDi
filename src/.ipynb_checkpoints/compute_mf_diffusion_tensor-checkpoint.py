import torch
from src.get_volume_mesh import get_volume_mesh
from src.mass_matrixP1_3D import mass_matrixP1_3D
from src.sparse_block_diagonal import sparse_block_diagonal


def compute_mf_diffusion_tensor(femesh, lap_eig, mf_jn, device):
    """
    Compute the effective diffusion tensor.
    """

    # Eigenvalues and moments
    eigfuncs = lap_eig["funcs"]

    # Sizes
    nsequence = mf_jn.size(0)
    ncompartment = len(femesh["points"])

    # Initialize output arguments
    diffusion_tensor = torch.zeros(3, 3, nsequence)

    volumes = torch.zeros(ncompartment)
    M_cmpts = []

    for icmpt in range(ncompartment):
        # Finite elements - assuming femesh['points'] and femesh['elements'] are lists of tensors
        points = femesh["points"][icmpt]
        elements = femesh["elements"][icmpt]

        # Get volume mesh - the function needs to be implemented or adapted for PyTorch
        volume_icmpt, fevolumes, _ = get_volume_mesh(points, elements)

        volumes[icmpt] = volume_icmpt

        # Compute mass matrix - this function also needs to be implemented or adapted for PyTorch
        M_cmpt = mass_matrixP1_3D(elements, fevolumes, device)
        M_cmpts.append(M_cmpt)

    M = sparse_block_diagonal(M_cmpts)

    points = torch.cat(femesh["points"], dim=1)

    a = points @ M @ eigfuncs

    # Initialize an empty list to store each diffusion tensor
    diffusion_tensor = torch.zeros(3, 3, nsequence)

    # Iterate over each sequence
    for iseq in range(nsequence):
        jn = mf_jn[iseq, :].unsqueeze(
            0
        )  # Add an extra dimension to make 'jn' 2D [1, ?] for matrix multiplication
        # Compute the diffusion tensor for this sequence
        # And
        # Add the computed diffusion tensor to the list
        diffusion_tensor[:, :, iseq] = (
            torch.matmul(torch.mul(a, jn), a.T) / volumes.sum()
        ).unsqueeze(
            0
        )  # Add an extra dimension for concatenation later

    return diffusion_tensor
