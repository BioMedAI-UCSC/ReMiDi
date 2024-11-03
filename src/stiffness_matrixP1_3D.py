import torch
from torch import lgamma
from src.phider import phider
from src.torch_functions import torch_amtam, torch_astam, torch_smamt


def stiffness_matrixP1_3D(elements, coordinates, coeffs=None):
    # Adjusting shapes for PyTorch
    elements = elements.t().contiguous()
    coordinates = coordinates.t().contiguous()
    NE = elements.shape[0]  # Number of elements
    DIM = coordinates.shape[1]  # Problem dimension
    NLB = 4  # Number of local basic functions

    # No need to transpose in PyTorch as we adjust indexing directly
    elements_adj = elements

    coord = torch.zeros(
        (DIM, NLB, NE), dtype=coordinates.dtype, device=coordinates.device
    )
    for d in range(DIM):
        for i in range(NLB):
            coord[d, i, :] = coordinates[elements_adj[:, i], d]

    IP = torch.tensor(
        [1 / 4, 1 / 4, 1 / 4], dtype=torch.float32, device=coordinates.device
    ).reshape(-1, 1)
    dphi, detj, _ = phider(coord, IP, "P1")

    volumes = torch.abs(detj).squeeze() / lgamma(torch.tensor([DIM + 1])).exp()
    volumes = volumes.reshape(1, NE)

    dphi = dphi.squeeze()

    if coeffs is None:
        coeffs = torch.tensor(1.0, device=coordinates.device)
    elif coeffs.dim() == 2:
        coeffs = coeffs.squeeze()

    dphi_transposed = dphi.permute(1, 0, 2)

    if coeffs is None or torch.is_tensor(coeffs) and coeffs.numel() == 1:
        amtam_result = torch_amtam(dphi_transposed, dphi_transposed)
        Z = (
            torch_astam(volumes.t(), amtam_result)
            if coeffs is None
            else torch_astam((volumes * coeffs).t(), amtam_result)
        )
        # print("case 1")
    else:
        if coeffs.dim() == 2:
            Z = torch_astam(
                volumes.t(),
                torch.einsum(
                    "jik,ljk->ilk",
                    torch_smamt(coeffs, dphi_transposed),
                    dphi_transposed,
                ),
            )
            # print("case 2")
        elif coeffs.dim() == 3 and coeffs.size(2) == 1:
            coeffs_repeated = coeffs.repeat(1, 1, NE)
            multiplied = torch_amtam(dphi_transposed, coeffs_repeated)
            Z = torch_astam(volumes.t(), multiplied)
            # print("case 3")
        else:
            raise NotImplementedError(
                "P1->P0 averaging not implemented or coeffs dimension not supported."
            )

    # Constructing indices for the sparse matrix
    # X = elements_adj.repeat_interleave(NLB, dim=1).reshape(-1)
    # Y = elements_adj.repeat(NLB, 1).reshape(-1)

    Y = torch.reshape(
        torch.tile(elements_adj, (1, NLB)).t().contiguous(), (NLB, NLB, NE)
    )
    X = Y.permute(1, 0, 2)

    # Flatten Z to match the indices for sparse tensor creation
    Z_flattened = Z.flatten()

    # breakpoint()

    # Create the sparse tensor in PyTorch
    # indices = torch.stack([X.long(), Y.long()], dim=0)
    indices = torch.stack([X.flatten(), Y.flatten()], dim=0)
    size = (
        coordinates.size(0),
        coordinates.size(0),
    )  # Assuming a square matrix for simplicity

    M = torch.sparse_coo_tensor(indices, Z_flattened, size=size)
    # breakpoint()

    # Symmetry enforcement and further operations might need additional handling in PyTorch
    # Returning M directly for this example

    # Earlier 8/17
    # return M

    # Updated
    M_sym = (M + M.transpose(0, 1)) / 2

    return M_sym
