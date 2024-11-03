from src.get_volume_mesh import get_volume_mesh
from src.mass_matrixP1_3D import mass_matrixP1_3D
from src.stiffness_matrixP1_3D import stiffness_matrixP1_3D
from src.sparse_block_diagonal import sparse_block_diagonal
from src.assembleflux_matrix import assembleflux_matrix
from src.couple_flux_matrix import couple_flux_matrix
import torch


def get_fe_matrices(setup, pde, femesh, device):

    diffusivity = pde["diffusivity"]
    relaxation = pde["relaxation"]
    ncompartment = femesh["ncompartment"]
    initial_density = pde["initial_density"]

    # Initialize compartment FE matrices
    M_cmpts = []
    K_cmpts = []
    R_cmpts = []
    # Jx_cmpts initialization assumes a 3D context, but PyTorch does not use nested lists for this.
    # If Jx_cmpts refers to a tensor for each dimension, consider initializing empty tensors if the size is known,
    # or keep as a list to append tensors to, if the sizes vary.
    # Jx_cmpts = [[] for _ in range(ncompartment)]  # Adjust based on actual use
    Jx_cmpts = [[] for _ in range(3)]
    rho_cmpts = []

    for icmpt in range(ncompartment):
        points = femesh["points"][icmpt]  # Assuming this is already a PyTorch tensor
        elements = femesh["elements"][
            icmpt
        ]  # Assuming this is already a PyTorch tensor and long type for indexing
        npoint_cmpt = points.size(1)  # Using .size() for PyTorch tensors
        # print("npoint_cmpt", npoint_cmpt)

        # Assemble matrices
        # Assuming get_volume_mesh returns a tuple where the second value is volumes as a PyTorch tensor
        _, volumes, _ = get_volume_mesh(points, elements)
        # Assuming mass_matrixP1_3D has been adapted to accept PyTorch tensors and returns a PyTorch sparse tensor

        M_cmpt = mass_matrixP1_3D(elements, volumes, device)

        # print(M_cmpt.shape)

        K_cmpt = stiffness_matrixP1_3D(
            elements, points, diffusivity[icmpt, :, :]
        )  # was diffusivity[:,:,icmpt]
        # print("OK so far")
        R_cmpt = 1 / relaxation[icmpt] * M_cmpt
        # print("R_cmt shape", R_cmpt.shape)
        # print("R_cmt", R_cmpt)
        # print("R_cmt", R_cmpt.to_dense())

        # Assemble moment matrices
        for idim in range(3):
            # print(f"Once {idim}")
            Jx_cmpts[idim].append(mass_matrixP1_3D(elements, volumes, points[idim, :]))
            # print(len(Jx_cmpts[idim]))

        # breakpoint()
        initial_density_icmpt = initial_density[icmpt].to(dtype=torch.complex64)
        rho_cmpt = initial_density_icmpt * torch.ones(
            (npoint_cmpt, 1), dtype=torch.complex64
        )

        # Append the complex tensor to rho_cmpts list
        rho_cmpts.append(rho_cmpt)

        # Assuming M_cmpts is intended to store the results for each compartment
        M_cmpts.append(M_cmpt)
        K_cmpts.append(K_cmpt)
        R_cmpts.append(R_cmpt)

    # For M, K, and R
    M = sparse_block_diagonal(M_cmpts)
    K = sparse_block_diagonal(K_cmpts)
    R = sparse_block_diagonal(R_cmpts)

    # For Jx, assuming Jx_cmpts is a list of lists of tensors
    Jx = [sparse_block_diagonal(dim_cmpts) for dim_cmpts in Jx_cmpts]

    # Q matrix assembly here (assuming assemble_flux_matrix and couple_flux_matrix are defined)
    Q_blocks = assembleflux_matrix(femesh["points"], femesh["facets"])
    # print(len(Q_blocks))
    Q = couple_flux_matrix(femesh, setup.pde, Q_blocks, False)

    rho = torch.vstack(rho_cmpts)

    return M_cmpts, K, R, Q, rho, Jx, M
