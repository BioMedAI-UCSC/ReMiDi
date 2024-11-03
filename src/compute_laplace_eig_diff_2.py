import time
import torch
import xitorch
from src.get_volume_mesh import get_volume_mesh
from src.mass_matrixP1_3D import mass_matrixP1_3D
from src.stiffness_matrixP1_3D import stiffness_matrixP1_3D
from src.sparse_block_diagonal import sparse_block_diagonal
from src.assembleflux_matrix import assembleflux_matrix
from src.couple_flux_matrix import couple_flux_matrix
from xitorch.linalg import symeig
from xitorch import LinearOperator


def compute_laplace_eig_diff_2(femesh, device, pde, eiglim, neig_max=None):
    start_time = time.time()
    neig_max = float("inf") if neig_max is None else neig_max

    diffusivity = pde["diffusivity"]
    relaxation = pde["relaxation"]
    ncompartment = femesh["ncompartment"]
    initial_density = pde["initial_density"]

    M_cmpts = []
    K_cmpts = []
    R_cmpts = []
    Jx_cmpts = [[] for _ in range(3)]
    rho_cmpts = []

    for icmpt in range(ncompartment):
        points = femesh["points"][icmpt]
        elements = femesh["elements"][icmpt]
        npoint_cmpt = points.size(1)

        _, volumes, _ = get_volume_mesh(points, elements)

        M_cmpt = mass_matrixP1_3D(elements, volumes, device)
        K_cmpt = stiffness_matrixP1_3D(elements, points, diffusivity[icmpt, :, :])
        R_cmpt = 1 / relaxation[icmpt] * M_cmpt

        for idim in range(3):
            Jx_cmpts[idim].append(
                mass_matrixP1_3D(elements, volumes, device, points[idim, :])
            )

        initial_density_icmpt = initial_density[icmpt].to(dtype=torch.complex128)
        rho_cmpt = initial_density_icmpt * torch.ones(
            (npoint_cmpt, 1), dtype=torch.complex128
        )

        rho_cmpts.append(rho_cmpt)
        M_cmpts.append(M_cmpt)
        K_cmpts.append(K_cmpt)
        R_cmpts.append(R_cmpt)

    M = sparse_block_diagonal(M_cmpts).to_dense()
    K = sparse_block_diagonal(K_cmpts).to_dense()
    R = sparse_block_diagonal(R_cmpts).to_dense()
    Jx = [sparse_block_diagonal(dim_cmpts).to_dense() for dim_cmpts in Jx_cmpts]

    Q_blocks = assembleflux_matrix(femesh["points"], femesh["facets"])
    Q = couple_flux_matrix(femesh, pde, Q_blocks, False).to_dense()

    rho = torch.vstack(rho_cmpts)

    print(f"Eigendecomposition of FE matrices: size {M.shape[0]} x {M.shape[1]}")

    neig_max = min(neig_max, M.shape[0])

    A = LinearOperator.m(K + Q)
    B = LinearOperator.m(M)
    eigenvalues, eigenvectors = symeig(A, neig=neig_max, M=B, mode="lowest")

    sorted_values, indices = torch.sort(eigenvalues)
    values = sorted_values
    funcs = eigenvectors[:, indices]

    negative_indices = torch.where(values < 0)[0]
    if len(negative_indices) > 0:
        for idx in negative_indices:
            print(
                f"Warning: Found negative eigenvalue at index {idx}, value {values[idx]}. Setting it to zero."
            )
        values[negative_indices] = 0.0

    neig_all = len(values)
    inds_keep = values <= eiglim

    values = values[inds_keep]
    funcs = funcs[:, inds_keep]
    neig = values.shape[0]

    if neig == neig_all and not torch.isinf(eiglim):
        print(
            "Warning: No eigenvalues were outside the interval. Consider increasing neig_max "
            "if there are more eigenvalues that may not have been found in the interval."
        )

    print(f"Found {neig} eigenvalues on [0, {eiglim}].")

    print("Normalizing eigenfunctions")
    funcs = funcs / torch.sqrt(torch.sum(funcs * (M @ funcs), dim=0))[None, :]

    print("Computing first order moments of products of eigenfunction pairs")
    moments = torch.zeros(neig, neig, 3, dtype=funcs.dtype, device=funcs.device)

    for idim in range(3):
        moments[:, :, idim] = funcs.T @ Jx[idim] @ funcs

    print("Computing T2-weighted Laplace mass matrix")
    massrelax = funcs.T @ R @ funcs

    lap_eig = {
        "values": values,
        "funcs": funcs,
        "moments": moments,
        "massrelax": massrelax,
        "totaltime": time.time() - start_time,
    }

    return lap_eig
