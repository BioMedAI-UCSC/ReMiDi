import xitorch
import xitorch.linalg
import time
import torch
from src.get_volume_mesh import get_volume_mesh
from src.mass_matrixP1_3D import mass_matrixP1_3D
from src.stiffness_matrixP1_3D import stiffness_matrixP1_3D
from src.sparse_block_diagonal import sparse_block_diagonal
from src.assembleflux_matrix import assembleflux_matrix
from src.couple_flux_matrix import couple_flux_matrix
from xitorch.linalg import symeig
from xitorch import LinearOperator


def compute_laplace_eig_diff(femesh, setup, pde, eiglim, neig_max=None):
    """
    Compute Laplace eigenvalues, functions and product moments.
    """
    start_time = time.time()

    nieg_max = float("inf") if neig_max is None else neig_max

    diffusivity = pde["diffusivity"].float()
    relaxation = pde["relaxation"].float()
    ncompartment = femesh["ncompartment"]
    initial_density = pde["initial_density"]

    M_cmpts, K_cmpts, R_cmpts = [], [], []
    Jx_cmpts = [[] for _ in range(3)]
    rho_cmpts = []

    for icmpt in range(ncompartment):
        points = femesh["points"][icmpt].float()
        elements = femesh["elements"][icmpt].long()
        npoint_cmpt = points.size(1)

        _, volumes, _ = get_volume_mesh(points, elements)
        volumes = volumes.float()

        M_cmpt = mass_matrixP1_3D(elements, volumes).float()
        K_cmpt = stiffness_matrixP1_3D(elements, points, diffusivity[icmpt, :, :]).float()
        R_cmpt = (1 / relaxation[icmpt] * M_cmpt).float()

        for idim in range(3):
            Jx_cmpts[idim].append(mass_matrixP1_3D(elements, volumes, points[idim, :]).float())

        initial_density_icmpt = initial_density[icmpt].to(dtype=torch.complex128)
        rho_cmpt = initial_density_icmpt * torch.ones((npoint_cmpt, 1), dtype=torch.complex128)

        rho_cmpts.append(rho_cmpt)
        M_cmpts.append(M_cmpt)
        K_cmpts.append(K_cmpt)
        R_cmpts.append(R_cmpt)

    M = sparse_block_diagonal(M_cmpts).float()
    K = sparse_block_diagonal(K_cmpts).float()
    R = sparse_block_diagonal(R_cmpts).float()

    Jx = [sparse_block_diagonal(dim_cmpts).float() for dim_cmpts in Jx_cmpts]

    Q_blocks = assembleflux_matrix(femesh["points"], femesh["facets"])
    Q = couple_flux_matrix(femesh, setup.pde, Q_blocks, False).float()

    rho = torch.vstack(rho_cmpts)

    print(f"Eigendecomposition of FE matrices: size {M.shape[0]} x {M.shape[1]}")

    neig_max = min(neig_max, M.shape[0])

    A = K.to_dense() + Q.to_dense()
    # Updated
    A = (A + A.transpose(0, 1)) / 2
    B = M.to_dense()

    # Updated
    B = (B + B.transpose(0, 1)) / 2
    
    A = LinearOperator.m(A)
    B = LinearOperator.m(B)
    
    eigenvalues, eigenvectors = symeig(A, neig=neig_max, M=B, mode="lowest")

    sorted_values, indices = torch.sort(eigenvalues)
    values = sorted_values
    funcs = eigenvectors[:, indices]

    negative_indices = torch.where(values < 0)[0]

    if len(negative_indices) > 0:
        for idx in negative_indices:
            print(f"Warning: Found negative eigenvalue at index {idx}, value {values[idx]}. Setting it to zero.")
        values[negative_indices] = 0.0

    neig_all = len(values)
    inds_keep = values <= eiglim

    values = values[inds_keep]
    funcs = funcs[:, inds_keep]

    neig = values.shape[0]

    if neig == neig_all and not torch.isinf(eiglim):
        print("Warning: No eigenvalues were outside the interval. Consider increasing neig_max "
              "if there are more eigenvalues that may not have been found in the interval.")

    print(f"Found {neig} eigenvalues on [0, {eiglim}].")
    print("Normalizing eigenfunctions")

    funcs = funcs / torch.sqrt(torch.sum(funcs * (M.to_dense() @ funcs), dim=0))[None, :]

    print("Computing first order moments of products of eigenfunction pairs")

    moments = torch.zeros(neig, neig, 3, dtype=torch.float, device=funcs.device)

    for idim in range(3):
        moments[:, :, idim] = funcs.T @ Jx[idim].to_dense() @ funcs

    print("Computing T2-weighted Laplace mass matrix")
    massrelax = funcs.T @ R.to_dense() @ funcs

    lap_eig = {
        "values": values,
        "funcs": funcs,
        "moments": moments,
        "massrelax": massrelax,
        "totaltime": time.time() - start_time,
    }

    return lap_eig