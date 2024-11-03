import torch
import time
from src.get_volume_mesh import get_volume_mesh
from src.mass_matrixP1_3D import mass_matrixP1_3D
from src.stiffness_matrixP1_3D import stiffness_matrixP1_3D
from src.sparse_block_diagonal import sparse_block_diagonal
from src.assembleflux_matrix import assembleflux_matrix
from src.couple_flux_matrix import couple_flux_matrix


def compute_laplace_eig(femesh_all_split, setup, pde, eiglim, neig_max=None):
    """
    Compute Laplace eigenvalues, functions and product moments.
    """
    start_time = time.time()

    femesh = femesh_all_split

    nieg_max = float("inf") if neig_max is None else neig_max

    diffusivity = setup.pde["diffusivity"]
    relaxation = setup.pde["relaxation"]
    ncompartment = femesh["ncompartment"]
    initial_density = setup.pde["initial_density"]

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
        M_cmpt = mass_matrixP1_3D(elements, volumes)
        # print(M_cmpt.shape)

        K_cmpt = stiffness_matrixP1_3D(
            elements, points, diffusivity[icmpt, :, :]
        )  # was diffusivity[:,:,icmpt]
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
        initial_density_icmpt = torch.tensor(
            initial_density[icmpt], dtype=torch.complex128
        )
        rho_cmpt = initial_density_icmpt * torch.ones(
            (npoint_cmpt, 1), dtype=torch.complex128
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

    # M_dense = M.to_dense()  # Convert sparse matrix M to dense
    # M_inv_dense = torch.inverse(M_dense)

    print(f"Eigendecomposition of FE matrices: size {M.shape[0]} x {M.shape[1]}")

    neig_max = neig_max if neig_max < M.shape[0] else M.shape[0]

    A = K + Q
    X_init = (
        torch.randn(A.size(0), neig_max, dtype=A.dtype, device=A.device) * 0.001 + 0
    )

    eigenvalues, eigenvectors = torch.lobpcg(
        A=K + Q, B=M, X=X_init, largest=False, tol=1e-14, method="ortho", n=neig_max
    )

    # Uncomment for more details
    # print(eigenvalues)
    # print(eigenvalues.shape)
    # print(eigenvectors)
    # print(eigenvectors.shape)

    # Sort the eigenvalues in ascending order and get the sorted indices
    sorted_values, indices = torch.sort(eigenvalues)

    # Update 'values' to be in sorted order. This is straightforward as 'sorted_values' is already what we need.
    values = sorted_values

    # Reorder 'funcs' (eigenvectors) according to the sorted indices of 'values'
    funcs = eigenvectors[:, indices]

    negative_indices = torch.where(values < 0)[0]

    if len(negative_indices) > 0:
        # Print indices and corresponding negative values for warning
        for idx in negative_indices:
            print(
                f"Warning: Found negative eigenvalue at index {idx}, value {values[idx]}. Setting it to zero."
            )
        # Set negative eigenvalues to zero
        values[negative_indices] = 0.0

    # All the eigen values we have found
    neig_all = len(values)
    # Find indices of eigenvalues less than or equal to 'eiglim'
    inds_keep = values <= eiglim

    # Filter eigenvalues and corresponding eigenvectors
    values = values[inds_keep]
    funcs = funcs[:, inds_keep]

    # Update the count of eigenvalues after filtering
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

    # Loop over the 3 dimensions
    for idim in range(3):
        # Compute the moment matrix for each dimension
        moments[:, :, idim] = funcs.T @ Jx[idim] @ funcs

    print("Computing T2-weighted Laplace mass matrix")
    # Compute the T2-weighted Laplace mass matrix
    massrelax = funcs.T @ R @ funcs
    # breakpoint()
    # Create the result dictionary
    lap_eig = {
        "values": values,
        "funcs": funcs,
        "moments": moments,
        "massrelax": massrelax,
        "totaltime": time.time() - start_time,
    }

    return lap_eig
