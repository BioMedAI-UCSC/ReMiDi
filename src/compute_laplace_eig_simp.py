import xitorch
import xitorch.linalg
from xitorch.linalg import symeig
from xitorch import LinearOperator
import torch
import time


def compute_laplace_eig_simp(M, K, R, Q, rho, Jx, eiglim, neig_max=None):

    start_time = time.time()

    # M = sparse_block_diagonal(M_cmpts)

    print(f"Eigendecomposition of FE matrices: size {M.shape[0]} x {M.shape[1]}")

    neig_max = neig_max if neig_max < M.shape[0] else M.shape[0]

    # A = K + Q
    # X_init = torch.randn(A.size(0), neig_max, dtype=A.dtype, device=A.device) * 0.001 + 0

    A = K.to_dense() + Q.to_dense()
    B = M.to_dense()
    A = LinearOperator.m(A)
    B = LinearOperator.m(B)
    # eigenvalues, eigenvectors = eigs(A, B, neig_max, 1e-14)
    eigenvalues, eigenvectors = symeig(A, neig=neig_max, M=B, mode="lowest")
    # print(eigenvalues)

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

    funcs = (
        funcs / torch.sqrt(torch.sum(funcs * (M.to_dense() @ funcs), dim=0))[None, :]
    )

    print("Computing first order moments of products of eigenfunction pairs")

    moments = torch.zeros(neig, neig, 3, dtype=funcs.dtype, device=funcs.device)

    # Loop over the 3 dimensions
    for idim in range(3):
        # Compute the moment matrix for each dimension
        moments[:, :, idim] = funcs.T @ Jx[idim].to_dense() @ funcs

    print("Computing T2-weighted Laplace mass matrix")
    # Compute the T2-weighted Laplace mass matrix
    massrelax = funcs.T @ R.to_dense() @ funcs
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
