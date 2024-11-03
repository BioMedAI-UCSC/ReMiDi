import torch
from torch import sparse_coo_tensor as coo_tensor


def mass_matrixP1_3D(elements, volumes, coeffs=None):
    elements = elements.t().contiguous().long()
    volumes = volumes.t().contiguous()

    num_elements = elements.size(1)

    if coeffs is not None:
        coeffs = coeffs.t()

    ones_4 = torch.ones((1, 4), dtype=torch.long, device=elements.device)
    X = torch.kron(ones_4, elements)
    Y = torch.kron(elements, ones_4)
    # print(X.shape, Y.shape)

    if coeffs is None:
        matrix = (torch.ones(4, 4) + torch.eye(4)) / 20
        matrix_flat = matrix.reshape(
            1, -1
        )  # Reshape similar to MATLAB's reshape(..., 1, 16)
        Z = torch.kron(volumes.unsqueeze(1), matrix_flat)
        # print(Z.shape)
    elif len(coeffs) == num_elements:
        matrix = (torch.ones(4, 4, device=elements.device) + torch.eye(4, device=elements.device)) / 20
        matrix_flat = matrix.reshape(1, -1)
        Z = torch.kron((volumes * coeffs).unsqueeze(1), matrix_flat)
        # print(Z.shape)
    else:
        # Define the matrices
        M1 = (
            torch.tensor(
                [[6, 2, 2, 2], [2, 2, 1, 1], [2, 1, 2, 1], [2, 1, 1, 2]],
                dtype=torch.float32,
            )
            / 120
        )
        sequence = [3, 0, 1, 2]
        M2 = M1[sequence, :][:, sequence]
        M3 = M2[sequence, :][:, sequence]
        M4 = M3[sequence, :][:, sequence]

        # Reshape matrices for Kronecker product
        M1_flat = M1.reshape(1, -1)
        M2_flat = M2.reshape(1, -1)
        M3_flat = M3.reshape(1, -1)
        M4_flat = M4.reshape(1, -1)

        # Compute the Kronecker products and sum them up
        # Note: coeffs(elements[:, i]) in MATLAB/Python is indexed by coeffs[elements[:, i]] in PyTorch
        Z1 = torch.kron(volumes * coeffs[elements[:, 0]], M1_flat)
        Z2 = torch.kron(volumes * coeffs[elements[:, 1]], M2_flat)
        Z3 = torch.kron(volumes * coeffs[elements[:, 2]], M3_flat)
        Z4 = torch.kron(volumes * coeffs[elements[:, 3]], M4_flat)
        Z = Z1 + Z2 + Z3 + Z4

        # Flatten the result if necessary (PyTorch's kron result might already be in the desired shape)
        Z_flattened = Z.flatten()

        # print(Z.shape)

    Z = Z.view(-1)
    # print(Z.shape)

    # Flatten X and Y to match the nnz expected format
    X_flat = X.view(-1)  # Flatten X
    Y_flat = Y.view(-1)  # Flatten Y
    Z_flat = Z.view(-1)  # Flatten Z to match the nnz

    # print(Z)
    # print(torch.max(Z))

    # Ensure the indices tensor is correctly shaped as [2, nnz]
    indices = torch.stack([X_flat, Y_flat], dim=0)

    # Now, use the flattened indices and values to create the sparse tensor
    size = (
        torch.max(X_flat).item() + 1,
        torch.max(Y_flat).item() + 1,
    )  # Adjust size calculation
    M = torch.sparse_coo_tensor(indices, Z_flat, size=size)

    # Enforcing symmetry is more complex in sparse format, so it's commented out here
    # M = (M + M.t()) / 2

    # breakpoint()

    # Earlier 8/17
    # return M

    # Updated
    M_sym = (M + M.t()) / 2

    return M_sym
