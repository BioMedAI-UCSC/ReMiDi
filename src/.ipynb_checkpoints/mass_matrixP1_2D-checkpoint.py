import torch
from src.torch_kron import torch_kron


def mass_matrixP1_2D(elements, areas, coeffs=None):
    elements = elements.long()  # Ensure elements are integers for indexing
    # print(f'elements {elements.shape}, elements type {elements.type}')

    X = torch_kron(torch.ones((1, 3), dtype=torch.long), elements).flatten()
    # print(f'X torch = {X}\nX.shape = {X.shape}')

    Y = torch_kron(elements, torch.ones((1, 3), dtype=torch.long)).flatten()
    # print(f'Y torch = {Y}\nY.shape = {Y.shape}')

    if coeffs is None:
        Z = torch.kron(
            areas, torch.reshape((torch.ones(3, 3) + torch.eye(3)) / 12, (1, 9))
        ).view(-1)
    elif coeffs.nelement() == elements.size(0):
        # P0 coefficients
        Z = torch.kron(
            areas * coeffs,
            torch.reshape((torch.ones(3, 3) + torch.eye(3)) / 12, (1, 9)),
        ).view(-1)
    else:
        # P1 coefficients
        M1 = torch.tensor([[6, 2, 2], [2, 2, 1], [2, 1, 2]], dtype=torch.float32) / 60
        M2 = M1[[2, 0, 1], :][:, [2, 0, 1]]
        M3 = M2[[2, 0, 1], :][:, [2, 0, 1]]

        Z1 = torch.kron(areas * coeffs[elements[:, 0]], M1.reshape(1, 9))
        Z2 = torch.kron(areas * coeffs[elements[:, 1]], M2.reshape(1, 9))
        Z3 = torch.kron(areas * coeffs[elements[:, 2]], M3.reshape(1, 9))
        Z = (Z1 + Z2 + Z3).view(-1)

    # Creating a sparse tensor in PyTorch
    indices = torch.stack(
        [X, Y]
    )  # Indices must be a 2 x N tensor, where N is the number of elements
    values = Z  # Values tensor, must have the same type as indices

    # Determine the size of the sparse matrix
    # size = torch.tensor([elements.max() + 1, elements.max() + 1], dtype=torch.long)
    size = (elements.max() + 1, elements.max() + 1)

    # print(f'indices = {indices}\nindices.shape = {indices.shape}')
    # print(f'values = {values}\nvalues.shape = {values.shape}')
    # print(f'size = {size}')
    # Create the sparse tensor
    M_sparse = torch.sparse_coo_tensor(indices, values, size, dtype=torch.float32)
    # breakpoint()

    # Earlier before 8/17
    # return M_sparse

    # Update
    M_sparse_sym = (M_sparse + M_sparse.transpose(0, 1)) / 2

    return M_sparse_sym
