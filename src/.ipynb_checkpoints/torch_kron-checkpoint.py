def torch_kron(a, b):
    """
    Compute the Kronecker product of two PyTorch tensors.
    """
    # Get the shapes of the input tensors
    a_rows, a_cols = a.shape[-2], a.shape[-1]
    b_rows, b_cols = b.shape[-2], b.shape[-1]

    # Reshape a for broadcasting to match the expanded form of b
    a_reshaped = a.reshape(*a.shape[:-2], a_rows, 1, a_cols, 1)

    # Expand b to prepare for element-wise multiplication
    b_expanded = b.reshape(*b.shape[:-2], 1, b_rows, 1, b_cols)

    # Perform the element-wise multiplication
    result = a_reshaped * b_expanded

    # Calculate the resulting shape
    result_shape = a.shape[:-2] + (a_rows * b_rows, a_cols * b_cols)

    # Reshape to the final form
    return result.reshape(result_shape)


import torch
