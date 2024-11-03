import torch


def sparse_block_diagonal(sparse_matrices):
    """
    Create a block diagonal sparse matrix from a list of 2D sparse tensors.

    @param sparse_matrices: List of 2D sparse tensors.
    @return: A large sparse tensor where input tensors form the block diagonal.
    """
    indices_list = []
    values_list = []
    total_rows = total_cols = 0

    for sparse_mat in sparse_matrices:
        # Ensure the sparse matrix is coalesced
        coalesced_sparse_mat = sparse_mat.coalesce()

        indices = coalesced_sparse_mat.indices()
        values = coalesced_sparse_mat.values()
        rows, cols = coalesced_sparse_mat.size()

        # Adjust indices based on the current total size
        new_indices = indices + torch.tensor(
            [[total_rows], [total_cols]], device=indices.device
        )
        indices_list.append(new_indices)
        values_list.append(values)

        total_rows += rows
        total_cols += cols

    new_indices = torch.cat(indices_list, dim=1)
    new_values = torch.cat(values_list, dim=0)
    new_size = (total_rows, total_cols)

    return torch.sparse_coo_tensor(new_indices, new_values, new_size)
