from src.evaluate_area import evaluate_area
from src.mass_matrixP1_2D import mass_matrixP1_2D
import torch


def flux_matrixP1_3D(neumann, coordinates, coeffs=None):
    neumann2area = evaluate_area(
        neumann, coordinates
    )  # Assume this is now a PyTorch operation
    if coeffs is None:
        M = mass_matrixP1_2D(neumann, neumann2area)  # Assume adapted to PyTorch
    else:
        M = mass_matrixP1_2D(neumann, neumann2area, coeffs)  # Assume adapted to PyTorch

    # Conversion of sparse matrix handling to PyTorch
    M = M.coalesce()  # Ensure M is in COO format and coalesced
    max_index = max(
        coordinates.size(0),
        M.indices()[0].max().item() + 1,
        M.indices()[1].max().item() + 1,
    )

    # Add line here
    indices = M.indices()

    # Get the values from M
    values = M.values()

    # Define the shape of the new matrix
    shape = (max_index, max_index)

    # Create the new sparse COO tensor in PyTorch
    M_boundary = torch.sparse_coo_tensor(indices, values, size=shape)

    return M_boundary, neumann2area
