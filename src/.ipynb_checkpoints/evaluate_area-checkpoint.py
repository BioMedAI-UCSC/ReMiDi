import torch


def evaluate_area(elements, coordinates):
    # Indexing with long tensor for compatibility
    elements = elements.long()
    v1 = coordinates[elements[:, 1], :] - coordinates[elements[:, 0], :]
    v2 = coordinates[elements[:, 2], :] - coordinates[elements[:, 0], :]

    # Initialize a 3D tensor for storing the 2x2 matrices for each element
    matrix_3D = torch.zeros(
        (len(elements), 2, 2), dtype=coordinates.dtype, device=coordinates.device
    )
    matrix_3D[:, 0, 0] = torch.sum(v1 * v1, dim=1)
    matrix_3D[:, 0, 1] = torch.sum(v1 * v2, dim=1)
    matrix_3D[:, 1, 0] = matrix_3D[:, 0, 1]
    matrix_3D[:, 1, 1] = torch.sum(v2 * v2, dim=1)

    # Calculate the determinant of each 2x2 matrix and then compute the area
    elements2area = torch.sqrt(torch.det(matrix_3D)) / 2

    return elements2area
