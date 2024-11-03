import torch


def average_distance_per_element(points, elements):
    """
    Calculate the average Euclidean distance between the vertices of each tetrahedron in a vectorized manner.

    Args:
        points (Tensor): Tensor of shape [3, N] where N is the number of vertices.
        elements (Tensor): Tensor of shape [4, M] where M is the number of tetrahedra.

    Returns:
        float: The average of the average Euclidean distances within each tetrahedron.
    """
    # Extract points corresponding to each element's vertices [3, 4, M]
    element_points = points[:, elements]  # This uses advanced indexing

    # Compute pairwise distances for all elements simultaneously [4, 4, M]
    pairwise_distances = torch.norm(
        element_points.unsqueeze(2) - element_points.unsqueeze(1), dim=0
    )

    # Mask to exclude the diagonal (self-distances)
    mask = torch.ones((4, 4), dtype=torch.bool).fill_diagonal_(0)
    masked_distances = pairwise_distances[mask].view(4, 4 - 1, -1)  # Shape [4, 3, M]

    # Compute the average distance for each element
    average_distances = masked_distances.mean(
        dim=1
    )  # Mean across 3 distances, shape [4, M]

    # Return the mean of average distances across all elements
    return average_distances.mean()
