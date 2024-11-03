from src.build_adjacency_list import build_adjacency_list


def laplacian_smoothing(points, elements, alpha=0.5):
    adjacency_list = build_adjacency_list(elements)
    smoothed_points = points.clone()

    for i in range(points.shape[1]):
        if i in adjacency_list:
            neighbors = list(adjacency_list[i])
            if neighbors:
                neighbor_points = points[:, neighbors]
                average_position = neighbor_points.mean(dim=1)
                local_alpha = alpha  # Can be adjusted based on vertex property
                smoothed_points[:, i] = (
                    local_alpha * average_position
                    + (1 - local_alpha) * smoothed_points[:, i]
                )
            else:
                smoothed_points[:, i] = smoothed_points[:, i]
        else:
            smoothed_points[:, i] = smoothed_points[:, i]

    return smoothed_points
