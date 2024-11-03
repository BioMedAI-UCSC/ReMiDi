import torch

def get_volume_mesh(points, elements):
    # Ensure input tensors are of float type for computations
    points = points
    elements = elements.long()

    nelement = elements.shape[1]
    volumes = torch.zeros(nelement, dtype=torch.float64)
    centers = torch.zeros((3, nelement), dtype=torch.float64)

    # Compute as per the given logic
    x = points[:, elements]
    
    centers = torch.mean(x, dim=1)

    for i in range(nelement):
        areavectors = torch.linalg.cross(
            x[:, 1, i] - x[:, 3, i], x[:, 2, i] - x[:, 3, i]
        )
        volumes[i] = torch.abs(torch.dot(x[:, 0, i] - x[:, 3, i], areavectors)) / 6

    total_volume = torch.sum(volumes)
    return total_volume, volumes, centers
