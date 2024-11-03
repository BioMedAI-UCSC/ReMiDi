import torch


def scale_mesh(mesh, scale_factor):
    points = mesh["points"]

    # Calculate the center of the mesh
    center = points.mean(dim=1, keepdim=True)

    # Move points to be centered at the origin
    centered_points = points - center

    # Apply scaling
    scaled_points = centered_points * scale_factor

    # Move points back to the original center
    new_points = scaled_points + center

    # Update the mesh dictionary with new points
    mesh["points"] = new_points

    return mesh
