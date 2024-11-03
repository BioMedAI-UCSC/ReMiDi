def ellipsoidal_scale_mesh(mesh, scale_factors):
    points = mesh["points"].clone().detach()

    # Calculate the center of the mesh
    center = points.mean(dim=1, keepdim=True)

    # Move points to be centered at the origin
    centered_points = points - center

    # Apply scaling (ensure scale_factors is a tensor of shape (3,))
    scale_factors = scale_factors.view(-1, 1)  # Reshape to (3, 1) for broadcasting
    scaled_points = centered_points * scale_factors

    # Move points back to the original center
    new_points = scaled_points + center

    # Update the mesh dictionary with new points
    # mesh["points"] = new_points

    new_mesh = {}
    for key, value in mesh.items():
        if key == 'points':
            new_mesh[key] = new_points
        else:
            new_mesh[key] = value.clone().detach()

    return new_mesh
