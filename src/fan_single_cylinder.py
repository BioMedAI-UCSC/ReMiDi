import torch

def fan_single_cylinder(vertices, bend_point=0.5, bend_angle=45, axis=1):
    """
    Apply a bending effect to a cylindrical mesh using PyTorch, similar to a bent straw.
    
    Parameters:
    - vertices: torch.Tensor of shape (3, n) where n is the number of vertices
    - bend_point: Point along the cylinder's height where the bend occurs (0 to 1)
    - bend_angle: Angle of the bend in degrees
    - axis: Axis along which the cylinder is oriented (0 for X, 1 for Y, 2 for Z)
    
    Returns:
    - torch.Tensor of shape (3, n) with bent vertices
    """
    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices, dtype=torch.float32)
    
    # Determine the axes
    axes = [0, 1, 2]
    height_axis = axis
    bend_axis = (axis + 1) % 3
    third_axis = (axis + 2) % 3
    axes.remove(height_axis)
    axes.remove(bend_axis)
    
    # Normalize positions along the height axis
    height = vertices[height_axis]
    min_height, max_height = torch.min(height), torch.max(height)
    normalized_height = (height - min_height) / (max_height - min_height)
    
    # Calculate bend angle in radians
    bend_angle_rad = torch.deg2rad(torch.tensor(bend_angle))
    
    # Apply the bend
    bent_vertices = vertices.clone()
    
    # Calculate the bend radius
    total_height = max_height - min_height
    bend_radius = total_height / bend_angle_rad
    
    # Apply bend to vertices above the bend point
    mask_above = normalized_height > bend_point
    height_above_bend = (normalized_height[mask_above] - bend_point) * total_height
    
    angle_above = height_above_bend / bend_radius
    
    # Calculate new positions for vertices above the bend point
    bent_vertices[height_axis][mask_above] = min_height + bend_point * total_height + torch.sin(angle_above) * bend_radius
    bent_vertices[bend_axis][mask_above] += (1 - torch.cos(angle_above)) * bend_radius
    
    # Rotate vertices above the bend point
    cos_angle = torch.cos(angle_above)
    sin_angle = torch.sin(angle_above)
    
    # Center of rotation
    center_height = min_height + bend_point * total_height
    center_bend = bent_vertices[bend_axis][mask_above].min()
    
    # Apply rotation
    height_diff = bent_vertices[height_axis][mask_above] - center_height
    bend_diff = bent_vertices[bend_axis][mask_above] - center_bend
    
    bent_vertices[height_axis][mask_above] = center_height + cos_angle * height_diff - sin_angle * bend_diff
    bent_vertices[bend_axis][mask_above] = center_bend + sin_angle * height_diff + cos_angle * bend_diff
    
    return bent_vertices

# Example usage:
# cylinder_vertices = torch.tensor(your_mesh_data, dtype=torch.float32)
# bent_cylinder = bend_cylinder(cylinder_vertices, bend_point=0.5, bend_angle=45, axis=1)