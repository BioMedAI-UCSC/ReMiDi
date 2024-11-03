import torch

def fan_single_cylinder_batches(vertices, bend_point, bend_angle, axis=2):
    """
    Apply a bending effect to a cylindrical mesh using PyTorch, similar to a bent straw.
    
    Parameters:
    - vertices: torch.Tensor of shape (batch_size, 3, n) where n is the number of vertices
    - bend_point: torch.Tensor of shape (batch_size,)
    - bend_angle: torch.Tensor of shape (batch_size,)
    - axis: int, axis along which the cylinder is oriented (0 for X, 1 for Y, 2 for Z)
    
    Returns:
    - torch.Tensor of shape (batch_size, 3, n) with bent vertices
    """
    batch_size, _, num_vertices = vertices.shape

    # print(vertices.shape)
    
    # Ensure bend_point and bend_angle are the correct shape
    bend_point = bend_point.view(batch_size, 1)
    bend_angle = bend_angle.view(batch_size, 1)
    
    # Determine the axes
    height_axis = axis
    bend_axis = (axis + 1) % 3
    
    # Normalize positions along the height axis
    height = vertices[:, height_axis, :]
    min_height = height.min(dim=1, keepdim=True)[0]
    max_height = height.max(dim=1, keepdim=True)[0]
    normalized_height = (height - min_height) / (max_height - min_height)
    
    # Calculate bend angle in radians
    bend_angle_rad = torch.deg2rad(bend_angle)
    
    # Apply the bend
    bent_vertices = vertices.clone()
    
    # Calculate the bend radius
    total_height = max_height - min_height
    bend_radius = total_height / bend_angle_rad
    
    # Apply bend to vertices above the bend point
    mask_above = normalized_height > bend_point
    
    # Check if there are any vertices above the bend point
    if torch.any(mask_above):
        height_above_bend = (normalized_height[mask_above] - bend_point.unsqueeze(-1).expand_as(normalized_height)[mask_above]) * total_height.unsqueeze(-1).expand_as(normalized_height)[mask_above]
        
        angle_above = height_above_bend / bend_radius.unsqueeze(-1).expand_as(normalized_height)[mask_above]
        
        # Calculate new positions for vertices above the bend point
        bent_vertices[:, height_axis][mask_above] = (
            min_height.unsqueeze(-1).expand_as(normalized_height)[mask_above] + 
            bend_point.unsqueeze(-1).expand_as(normalized_height)[mask_above] * total_height.unsqueeze(-1).expand_as(normalized_height)[mask_above] + 
            torch.sin(angle_above) * bend_radius.unsqueeze(-1).expand_as(normalized_height)[mask_above]
        )
        bent_vertices[:, bend_axis][mask_above] += (1 - torch.cos(angle_above)) * bend_radius.unsqueeze(-1).expand_as(normalized_height)[mask_above]
        
        # Rotate vertices above the bend point
        cos_angle = torch.cos(angle_above)
        sin_angle = torch.sin(angle_above)
        
        # Center of rotation
        center_height = min_height + bend_point * total_height
        center_bend = bent_vertices[:, bend_axis][mask_above].view(batch_size, -1).min(dim=1, keepdim=True)[0]
        
        # Apply rotation
        height_diff = bent_vertices[:, height_axis][mask_above] - center_height.unsqueeze(-1).expand_as(normalized_height)[mask_above]
        bend_diff = bent_vertices[:, bend_axis][mask_above] - center_bend.unsqueeze(-1).expand_as(normalized_height)[mask_above]
        
        bent_vertices[:, height_axis][mask_above] = (
            center_height.unsqueeze(-1).expand_as(normalized_height)[mask_above] + 
            cos_angle * height_diff - 
            sin_angle * bend_diff
        )
        bent_vertices[:, bend_axis][mask_above] = (
            center_bend.unsqueeze(-1).expand_as(normalized_height)[mask_above] + 
            sin_angle * height_diff + 
            cos_angle * bend_diff
        )
    
    return bent_vertices