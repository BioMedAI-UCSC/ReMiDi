import torch

def apply_fanning(vertices, fanning_axis=1, fanning_center=0.5, fanning_strength=1.0, fanning_direction=1, fanning_mode='both'):
    """
    Apply a fanning effect to a cylindrical mesh using PyTorch.
    
    Parameters:
    - vertices: torch.Tensor of shape (3, n) where n is the number of vertices
    - fanning_axis: axis along which to apply fanning (0 for X, 1 for Y, 2 for Z)
    - fanning_center: point along the fanning axis where the fanning originates (0 to 1)
    - fanning_strength: controls the intensity of the fanning effect
    - fanning_direction: 1 for outward fanning, -1 for inward fanning
    - fanning_mode: 'both' for fanning on both sides, 'top' for fanning above center, 'bottom' for fanning below center
    
    Returns:
    - torch.Tensor of shape (3, n) with fanned vertices
    """
    # Ensure input is a PyTorch tensor
    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices, dtype=torch.float32)
    
    # Normalize the positions along the fanning axis
    axis_positions = vertices[fanning_axis]
    min_pos, max_pos = torch.min(axis_positions), torch.max(axis_positions)
    normalized_positions = (axis_positions - min_pos) / (max_pos - min_pos)
    
    # Calculate the distance from the fanning center
    distances = normalized_positions - fanning_center
    
    # Apply fanning mode
    if fanning_mode == 'top':
        distances = torch.where(distances > 0, distances, torch.zeros_like(distances))
    elif fanning_mode == 'bottom':
        distances = torch.where(distances < 0, -distances, torch.zeros_like(distances))
    else:  # 'both'
        distances = torch.abs(distances)
    
    # Calculate the fanning factor
    fanning_factor = distances * fanning_strength
    
    # Create a copy of the vertices to modify
    fanned_vertices = vertices.clone()
    
    # Apply the fanning effect to the other two axes
    for i in range(3):
        if i != fanning_axis:
            fanned_vertices[i] += fanning_factor * fanning_direction * vertices[i]
    
    return fanned_vertices

# Example usage:
# cylinder_vertices = torch.tensor(your_mesh_data, dtype=torch.float32)
# fanned_cylinder = apply_fanning(cylinder_vertices, fanning_axis=1, fanning_center=0.5, 
#                                 fanning_strength=0.5, fanning_direction=1, fanning_mode='both')