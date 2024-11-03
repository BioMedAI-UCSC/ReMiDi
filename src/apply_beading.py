import torch

def apply_beading(vertices, num_beads=3, bead_strength=0.2, axis=1):
    """
    Apply a beading effect to a cylindrical mesh using PyTorch.
    
    Parameters:
    - vertices: torch.Tensor of shape (3, n) where n is the number of vertices
    - num_beads: number of beads to apply along the cylinder
    - bead_strength: controls the intensity of the beading effect (0 to 1)
    - axis: axis along which to apply beading (0 for X, 1 for Y, 2 for Z)
    
    Returns:
    - torch.Tensor of shape (3, n) with beaded vertices
    """
    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices, dtype=torch.float32)
    
    # Normalize positions along the beading axis
    axis_positions = vertices[axis]
    min_pos, max_pos = torch.min(axis_positions), torch.max(axis_positions)
    normalized_positions = (axis_positions - min_pos) / (max_pos - min_pos)
    
    # Calculate the bead effect
    bead_phase = normalized_positions * 2 * torch.pi * num_beads
    bead_effect = torch.sin(bead_phase) * bead_strength
    
    # Apply the bead effect to the other two axes
    beaded_vertices = vertices.clone()
    for i in range(3):
        if i != axis:
            # Calculate radial distance from the axis
            radial_distance = torch.sqrt(torch.sum(vertices[[j for j in range(3) if j != axis]]**2, dim=0))
            # Apply beading effect proportional to radial distance
            beaded_vertices[i] += bead_effect * radial_distance * vertices[i] / (radial_distance + 1e-6)
    
    return beaded_vertices

# Example usage:
# cylinder_vertices = torch.tensor(your_mesh_data, dtype=torch.float32)
# beaded_cylinder = apply_beading(cylinder_vertices, num_beads=3, bead_strength=0.2, axis=1)