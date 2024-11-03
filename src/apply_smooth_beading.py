import torch

def apply_smooth_beading(vertices, num_beads=3, bead_strength=0.2, axis=1, smoothness=2.0, taper_factor=0.2):
    """
    Apply a smooth, non-linear beading effect to a cylindrical mesh using PyTorch.
    
    Parameters:
    - vertices: torch.Tensor of shape (3, n) where n is the number of vertices
    - num_beads: number of beads to apply along the cylinder
    - bead_strength: controls the intensity of the beading effect (0 to 1)
    - axis: axis along which to apply beading (0 for X, 1 for Y, 2 for Z)
    - smoothness: controls the smoothness of the bead transitions (higher values create smoother transitions)
    - taper_factor: controls how much the beads taper at the ends (0 to 1)
    
    Returns:
    - torch.Tensor of shape (3, n) with smoothly beaded vertices
    """
    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices, dtype=torch.float32)
    
    # Normalize positions along the beading axis
    axis_positions = vertices[axis]
    min_pos, max_pos = torch.min(axis_positions), torch.max(axis_positions)
    normalized_positions = (axis_positions - min_pos) / (max_pos - min_pos)
    
    # Calculate the bead effect with multiple frequencies
    bead_phase = normalized_positions * 2 * torch.pi * num_beads
    bead_effect = torch.sin(bead_phase)
    bead_effect += 0.5 * torch.sin(2 * bead_phase)  # Add higher frequency for detail
    bead_effect = torch.tanh(smoothness * bead_effect) / torch.tanh(torch.tensor(smoothness))  # Smooth transitions
    
    # Apply tapering effect
    taper = 1 - taper_factor * (1 - torch.sin(normalized_positions * torch.pi))
    bead_effect *= taper
    
    # Scale the effect
    bead_effect *= bead_strength
    
    # Apply the bead effect to the other two axes
    beaded_vertices = vertices.clone()
    non_axis_indices = [i for i in range(3) if i != axis]
    radial_distance = torch.sqrt(torch.sum(vertices[non_axis_indices]**2, dim=0))
    
    for i in non_axis_indices:
        # Apply beading effect proportional to radial distance with smooth falloff
        falloff = torch.exp(-((normalized_positions - 0.5) ** 2) / 0.15)  # Gaussian falloff
        beaded_vertices[i] += bead_effect * falloff * radial_distance * vertices[i] / (radial_distance + 1e-6)
    
    return beaded_vertices

# Example usage:
# cylinder_vertices = torch.tensor(your_mesh_data, dtype=torch.float32)
# beaded_cylinder = apply_smooth_beading(cylinder_vertices, num_beads=3, bead_strength=0.2, axis=1, smoothness=2.0, taper_factor=0.2)