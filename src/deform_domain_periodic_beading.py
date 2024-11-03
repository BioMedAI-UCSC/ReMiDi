import torch

def deform_domain_periodic_beading(nodes_in, amplitude, num_periods):

    nodes = nodes_in.clone()
    
    # Get the bounds of the shape
    z_min, z_max = torch.min(nodes[2]), torch.max(nodes[2])
    height = z_max - z_min
    
    # Normalize z coordinates to [0, 1]
    z_norm = (nodes[2] - z_min) / height
    if torch.any(torch.isnan(z_norm)):
        print("nan z_norm")
    
    # Create periodic deformation
    deformation = amplitude * torch.sin(2 * torch.pi * num_periods * z_norm)
    if torch.any(torch.isnan(deformation)):
        print("nan deformation")
    
    # Apply deformation radially
    radius = torch.sqrt(nodes[0]**2 + nodes[1]**2)
    if torch.any(torch.isnan(radius)):
        print("nan radius")
    angle = torch.atan2(nodes[1], nodes[0])
    if torch.any(torch.isnan(angle)):
        print("nan angle")
    
    new_radius = radius * (1 + deformation)
    if torch.any(torch.isnan(new_radius)):
        print("nan new_radius")
    
    # Update x and y coordinates
    nodes[0] = new_radius * torch.cos(angle)
    if torch.any(torch.isnan(nodes[0])):
        print("nan nodes[0]")
    nodes[1] = new_radius * torch.sin(angle)
    if torch.any(torch.isnan(nodes[1])):
        print("nan nodes[1]")
    
    return nodes
