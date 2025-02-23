import torch


def prepare_pde(setup):
    # Extract compartment information using attribute access
    ncell = setup.geometry["ncell"]
    cell_shape = setup.geometry["cell_shape"]
    include_in = setup.geometry["include_in"]
    in_ratio = setup.geometry["in_ratio"]
    include_ecs = setup.geometry["ecs_shape"] != "no_ecs"
    ecs_ratio = setup.geometry["ecs_ratio"]
    pde = setup.pde

    # Check for correct radius ratios and that neurons do not have in-compartments
    assert not include_in or (0 < in_ratio < 1 and cell_shape != "neuron")
    assert not include_ecs or (0 < ecs_ratio)

    # Number of compartments
    ncompartment = (1 + include_in) * ncell + include_ecs

    # Find number of boundaries
    if cell_shape == "cylinder":
        nboundary = (include_in + 1) * 2 * ncell + include_ecs
    elif cell_shape in ["sphere", "neuron"]:
        nboundary = (include_in + 1) * ncell + include_ecs

    compartments = []
    boundaries = []

    if include_in:
        compartments += ["in"] * ncell
        boundaries += ["in,out"] * ncell

    compartments += ["out"] * ncell

    if include_ecs:
        compartments.append("ecs")
        boundaries += ["out,ecs"] * ncell
    else:
        boundaries += ["out"] * ncell

    if cell_shape == "cylinder":
        if include_in:
            boundaries += ["in"] * ncell
        boundaries += ["out"] * ncell
        if include_ecs:
            boundaries.append("ecs")
    elif cell_shape in ["sphere", "neuron"] and include_ecs:
        boundaries.append("ecs")

    # Diffusion coefficients (tensorize if scalars)
    diffusivity_in = pde["diffusivity_in"] * torch.eye(3)
    diffusivity_out = pde["diffusivity_out"] * torch.eye(3)
    diffusivity_ecs = pde["diffusivity_ecs"] * torch.eye(3)

    diffusivity = torch.zeros(3, 3, ncompartment)
    relaxation = torch.zeros(ncompartment)
    initial_density = torch.zeros(ncompartment)
    permeability = torch.zeros(nboundary)

    for i, comp in enumerate(compartments):
        if comp == "in":
            diffusivity[:, :, i] = diffusivity_in
            relaxation[i] = pde["relaxation_in"]
            initial_density[i] = pde["initial_density_in"]
        elif comp == "out":
            diffusivity[:, :, i] = diffusivity_out
            relaxation[i] = pde["relaxation_out"]
            initial_density[i] = pde["initial_density_out"]
        elif comp == "ecs":
            diffusivity[:, :, i] = diffusivity_ecs
            relaxation[i] = pde["relaxation_ecs"]
            initial_density[i] = pde["initial_density_ecs"]

    for i, boundary in enumerate(boundaries):
        if boundary == "in":
            permeability[i] = pde["permeability_in"]
        elif boundary == "out":
            permeability[i] = pde["permeability_out"]
        elif boundary == "ecs":
            permeability[i] = pde["permeability_ecs"]
        elif boundary == "in,out":
            permeability[i] = pde["permeability_in_out"]
        elif boundary == "out,ecs":
            permeability[i] = pde["permeability_out_ecs"]

    # Update the setup.pde dictionary directly
    setup.pde["diffusivity"] = diffusivity.T
    setup.pde["relaxation"] = relaxation
    setup.pde["initial_density"] = initial_density
    setup.pde["permeability"] = permeability
    setup.pde["compartments"] = compartments
    setup.pde["boundaries"] = boundaries

    return setup.pde
