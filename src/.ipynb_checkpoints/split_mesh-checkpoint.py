import torch


def split_mesh(femesh_all, grad=False):
    # Extract global mesh
    # points_all = femesh_all['points'].clone().detach().requires_grad_(grad) if isinstance(femesh_all['points'], torch.Tensor) else torch.tensor(femesh_all['points'], dtype=torch.float, requires_grad=grad)
    points_all = femesh_all["points"].clone()
    facets_all = (
        femesh_all["facets"].clone().detach()
        if isinstance(femesh_all["facets"], torch.Tensor)
        else torch.tensor(femesh_all["facets"], dtype=torch.long)
    )
    elements_all = (
        femesh_all["elements"].clone().detach()
        if isinstance(femesh_all["elements"], torch.Tensor)
        else torch.tensor(femesh_all["elements"], dtype=torch.long)
    )
    facetmarkers = (
        femesh_all["facetmarkers"].clone().detach()
        if isinstance(femesh_all["facetmarkers"], torch.Tensor)
        else torch.tensor(femesh_all["facetmarkers"], dtype=torch.long)
    )
    elementmarkers = (
        femesh_all["elementmarkers"].clone().detach()
        if isinstance(femesh_all["elementmarkers"], torch.Tensor)
        else torch.tensor(femesh_all["elementmarkers"], dtype=torch.long)
    )

    # Identify compartments and boundaries
    compartments, _ = torch.unique(elementmarkers, return_inverse=True)
    boundaries, _ = torch.unique(facetmarkers, return_inverse=True)
    ncompartment = compartments.size(0)
    nboundary = boundaries.size(0)

    # Split points and elements into compartments
    elements = [None] * ncompartment
    point_map = [None] * ncompartment
    points = [None] * ncompartment
    for i, compartment in enumerate(compartments):
        mask = elementmarkers == compartment
        elements[i] = elements_all[:, mask]
        _, inverse_indices = torch.unique(elements[i], return_inverse=True)
        point_map[i] = torch.unique(elements[i]).long()
        points[i] = points_all[:, point_map[i]]

    # Split facets into boundaries
    boundary_facets = [None] * nboundary
    for i, boundary in enumerate(boundaries):
        mask = facetmarkers == boundary
        boundary_facets[i] = facets_all[:, mask]

    # Renumber nodes in elements and facets
    facets = [[None for _ in range(nboundary)] for _ in range(ncompartment)]
    for icmpt in range(ncompartment):
        old_to_new_map = {
            int(old): int(new) for new, old in enumerate(point_map[icmpt])
        }
        elements[icmpt] = torch.tensor(
            [old_to_new_map[int(old)] for old in elements[icmpt].view(-1)],
            dtype=torch.long,
        ).view(*elements[icmpt].shape)

        for iboundary in range(nboundary):
            boundary_on_compartment = torch.all(
                torch.isin(boundary_facets[iboundary], point_map[icmpt]), dim=0
            )
            if boundary_on_compartment.any():
                facets[icmpt][iboundary] = torch.tensor(
                    [
                        old_to_new_map[int(old)]
                        for old in boundary_facets[iboundary].view(-1)
                    ],
                    dtype=torch.long,
                ).view(*boundary_facets[iboundary].shape)

    femesh = {
        "ncompartment": ncompartment,
        "nboundary": nboundary,
        "points": points,
        "facets": facets,
        "elements": elements,
        "point_map": point_map,
    }

    return femesh
