import torch
from src.get_volume_mesh import get_volume_mesh
from src.get_surface_mesh import get_surface_mesh


def get_vol_sa(femesh):
    ncompartment = len(femesh["points"])
    volumes = torch.zeros(ncompartment, dtype=torch.float32)
    surface_areas = torch.zeros(ncompartment, dtype=torch.float32)

    for i in range(ncompartment):
        points = femesh["points"][i]
        elements = femesh["elements"][i]
        facets = femesh["facets"][i]

        if facets is not None and len(facets) > 0:
            total_volume, _, _ = get_volume_mesh(points, elements)
            volumes[i] = total_volume

            total_area, _, _, _ = get_surface_mesh(points, facets)
            surface_areas[i] = total_area
        else:
            # Handle the case where no facets are defined for a compartment
            volumes[i], _, _ = get_volume_mesh(points, elements)
            surface_areas[i] = 0

    return volumes, surface_areas
