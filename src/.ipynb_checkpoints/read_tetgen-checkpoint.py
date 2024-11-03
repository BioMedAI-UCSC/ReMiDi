import torch


def read_tetgen(filename, grad=False):
    """
    Read mesh from TetGen output files using PyTorch.

    Args:
        filename (str): Base filename without extension.

    Returns:
        dict: Dictionary containing mesh data with points, facets, elements, and markers.
    """
    femesh_all = {}

    # Read nodes
    with open(f"{filename}.node", "r") as file:
        lines = file.readlines()
        npoint = int(lines[0].split()[0])
        points = torch.zeros((3, npoint), dtype=torch.float)
        for i in range(1, 1 + npoint):
            _, x, y, z = lines[i].split()[:4]
            points[:, i - 1] = torch.tensor(
                [float(x), float(y), float(z)], dtype=torch.float
            )

    # Read facets and facet markers
    with open(f"{filename}.face", "r") as file:
        lines = file.readlines()
        nfacet = int(lines[0].split()[0])
        facets = torch.zeros((3, nfacet), dtype=torch.int)
        facetmarkers = torch.zeros(nfacet, dtype=torch.int)
        for i in range(1, 1 + nfacet):
            _, v1, v2, v3, marker = lines[i].split()[:5]
            facets[:, i - 1] = torch.tensor(
                [int(v1) - 1, int(v2) - 1, int(v3) - 1], dtype=torch.int
            )
            facetmarkers[i - 1] = int(marker)

    # Read elements and element markers
    with open(f"{filename}.ele", "r") as file:
        lines = file.readlines()
        nelement = int(lines[0].split()[0])
        elements = torch.zeros((4, nelement), dtype=torch.int)
        elementmarkers = torch.zeros(nelement, dtype=torch.int)
        for i in range(1, 1 + nelement):
            _, v1, v2, v3, v4, marker = lines[i].split()[:6]
            elements[:, i - 1] = torch.tensor(
                [int(v1) - 1, int(v2) - 1, int(v3) - 1, int(v4) - 1], dtype=torch.int
            )
            elementmarkers[i - 1] = int(marker)

    femesh_all["points"] = points.requires_grad_(grad)
    femesh_all["facets"] = facets
    femesh_all["elements"] = elements
    femesh_all["facetmarkers"] = facetmarkers
    femesh_all["elementmarkers"] = elementmarkers

    return femesh_all
