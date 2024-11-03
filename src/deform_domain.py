import torch


def deform_domain(nodes, deformation):
    height = torch.max(nodes[2]) - min(nodes[2])
    width = torch.max(nodes[0]) - min(nodes[0])

    bend = deformation[0]
    twist = deformation[1]

    thvec = nodes[2] / height * twist

    nodes[0] = nodes[0] + bend * 30 * width * ((nodes[2] / height) ** 2)

    max_values, _ = torch.max(nodes[0:2, :], dim=1)
    min_values, _ = torch.min(nodes[0:2, :], dim=1)
    center = 0.5 * (max_values + min_values)
    center = center.reshape(2, 1)
    center_expanded = center.expand_as(nodes[0:2, :])

    nodes[0:2] = nodes[0:2] - center_expanded

    cos_th = torch.cos(thvec)
    sin_th = torch.sin(thvec)
    nodes[0] = cos_th * nodes[0] - sin_th * nodes[1]
    nodes[1] = sin_th * nodes[0] + cos_th * nodes[1]

    nodes[0:2] = nodes[0:2] + center_expanded

    return nodes
