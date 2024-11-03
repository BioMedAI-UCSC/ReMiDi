import torch


def shapeder(point, etype):
    nod = point.size(0)
    nop = point.size(1)

    if nod == 1:
        # 1-D elements
        l1 = point[0, :]
        l2 = 1 - l1
        if etype == "P0":
            dshape = torch.zeros((1, 1, nop))
        elif etype == "P1":
            dshape = (
                torch.tensor([[1, -1]], dtype=torch.float32)
                .view(1, 2, 1)
                .expand(-1, -1, nop)
            )
        elif etype == "P2":
            dshape = torch.tensor(
                [[4 * l1 - 1, -4 * l2 + 1, 4 * (l2 - l1)]], dtype=torch.float32
            ).view(1, 3, nop)
        else:
            raise ValueError("Only P1 and P2 elements implemented.")
    elif nod == 2:
        # 2-D elements
        l1, l2 = point[0, :], point[1, :]
        l3 = 1 - l1 - l2
        if etype == "P0":
            dshape = torch.zeros((2, 1, nop))
        elif etype == "P1":
            dshape = (
                torch.tensor([[1, 0, -1], [0, 1, -1]], dtype=torch.float32)
                .view(2, 3, 1)
                .expand(-1, -1, nop)
            )
        elif etype == "P2":
            dshape = torch.tensor(
                [
                    [-4 * l3 + 1, 4 * l1 - 1, 0, 4 * l2, -4 * l2, 4 * (l3 - l1)],
                    [-4 * l3 + 1, 0, 4 * l2 - 1, 4 * l1, 4 * (l3 - l2), -4 * l1],
                ],
                dtype=torch.float32,
            ).view(2, 6, nop)
        else:
            raise ValueError("Only P1 and P2 elements implemented.")
    elif nod == 3:
        # 3-D elements
        l1, l2, l3 = point[0, :], point[1, :], point[2, :]
        l4 = 1 - l1 - l2 - l3
        if etype == "P0":
            dshape = torch.zeros((3, 1, nop))
        elif etype == "P1":
            dshape = (
                torch.tensor(
                    [[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, -1]], dtype=torch.float32
                )
                .view(3, 4, 1)
                .expand(-1, -1, nop)
            )
        elif etype == "P2":
            dshape = torch.tensor(
                [
                    [
                        -4 * l4 + 1,
                        4 * l1 - 1,
                        0,
                        0,
                        4 * (l4 - l1),
                        -4 * l2,
                        -4 * l3,
                        4 * l2,
                        4 * l3,
                        0,
                    ],
                    [
                        -4 * l4 + 1,
                        0,
                        4 * l2 - 1,
                        0,
                        -4 * l1,
                        4 * (l4 - l2),
                        -4 * l3,
                        4 * l1,
                        0,
                        4 * l3,
                    ],
                    [
                        -4 * l4 + 1,
                        0,
                        0,
                        4 * l3 - 1,
                        -4 * l1,
                        -4 * l2,
                        4 * (l4 - l3),
                        0,
                        4 * l1,
                        4 * l2,
                    ],
                ],
                dtype=torch.float32,
            ).view(3, 10, nop)
        elif etype == "Q1":
            # Tri-linear (8 node) hexahedron
            s = (
                torch.tensor(
                    [
                        [0, 0, 0],
                        [1, 0, 0],
                        [1, 1, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 1],
                        [1, 1, 1],
                        [0, 1, 1],
                    ],
                    dtype=torch.float32,
                )
                * 2
                - 1
            )
            dshape = torch.zeros((3, 8, nop), dtype=torch.float32)
            for i in range(8):
                dshape[0, i, :] = (
                    1 / 8 * s[i, 0] * (1 + s[i, 1] * l2) * (1 + s[i, 2] * l3)
                )
                dshape[1, i, :] = (
                    1 / 8 * (1 + s[i, 0] * l1) * s[i, 1] * (1 + s[i, 2] * l3)
                )
                dshape[2, i, :] = (
                    1 / 8 * (1 + s[i, 0] * l1) * (1 + s[i, 1] * l2) * s[i, 2]
                )
        else:
            raise ValueError("Only P1, P2, and Q1 elements implemented.")
    else:
        raise ValueError("Invalid element dimension.")

    return dshape
