import torch


def couple_flux_matrix(femesh, pde, Q_blocks, symmetrical):
    # Extracting information from femesh and pde structures
    point_map = femesh["point_map"]
    facets = femesh["facets"]
    nboundary = femesh["nboundary"]
    npoint_cmpts = [points.shape[1] for points in femesh["points"]]
    npoint = sum(npoint_cmpts)
    cmpt_inds = torch.cumsum(torch.tensor([0] + npoint_cmpts), 0)

    def get_inds(icmpt):
        return slice(cmpt_inds[icmpt].item(), cmpt_inds[icmpt + 1].item())

    # Initialize the global flux matrix Q
    Q = torch.zeros((npoint, npoint))

    for iboundary in range(nboundary):
        cmpts_touch = [
            i for i, facet in enumerate(facets) if facet[iboundary] is not None
        ]
        ntouch = len(cmpts_touch)

        if ntouch == 1:
            cmpt = cmpts_touch[0]
            k = pde["permeability"][iboundary]
            inds = get_inds(cmpt)
            Q[inds, inds] += k * Q_blocks[cmpt][iboundary].to_dense()
        elif ntouch == 2:
            cmpt1, cmpt2 = cmpts_touch
            Q11 = Q_blocks[cmpt1][iboundary]
            Q22 = Q_blocks[cmpt2][iboundary]

            Q12 = torch.zeros((npoint_cmpts[cmpt1], npoint_cmpts[cmpt2]))
            inds1 = torch.unique(facets[cmpt1][iboundary])
            inds2 = torch.unique(facets[cmpt2][iboundary])

            if torch.all(point_map[cmpt1][inds1] == point_map[cmpt2][inds2]):
                indinds1 = torch.arange(len(inds1))
                indinds2 = torch.arange(len(inds2))
            else:
                indinds1, indinds2 = torch.where(
                    point_map[cmpt1][inds1, None] == point_map[cmpt2][None, inds2]
                )

            Q11_dense = Q11.to_dense() if Q11.is_sparse else Q11

            # Initialize Q12 as a dense tensor
            Q12 = torch.zeros((npoint_cmpts[cmpt1], npoint_cmpts[cmpt2]))

            # Perform the assignment with dense tensors
            inds2_actual = inds2[
                indinds2
            ].long()  # Ensure indices are long type for indexing
            inds1_actual = inds1[indinds1].long()
            Q12[:, inds2_actual] = Q11_dense[:, inds1_actual]

            Q21 = Q12.transpose(0, 1)

            if symmetrical:
                c12 = c21 = 1
            else:
                rho1 = pde["initial_density"][cmpt1]
                rho2 = pde["initial_density"][cmpt2]
                c21 = 2 * rho2 / (rho1 + rho2)
                c12 = 2 * rho1 / (rho1 + rho2)

            k1 = c21 * pde["permeability"][iboundary]
            k2 = c12 * pde["permeability"][iboundary]

            inds1 = get_inds(cmpt1)
            inds2 = get_inds(cmpt2)

            Q[inds1, inds1] += k1 * Q11.to_dense()
            Q[inds1, inds2] -= k2 * Q12.to_dense()
            Q[inds2, inds1] -= k1 * Q21.to_dense()
            Q[inds2, inds2] += k2 * Q22.to_dense()

        elif ntouch > 2:
            raise ValueError("Each interface touch only 1 or 2 compartments")
            
    # Updating here on 08/17 for symmetry
    # if symmetrical:
    #     Q = (Q + Q.transpose(0, 1)) / 2

    # Update
    Q_sym = (Q + Q.transpose(0, 1)) / 2

    Q_sparse = Q_sym.to_sparse()

    return Q_sparse

    # Earlier
    # Q_sparse = Q.to_sparse()
    # return Q_sparse
