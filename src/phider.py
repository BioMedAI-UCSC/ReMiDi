import torch
from src.shapeder import shapeder


def phider(coord, point, etype):
    """
    Return gradients of basis functions w.r.t. local coordinates (x, y, ...) in PyTorch.
    coord: PyTorch tensor of shape (nod, nos, noe), the local coordinates of the nodes.
    point: PyTorch tensor of shape (nod, nop), the coordinates of the points on the reference element.
    etype: string, the element type ("P0", "P1", "P2", etc.).
    Returns a tuple (dphi, detj, jac):
        dphi: PyTorch tensor of shape (nod, nos, nop, noe), gradients of the shape functions.
        detj: PyTorch tensor of shape (nop, noe), determinants of the Jacobian matrices.
        jac: PyTorch tensor of shape (nod, nod, nop, noe), the Jacobian matrices.
    """
    nod, nop = point.size()
    nos, noe = coord.size()[1], coord.size()[2]

    # Derivatives with respect to the reference coordinates
    dshape = shapeder(point, etype)

    jac = torch.zeros((nod, nod, nop, noe), dtype=coord.dtype)
    detj = torch.zeros((nop, noe), dtype=coord.dtype)
    dphi = torch.zeros((nod, nos, nop, noe), dtype=coord.dtype)

    for poi in range(nop):
        for ele in range(noe):
            tjac = torch.einsum("ik,jk->ij", dshape[:, :, poi], coord[:, :, ele])
            try:
                tjacinv = torch.inverse(tjac)
                tjacdet = torch.det(tjac)
            except RuntimeError:
                # Handle non-invertible matrix case
                tjacinv = torch.zeros_like(tjac)
                tjacdet = torch.tensor(0, dtype=tjac.dtype)

            dphi[:, :, poi, ele] = torch.einsum("ij,jk->ik", tjacinv, dshape[:, :, poi])

            jac[:, :, poi, ele] = tjac
            detj[poi, ele] = torch.abs(tjacdet)

    # Removing the third dimension if it's of size 1 (squeezing if necessary)
    jac = torch.squeeze(jac, axis=2)

    # breakpoint()

    return dphi, detj, jac
