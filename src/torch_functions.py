import torch


def torch_amtam(A, B):
    # A shape: [3, 4, n], B shape: [4, 3, n]
    # PyTorch equivalent of the np.einsum operation
    return torch.einsum("ikz,ljz->ilz", A, B)


def torch_avtam(avx, ama):
    _, ny, nz = ama.shape
    avx_expanded = avx.unsqueeze(1).expand(-1, ny, -1)
    return torch.sum(ama * avx_expanded, dim=0)


def torch_astam(asx, ama):
    nx, ny, nz = ama.shape
    asx_expanded = asx.view(1, 1, nz).expand(nx, ny, -1)
    return ama * asx_expanded


def torch_smamt(smx, ama):
    ny, _, nz = ama.shape
    nk, _ = smx.shape

    amb = torch.zeros((nk, ny, nz), dtype=ama.dtype)
    for row in range(nk):
        amb[row, :, :] = torch_svamt(smx[row, :], ama)

    return amb


def torch_svamt(svx, ama):
    ny, _, nz = ama.shape

    avx = svx.unsqueeze(0).unsqueeze(2)

    avb = ama * avx
    avb = torch.sum(avb, dim=1)
    avb = avb.view(1, ny, nz)

    return avb
