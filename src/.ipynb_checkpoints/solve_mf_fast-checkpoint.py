import torch
import time
from setup_mesh.PGSE import PGSE
from src.sparse_block_diagonal import sparse_block_diagonal
from src.get_volume_mesh import get_volume_mesh
from src.mass_matrixP1_3D import mass_matrixP1_3D

def solve_mf_fast(femesh, setup, lap_eig):
    """
    Compute the solution to the BTPDE using Matrix Formalism, optimized for GPU computation and gradient flow.
    """
    start_time_all = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Clone and move tensors to GPU
    femesh = {k: v.clone().detach().to(device) if isinstance(v, torch.Tensor) else 
              [t.clone().detach().to(device) if isinstance(t, torch.Tensor) else t for t in v] if isinstance(v, list) else v 
              for k, v in femesh.items()}

    initial_density = torch.tensor(setup.pde["initial_density"], device=device)
    qvalues = setup.gradient["qvalues"].clone().detach().to(device)
    bvalues = setup.gradient["bvalues"].clone().detach().to(device)
    sequences = setup.gradient["sequences"]
    directions = setup.gradient["directions"].clone().detach().to(device)
    eigfuncs = lap_eig["funcs"].clone().detach().to(device)
    moments = lap_eig["moments"].clone().detach().to(device)
    T2 = lap_eig["massrelax"].clone().detach().to(device)
    L = torch.diag(lap_eig["values"].clone().detach().to(device))

    # Sizes
    ncompartment = femesh["ncompartment"]
    namplitude, nsequence, ndirection = qvalues.shape[0], len(sequences), directions.shape[1]
    neig = L.shape[0]
    ninterval = setup.mf["ninterval"]

    # Number of points in each compartment
    npoint_cmpts = torch.tensor([x.shape[1] for x in femesh["points"]], device=device)

    # Initialize tensors for results
    signal = torch.zeros((ncompartment, namplitude, nsequence, ndirection), dtype=torch.complex64, device=device)
    signal_allcmpts = torch.zeros((namplitude, nsequence, ndirection), dtype=torch.complex64, device=device)
    itertimes = torch.zeros((namplitude, nsequence, ndirection), device=device)

    # Assemble mass matrices
    M_cmpts = []
    rho_cmpts = []
    for icmpt in range(ncompartment):
        points = femesh["points"][icmpt]
        elements = femesh["elements"][icmpt]
        _, volumes, _ = get_volume_mesh(points, elements)
        M = mass_matrixP1_3D(elements, volumes)
        M_cmpts.append(M)
        initial_cond = torch.full((points.shape[1],), initial_density[icmpt], dtype=torch.complex64, device=device)
        rho_cmpts.append(initial_cond)

    M = sparse_block_diagonal(M_cmpts).to_dense().to(dtype=torch.complex64, device=device)
    rho = torch.cat(rho_cmpts, dim=0).unsqueeze(1).to(device)

    nu0 = torch.matmul(eigfuncs.t().to(dtype=torch.complex64), torch.matmul(M, rho))

    def compute_magnetization(q, seq, g):
        # Ensure g is 2D: (3, ndirection)
        if g.dim() == 1:
            g = g.unsqueeze(1)
        # Ensure moments is 3D: (neig, 3, nmodes)
        if moments.dim() == 2:
            moments = moments.unsqueeze(2)
        
        A = torch.sum(moments * g.unsqueeze(0), dim=1)
        if isinstance(seq, PGSE):
            K = L + T2 + 1j * q * A
            edK = torch.matrix_exp(-seq.delta * K)
            edL = torch.matrix_exp(-(seq.Delta - seq.delta) * (L + T2))
            nu = torch.matmul(edK.conj().t(), torch.matmul(edL, torch.matmul(edK, nu0)))
        else:
            time_intervals = torch.linspace(0, seq.echotime, ninterval + 1, device=device)
            nu = nu0.clone()
            for i in range(ninterval):
                dt = time_intervals[i + 1] - time_intervals[i]
                ft = (seq.call(time_intervals[i + 1]) + seq.call(time_intervals[i])) / 2
                K = L + T2 + 1j * q * ft * A
                nu = torch.matmul(torch.matrix_exp(-dt * K), nu)
        return torch.matmul(eigfuncs.to(dtype=torch.complex64), nu)

    # Compute magnetization for all combinations
    for iamp in range(namplitude):
        for iseq in range(nsequence):
            mag_batch = compute_magnetization(qvalues[iamp, iseq], sequences[iseq], directions)
            mag_cmpts = torch.split(mag_batch, npoint_cmpts.tolist())
            for icmpt, mag_c in enumerate(mag_cmpts):
                signal[icmpt, iamp, iseq] = torch.sum(torch.matmul(M_cmpts[icmpt].to(device), mag_c), dim=0)

    signal_allcmpts = torch.sum(signal, dim=0)

    results_mf = {
        "signal": signal,
        "signal_allcmpts": signal_allcmpts,
        "itertimes": itertimes,
        "time_taken": time.time() - start_time_all,
    }

    return results_mf