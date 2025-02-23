import time
import torch
from mesh_setup.PGSE import PGSE
from src.sparse_block_diagonal import sparse_block_diagonal
from src.get_volume_mesh import get_volume_mesh
from src.mass_matrixP1_3D import mass_matrixP1_3D


def solve_mf_gpu(femesh, setup, lap_eig):
    """
    Compute the solution to the BTPDE using Matrix Formalism.
    """
    start_time_all = time.time()

    # ExtraWct parameters
    initial_density = setup.pde["initial_density"]
    qvalues = setup.gradient["qvalues"]
    bvalues = setup.gradient["bvalues"]
    sequences = setup.gradient["sequences"]
    directions = setup.gradient["directions"]
    eigfuncs = lap_eig["funcs"]
    moments = lap_eig["moments"]
    T2 = lap_eig["massrelax"]
    L = torch.diag(lap_eig["values"])

    # Sizes
    ncompartment = femesh["ncompartment"]
    namplitude = len(setup.gradient["values"])
    nsequence = len(sequences)
    ndirection = directions.shape[1]
    neig = len(lap_eig["values"])
    ninterval = setup.mf["ninterval"]

    # Number of points in each compartment
    npoint_cmpts = [x.shape[1] for x in femesh["points"]]

    # Initialize 'magnetization' as a nested list of lists
    magnetization = [
        [
            [[[] for _ in range(ndirection)] for _ in range(nsequence)]
            for _ in range(namplitude)
        ]
        for _ in range(ncompartment)
    ]

    # Initialize 'signal', 'signal_allcmpts', and 'itertimes' as tensors filled with zeros
    signal = torch.zeros(ncompartment, namplitude, nsequence, ndirection, dtype=torch.complex64)
    signal_allcmpts = torch.zeros(namplitude, nsequence, ndirection, dtype=torch.complex64)
    itertimes = torch.zeros(namplitude, nsequence, ndirection)

    # Initializing lists for mass matrices and initial densities
    M_cmpts = []
    rho_cmpts = []

    print("Assembling mass matrices")
    for icmpt in range(ncompartment):
        points = femesh["points"][icmpt]
        elements = femesh["elements"][icmpt]

        # Get volume mesh and mass matrix
        _, volumes, _ = get_volume_mesh(points, elements)
        M = mass_matrixP1_3D(elements, volumes)

        # Add the mass matrix to the list
        M_cmpts.append(M)

        # Create initial conditions (complex values)
        initial_cond = torch.full(
            size=(points.shape[1],),
            fill_value=initial_density[icmpt],
            dtype=torch.complex64,
        )
        rho_cmpts.append(initial_cond)

    # Create sparse block diagonal matrix for mass matrices
    M = sparse_block_diagonal(M_cmpts)
    rho = torch.cat(rho_cmpts, dim=0).unsqueeze(1)

    # Initial magnetization in Laplace basis
    nu0 = torch.matmul(
        eigfuncs.t().to(dtype=torch.complex64),
        torch.matmul(M.to_dense().to(dtype=torch.complex64), rho)
    )

    allinds = torch.tensor([namplitude, nsequence, ndirection])

    for iall in range(torch.prod(allinds)):
        # Measure iteration time
        itertime = time.time()

        # Convert linear index to subscript indices
        iamp = iall % namplitude
        iseq = (iall // namplitude) % nsequence
        idir = ((iall // namplitude) // nsequence) % ndirection

        # Experiment parameters
        q = qvalues[iamp, iseq]
        b = bvalues[iamp, iseq]
        seq = sequences[iseq]
        g = directions[:, idir]

        # Create time intervals for time profile approximation
        time_intervals = torch.linspace(0, seq.echotime, ninterval + 1)

        # BT operator matrix components
        A = torch.einsum('ijk,k->ij', moments, g)

        # Constant BT operator in Laplace basis
        K = torch.add(L, torch.add(T2, 1j * q * A))

        # Compute matrix exponentials
        edK = torch.matrix_exp(-seq.delta * K)
        edL = torch.matrix_exp(-(seq.Delta - seq.delta) * (L + T2))

        # Final magnetization in Laplace basis
        edK_nu0 = torch.matmul(edK, nu0)
        nu = torch.matmul(edK.conj().t(), torch.matmul(edL.to(dtype=torch.complex64), edK_nu0))

        # Final magnetization coefficients in finite element nodal basis
        mag = torch.matmul(eigfuncs.to(dtype=torch.complex64), nu)

        # Split magnetization into compartments
        mag_cmpts = torch.split(mag, npoint_cmpts)

        # Store magnetization
        for icmpt, mag_c in enumerate(mag_cmpts):
            magnetization[icmpt][iamp][iseq][idir] = mag_c.squeeze()

        # Convert mass matrices to complex tensors
        M_cmpts_cmplx = [M.to_dense().to(dtype=torch.complex64) for M in M_cmpts]

        # Compute signal
        signal[:, iamp, iseq, idir] = torch.stack(
            [torch.sum(torch.matmul(M, m)) for M, m in zip(M_cmpts_cmplx, mag_cmpts)]
        )

        # Store iteration time
        itertimes[iamp, iseq, idir] = time.time() - itertime

    # Sum signal across compartments
    signal_allcmpts = signal.sum(dim=0)

    results_mf = {
        "magnetization": magnetization,
        "signal": signal,
        "signal_allcmpts": signal_allcmpts,
        "itertimes": itertimes,
        "time_taken": time.time() - start_time_all,
    }

    return results_mf
