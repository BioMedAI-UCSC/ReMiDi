import time
import torch
from mesh_setup.PGSE import PGSE
from src.sparse_block_diagonal import sparse_block_diagonal
from src.get_volume_mesh import get_volume_mesh
from src.mass_matrixP1_3D import mass_matrixP1_3D


def solve_mf(femesh, setup, lap_eig):
    """
    Compute the solution to the BTPDE using Matrix Formalism.
    """
    start_time_all = time.time()

    femesh = femesh

    # Extract parameters
    initial_density = setup.pde["initial_density"]
    qvalues = setup.gradient["qvalues"]
    bvalues = setup.gradient["bvalues"]
    sequences = setup.gradient["sequences"]
    directions = setup.gradient["directions"]
    eigfuncs = lap_eig["funcs"]
    moments = lap_eig["moments"]
    T2 = lap_eig["massrelax"]
    # L = torch.diag(lap_eig['values'].squeeze(1))
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

    # For heterogeneous data or variable sizes, initialize 'magnetization' as a nested list of lists
    magnetization = [
        [
            [[[] for _ in range(ndirection)] for _ in range(nsequence)]
            for _ in range(namplitude)
        ]
        for _ in range(ncompartment)
    ]

    # For homogeneous numeric data, initialize 'signal', 'signal_allcmpts', and 'itertimes' as tensors filled with zeros
    signal = torch.zeros(ncompartment, namplitude, nsequence, ndirection).to(
        dtype=torch.complex64
    )
    signal_allcmpts = torch.zeros(namplitude, nsequence, ndirection).to(
        dtype=torch.complex64
    )
    itertimes = torch.zeros(namplitude, nsequence, ndirection)

    # Initializing lists for mass matrices and initial densities
    M_cmpts = []
    rho_cmpts = []

    print("Assembling mass matrices")
    for icmpt in range(ncompartment):
        # Finite elements
        # points = torch.tensor(femesh['points'][icmpt])
        # elements = torch.tensor(femesh['elements'][icmpt])
        points = femesh["points"][icmpt]
        elements = femesh["elements"][icmpt]

        # Assuming get_volume_mesh() returns volumes
        _, volumes, _ = get_volume_mesh(points, elements)

        # Assuming mass_matrixP1_3D() returns the mass matrix
        M = mass_matrixP1_3D(elements, volumes)

        # Adding the mass matrix to the list
        M_cmpts.append(M)

        # Creating initial conditions (enforcing complex values)
        initial_cond = torch.full(
            size=(points.shape[1],),
            fill_value=initial_density[icmpt],
            dtype=torch.complex64,
        )
        rho_cmpts.append(initial_cond)
        # breakpoint()

    M = sparse_block_diagonal(M_cmpts)
    rho = torch.cat(rho_cmpts, dim=0).unsqueeze(1)

    nu0 = lap_eig["funcs"].t().to(dtype=torch.complex64) @ (
        M.to_dense().to(dtype=torch.complex64) @ rho.to_dense()
    )

    allinds = torch.tensor([namplitude, nsequence, ndirection])

    for iall in range(torch.prod(allinds)):
        # for iall in range(5):
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

        # breakpoint()
        # Create time intervals for time profile approximation
        time_intervals = torch.linspace(0, seq.echotime, ninterval + 1)

        # Display state of iterations
        # Uncomment for more info
        # print(f"Computing MF magnetization using {neig} eigenvalues\n"
        #       f"  Direction {idir + 1} of {ndirection}: g = [{g[0]:.2f}; {g[1]:.2f}; {g[2]:.2f}]\n"
        #       f"  Sequence  {iseq + 1} of {nsequence}: f = {seq}\n"
        #       f"  Amplitude {iamp + 1} of {namplitude}: q = {q}, b = {b}")

        # Components of BT operator matrix
        # breakpoint()
        A = torch.sum(moments * g.unsqueeze(0).unsqueeze(0), dim=2)
        # print('A')
        # print(A)
        # breakpoint()

        # Compute final magnetization (in Laplace basis)
        if isinstance(seq, PGSE):  # Assuming PGSE is a class or type you've defined
            # Constant BT operator in Laplace basis
            # q = torch.tensor(0.0001)
            # print("iall", iall)
            # q = 1
            K = L + T2 + 1j * q * A
            # print('K')
            # print(K)
            edK = torch.matrix_exp(-seq.delta * K)
            # print('edK')
            # print(edK)
            edL = torch.matrix_exp(-(seq.Delta - seq.delta) * (L + T2))
            # print('edL')
            # print(edL)
            # breakpoint()
            # Laplace coefficients of final magnetization
            # if iall >= 21:
            # breakpoint()
            # nu = torch.mm(edK.t(), torch.mm(edL.to(dtype=torch.complex64), torch.mm(edK, nu0)))
            # For complex matrices matlab makes use of complex transpose (conjugate + transpose)
            nu = edK.conj().t() @ (edL.to(dtype=torch.complex64) @ (edK @ nu0))
            # print('nu')
            # print(nu)
            # breakpoint()
        else:
            # BT operator in Laplace basis for a given time profile value
            def K(ft):
                return L + T2 + 1j * q * ft * A

            # Transform Laplace coefficients using piecewise constant approximation of time profile
            nu = nu0.clone()
            for i in range(ninterval):
                dt = time_intervals[i + 1] - time_intervals[i]
                ft = (seq.call(time_intervals[i + 1]) + seq.call(time_intervals[i])) / 2

                # Laplace coefficients of magnetization at end of interval
                nu = torch.mm(torch.matrix_exp(-dt * K(ft)), nu)

        # Final magnetization coefficients in finite element nodal basis
        # breakpoint()
        mag = eigfuncs.to(dtype=torch.complex64) @ nu
        # print('mag')
        # print(mag)
        # print(mag)

        # Split magnetization into compartments
        mag_cmpts = torch.split(mag, npoint_cmpts)

        # Here, you might need to adjust storage of magnetization according to your data structure
        # For signal and itertimes, ensure they're initialized as tensors or lists as needed
        for icmpt, mag_c in enumerate(mag_cmpts):
            # breakpoint()
            magnetization[icmpt][iamp][iseq][
                idir
            ] = mag_c.squeeze()  # Adjust dimensions as necessary

        M_cmpts = [M.to_dense().to(dtype=torch.complex64) for M in M_cmpts]
        # breakpoint()
        for icmpt in range(ncompartment):
            signal[icmpt, iamp, iseq, idir] = [
                torch.sum(torch.mm(M, m)) for M, m in zip(M_cmpts, mag_cmpts)
            ][icmpt]
            # print([torch.sum(torch.mm(M, m)) for M, m in zip(M_cmpts, mag_cmpts)][icmpt])
        # print([torch.sum(torch.mm(M, m)) for M, m in zip(M_cmpts, mag_cmpts)])
        # print(signal[:, iamp, iseq, idir])
        # Store timing
        itertimes[iamp, iseq, idir] = time.time() - itertime

    # breakpoint()
    signal_allcmpts = signal.sum(dim=0)  # Sum over the first axis

    # For flattening
    signal_allcmpts_flattened = signal_allcmpts.flatten()

    results_mf = {
        "magnetization": magnetization,
        "signal": signal,
        "signal_allcmpts": signal_allcmpts,
        "itertimes": itertimes,
        "time_taken": time.time() - start_time_all,
    }

    return results_mf
