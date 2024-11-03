import torch


def compute_mfga_signal(setup, initial_signal, dtensor):
    """
    Compute the Matrix Formalism Gaussian approximation signal.
    """
    # Extract experiment parameters
    bvalues = setup.gradient["bvalues"]
    directions = setup.gradient["directions"]

    # Sizes
    namplitude = bvalues.shape[0]
    nsequence = bvalues.shape[1]
    ndirection = directions.shape[1]

    # Initialize output arguments
    signal = torch.zeros(namplitude, nsequence, ndirection)
    adc = torch.zeros(nsequence, ndirection)

    for idir in range(ndirection):
        g = directions[:, idir]
        for iseq in range(nsequence):
            # print(bvalues[:, iseq].shape)
            b = bvalues[:, iseq]
            D = torch.matmul(torch.matmul(g.T, dtensor[:, :, iseq]), g)
            adc[iseq, idir] = D
            # print(b)
            signal[:, iseq, idir] = initial_signal * torch.exp(-D * b)
            # print(initial_signal * torch.exp(-D * b))
            # print((initial_signal * torch.exp(-D * b)).shape)

    result = {"signal": signal, "adc": adc}

    return result
