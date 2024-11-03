import torch


def compute_free_diffusion(bvalues, diffusivity, volumes, initial_density):
    # Take direction average diffusivity from tensor (trace)
    trace_diffusivity = trace_diffusivity = (
        diffusivity.diagonal(dim1=-2, dim2=-1).sum(-1) / 3
    )  # torch.mean(diffusivity.diagonal(dim1=0, dim2=1), dim=0)

    # Initial signal
    S0 = initial_density * volumes

    # Free signal
    expanded_diffusivity = trace_diffusivity.unsqueeze(-1).unsqueeze(-1)
    expanded_bvalues = bvalues.unsqueeze(0)
    signal = S0.unsqueeze(-1).unsqueeze(-1) * torch.exp(
        -expanded_diffusivity * expanded_bvalues
    )
    signal_allcmpts = torch.sum(signal, dim=0)

    # ADC is simply the diffusion coefficient
    adc = trace_diffusivity

    # Total apparent diffusion coefficient weighted by volumes
    adc_allcmpts = torch.sum(volumes * adc) / torch.sum(volumes)

    # Create output structure
    free = {
        "signal": signal,
        "signal_allcmpts": signal_allcmpts,
        "adc": adc,
        "adc_allcmpts": adc_allcmpts,
    }

    return free
