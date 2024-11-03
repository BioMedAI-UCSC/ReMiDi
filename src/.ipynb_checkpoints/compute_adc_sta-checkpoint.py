import torch
import warnings
import time
from src.get_vol_sa import get_vol_sa
from src.get_surface_mesh import get_surface_mesh


def compute_adc_sta(femesh, setup, show_warning=True):
    start_time = time.time()
    # Convert diffusivity values to tensors
    diffusivity_in = torch.tensor(setup.pde["diffusivity_in"], dtype=torch.float32)
    diffusivity_out = torch.tensor(setup.pde["diffusivity_out"], dtype=torch.float32)
    diffusivity_ecs = torch.tensor(setup.pde["diffusivity_ecs"], dtype=torch.float32)
    diffusivity = torch.stack([diffusivity_in, diffusivity_out, diffusivity_ecs])

    initial_density = setup.pde["initial_density"]

    sequences = setup.gradient["sequences"]
    directions = setup.gradient["directions"]

    nsequence = len(sequences)
    ncompartment = femesh["ncompartment"]  # 'in', 'out', 'ecs'
    ndirection = 1 if directions.ndim == 1 else directions.shape[1]

    adc = torch.zeros((ncompartment, nsequence, ndirection), dtype=torch.float32)
    adc_allcmpts = torch.zeros((nsequence, ndirection), dtype=torch.float32)

    volumes, surface_areas = get_vol_sa(femesh)  # Ensure these return tensors

    for icmpt in range(ncompartment):
        for idir in range(ndirection):
            g = directions if directions.ndim == 1 else directions[..., idir]
            g = g.unsqueeze(1) if g.ndim == 1 else g
            # print(g)

            D0 = diffusivity[icmpt] * (g.T @ g).item()

            _, areas, _, normals = get_surface_mesh(
                femesh["points"][icmpt], femesh["facets"][icmpt]
            )  # Ensure these return tensors
            SAu = (g.T @ normals) ** 2 @ areas

            for iseq, seq in enumerate(sequences):
                # diffusion_time_sta = seq.diffusion_time_sta()
                diffusion_time_sta = torch.tensor(
                    seq.diffusion_time_sta(), dtype=torch.float32
                )
                D = D0 * (
                    1
                    - 4
                    / 3
                    * torch.sqrt(D0 / torch.pi)
                    * SAu
                    / volumes[icmpt]
                    * torch.sqrt(diffusion_time_sta)
                )
                adc[icmpt, iseq, idir] = D
                # print(D)

        if show_warning:
            adc_flag = torch.any(adc < 0, dim=(0, 2))
            for iflag in range(nsequence):
                if adc_flag[iflag]:
                    seq_str = f"Sequence {iflag + 1}: "
                    warnings.warn(
                        f"{seq_str}Negative STA ADC, short diffusion time approximation does not hold."
                    )

    weights = initial_density * volumes
    weights /= torch.sum(weights)
    adc_allcmpts = torch.sum(weights[:, None, None] * adc, dim=0)

    return adc, adc_allcmpts, time.time() - start_time
