from src.get_vol_sa import get_vol_sa
from src.compute_adc_sta import compute_adc_sta
from src.calculate_generalized_mean_diffusivity import calculate_generalized_mean_diffusivity
from src.length2eig import length2eig
from src.compute_laplace_eig_diff import compute_laplace_eig_diff
from src.eig2length import eig2length
from src.solve_mf import solve_mf
from src.compute_free_diffusion import compute_free_diffusion
from plot.plot_femesh import plot_femesh

def wrapped_simulator(femesh_all, setup):
    femesh_all_2_split = split_mesh(femesh_all)
    
    neig_max = setup.mf['neig_max']
    volumes, surface_areas = get_vol_sa(femesh_all_2_split)

    mean_diffusivity = calculate_generalized_mean_diffusivity(setup.pde['diffusivity'], volumes)
    eiglim = length2eig(setup.mf['length_scale'], mean_diffusivity)
    lap_eig = compute_laplace_eig_diff(femesh_all_2_split, setup, setup.pde, eiglim, neig_max)
    lap_eig['length_scales'] = eig2length(lap_eig['values'], mean_diffusivity)
    mf_signal = solve_mf(femesh_all_2_split, setup, lap_eig)

    print('signal computed')

    return mf_signal['signal_allcmpts'] / mf_signal['signal_allcmpts'][0, 0, :, 0].view(1, 1, -1, 1)