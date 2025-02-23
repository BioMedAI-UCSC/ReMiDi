import torch as tch
import gc
import sys
import os as ops
import argparse
import json
import importlib
import numpy as np
import torch.optim as optim
import time as time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
import cv2 as cv2
from natsort import natsorted
from matplotlib.backends.backend_agg import FigureCanvasAgg
import roma

sys.path.append("SAE")

from models.utils.init_weights import init_weights

# sys.path.append('../DMRI_Code')
from src.scale_mesh import scale_mesh
from src.read_tetgen import read_tetgen
from src.deform_domain import deform_domain
from src.split_mesh import split_mesh
from src.get_vol_sa import get_vol_sa
from src.compute_adc_sta import compute_adc_sta
from src.calculate_generalized_mean_diffusivity import (
    calculate_generalized_mean_diffusivity,
)
from src.length2eig import length2eig
from src.compute_laplace_eig_diff import compute_laplace_eig_diff
from src.eig2length import eig2length
from src.solve_mf import solve_mf
from src.compute_free_diffusion import compute_free_diffusion
from plot.plot_femesh import plot_femesh
from mesh_setup import Setup1AxonAnalytical_LowRes as Setup1AxonAnalytical_LowRes
from mesh_setup.update_pde import update_pde
from mesh_setup.prepare_pde import prepare_pde
from mesh_setup.prepare_experiments import prepare_experiments
from plot.plot_femesh_plotly_2 import plot_femesh_plotly_2
from plot.plot_point_cloud_plotly import plot_point_cloud_plotly
from src.get_volume_mesh import get_volume_mesh
from src.rotate_point_cloud import rotate_point_cloud
from src.get_rotations import get_rotations

# Importing scaling
from src.ellipsoidal_scale_mesh import ellipsoidal_scale_mesh


def load_config(config_path):
    with open(config_path, "r") as config_file:
        return json.load(config_file)


def cleanup():
    gc.collect()
    tch.cuda.empty_cache()


def main(config):

    gc.collect()
    tch.cuda.empty_cache()
    tch.device(config["device"])

    # sys.path.append(config['base_folder_path'])

    # Import cell setup and pde properties
    importlib.reload(Setup1AxonAnalytical_LowRes)
    setup = Setup1AxonAnalytical_LowRes.Setup1AxonAnalytical_LowRes()
    update_pde(setup)
    # Call prepare_pde with this setup instance
    U_pde = prepare_pde(setup)
    # Inspect the output
    print("Updated PDE Parameters:")
    for key, value in U_pde.items():
        print(f"{key}: {value}")

    # Prepare b-values and pde
    prepare_experiments(setup)

    def load_model_from_checkpoint(opt, job_id):
        # Dynamically import the model class
        exec("from models." + opt["model_type"] + " import " + opt["model_type"])
        model_class = eval(opt["model_type"])

        with open(
            f"{config['checkpoint_base_path']}/" + str(job_id) + "/infos.json", "r"
        ) as outfile:
            opt = json.load(outfile)
        # Initialize the model
        opt["model"] = model_class(opt).to(opt["device"])
        opt["model"].apply(init_weights)

        # Path to the checkpoint
        # best_checkpoint_filename = os.path.join('/media/DATA_18_TB_1/shri/SAE-main/checkpoints', str(job_id), "last.ckpt")
        best_checkpoint_filename = ops.path.join(
            config["checkpoint_base_path"], str(job_id), "last.ckpt"
        )
        print("Loading checkpoint:", best_checkpoint_filename)

        # Load the model from the checkpoint
        opt["model"] = model_class.load_from_checkpoint(
            best_checkpoint_filename, opt=opt
        ).to(opt["device"])

        return opt["model"]

    # Example usage
    opt = {
        "model_type": "LearnedPooling",  # Replace with your model type
        "device": config["device"],
        # Add other options required for your model
    }
    job_id = config["checkpoint_name"]  # Replace with your job ID

    # Load the model
    model = load_model_from_checkpoint(opt, job_id)

    opt["nb_freq"] = config["num_fequencies"]
    opt["evecs"] = (
        tch.from_numpy(np.load(config["path_evecs"]))
        .float()
        .to(opt["device"])[:, : opt["nb_freq"]]
    )

    # Set model to eval mode so weights don't update
    model.eval()

    # Create a reference signal for reconstruction
    filename = config["main_mesh_file"]
    femesh_all_2 = read_tetgen(filename)
    # Deform mesh to deviate for reference
    femesh_all_2["points"] = deform_domain(
        femesh_all_2["points"],
        tch.tensor([config["deformation_bend"], config["deformation_twist"]]),
    )

    # Adding scaling to the mesh to make more examples
    # scaled_mesh = ellipsoidal_scale_mesh(femesh_all_2, tch.tensor([config['scale_x'], config['scale_y'], config['scale_z']]))

    rotation_matrices = get_rotations()

    # Rotating mesh for reference
    rotated_mesh = rotate_point_cloud(
        femesh_all_2, rotation_matrices, config["rotation_inx"]
    )

    # Run a forward pass to get the reference signal
    femesh_all_2_split = split_mesh(rotated_mesh)
    # plot_femesh(femesh_all_2_split)

    neig_max = setup.mf["neig_max"]
    volumes, surface_areas = get_vol_sa(femesh_all_2_split)
    # free = compute_free_diffusion(setup.gradient['bvalues'], setup.pde['diffusivity'], volumes, setup.pde['initial_density'])
    mean_diffusivity = calculate_generalized_mean_diffusivity(
        setup.pde["diffusivity"], volumes
    )
    eiglim = length2eig(setup.mf["length_scale"], mean_diffusivity)
    lap_eig = compute_laplace_eig_diff(
        femesh_all_2_split, setup, setup.pde, eiglim, neig_max
    )
    lap_eig["length_scales"] = eig2length(lap_eig["values"], mean_diffusivity)
    mf_signal = solve_mf(femesh_all_2_split, setup, lap_eig)

    # Save the reference signal into a temp location for reconstruction
    mf_signal_orig = mf_signal

    # Minimum value for clamping
    min_value = 1e-6
    # mf_signal_orig['signal_allcmpts'] = (tch.abs(mf_signal_orig['signal_allcmpts']) / tch.abs(mf_signal['signal_allcmpts'][..., 0, 0, 0])) * 100
    # Using view for getting consistent signal shape
    # Using clamping to prevent divide by zero
    mf_signal_orig["signal_allcmpts"] = (
        tch.abs(mf_signal_orig["signal_allcmpts"])
        / tch.abs(mf_signal["signal_allcmpts"][0, :, 0].view(1, -1, 1)).clamp(
            min=min_value
        )
    ) * 100

    # Save reference mesh and point cloud for diagnostic and result plots
    # 3D Mesh
    plot_femesh_plotly_2(
        rotated_mesh,
        setup.pde["compartments"],
        0,
        f"reference_cylinder_{config['deformation_bend']}_{config['deformation_twist']}_r{config['rotation_inx']}_h".replace(
            ".", "_"
        ),
        f"reference_cylinder_{config['deformation_bend']}_{config['deformation_twist']}_r{config['rotation_inx']}_p".replace(
            ".", "_"
        ),
    )
    # Point cloud
    plot_point_cloud_plotly(
        rotated_mesh["points"],
        0,
        f"reference_cylinder_{config['deformation_bend']}_{config['deformation_twist']}_r{config['rotation_inx']}_pc_h".replace(
            ".", "_"
        ),
        f"reference_cylinder_{config['deformation_bend']}_{config['deformation_twist']}_r{config['rotation_inx']}_pc_p".replace(
            ".", "_"
        ),
    )

    # Create logging directory
    ops.makedirs(config["log_directory"], exist_ok=True)

    # Saving complete reference mesh file
    tch.save(
        rotated_mesh,
        f"{config['log_directory']}/reference_cylinder_{config['deformation_bend']}_{config['deformation_twist']}_r{config['rotation_inx']}_mesh".replace(
            ".", "_"
        )
        + ".pth",
    )

    # filename = "1low_res_sphere2_dir/1low_res_sphere2_no_ecs0.5_refinement8_mesh.1"
    filename = config["main_mesh_file"]
    femesh_all_2 = read_tetgen(filename)

    v = femesh_all_2["points"]
    # v=deform_domain(transposed,[0.10,0.10])
    spectral_coefficients = tch.einsum(
        "ij, ki->jk", opt["evecs"].to(config["device"]), v.to(config["device"])
    )
    # spectral_coefficients

    latents = model.enc(
        spectral_coefficients[: opt["nb_freq"]].unsqueeze(0).to(opt["device"])
    )
    # print(latents)

    latents = latents.detach()
    # print(latents)

    # Save original latents
    tch.save(latents, f"{config['log_directory']}/latents_start.pth")

    latents = latents.clone().to(config["device"])

    update_mask = tch.zeros_like(latents).requires_grad_(True)
    # print(update_mask)

    loss_overtime = []

    optimizer = optim.AdamW([update_mask], lr=config["base_lr"], weight_decay=0.0)

    scheduler = tch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=1, eta_min=config["min_lr"]
    )

    # Initialize arrays to store iteration numbers, gradient norms, and parameter update norms
    iterations = []
    gradient_norms = []
    param_update_norms = []
    latent_points_history = []

    # Initialize lists to store RMSD values
    rmsd_from_original = []
    rmsd_from_previous = []
    rmsd_from_reference = []

    # Store the original mesh points
    original_mesh_points = femesh_all_2["points"].clone().detach()
    previous_mesh_points = original_mesh_points.clone()

    # reference_mesh_points = deform_domain(femesh_all_2['points'].clone().detach(), tch.tensor([config['deformation_bend'], config['deformation_twist']]))
    reference_mesh_points = rotated_mesh["points"].clone().detach()

    # Initialize lists to store values
    volumes_history = []
    surface_areas_history = []
    mean_diffusivity_history = []

    # Initialize lists to store RMSD values for spectral coefficients
    spectral_rmsd_from_original = []
    spectral_rmsd_from_previous = []

    # Store the original spectral coefficients
    original_spectral = None
    previous_spectral = None

    # Save all meshes as they optimize over iterations
    all_meshes_generated = []

    # Save all latents as they optimize over iterations
    all_latents_generated = []

    # Initialize dictionary to store gradients
    gradients = {}

    # Meshes over time
    meshes_overtime = []

    def save_grad(name):
        def hook(grad):
            gradients[name] = grad

        return hook

    # Function to recursively register hooks on all tensors
    def register_hooks(tensor, name=""):
        if isinstance(tensor, tch.Tensor):
            tensor.register_hook(save_grad(name))
        elif isinstance(tensor, (list, tuple)):
            for idx, t in enumerate(tensor):
                register_hooks(t, f"{name}_{idx}")
        elif isinstance(tensor, dict):
            for key, t in tensor.items():
                register_hooks(t, f"{name}_{key}")

    # Register hooks on the initial tensor
    register_hooks(update_mask, "update_mask")

    def print_grad_norm(tensor, name):
        if tensor.grad is not None:
            print(f"Gradient norm for {name}: {tensor.grad.norm().item()}")

    optimization_success = True

    for iter in range(0, config["max_iters"]):
        time_begin = time.time()
        optimizer.zero_grad()  # Reset gradients

        latent_points = latents + update_mask

        # Save updated latents for each iteration
        all_latents_generated.append(latent_points.detach())

        # Store the current latent points
        latent_points_history.append(latent_points.detach().cpu().numpy().flatten())

        # Decode latents back to spectral coefficients
        spectral = model.dec(latent_points)

        if iter == 0:
            # Store the original spectral coefficients
            original_spectral = model.dec(latent_points).detach()
            previous_spectral = original_spectral.clone()

        # Calculate RMSD from original spectral coefficients
        rmsd_spectral_orig = tch.sqrt(tch.mean((spectral - original_spectral) ** 2))
        spectral_rmsd_from_original.append(rmsd_spectral_orig.item())

        # Calculate RMSD from previous iteration's spectral coefficients
        if iter > 0:
            rmsd_spectral_prev = tch.sqrt(tch.mean((spectral - previous_spectral) ** 2))
            spectral_rmsd_from_previous.append(rmsd_spectral_prev.item())

        # Update previous_spectral for next iteration
        previous_spectral = spectral.clone()

        # spatial = tch.matmul(opt['evecs'], spectral)
        spatial = tch.einsum("ij, ijk->ik", opt["evecs"], spectral)

        femesh_all_2["points"] = tch.einsum("ij->ji", spatial.squeeze(0))

        # Rotate the initial mesh
        femesh_all_2_r = rotate_point_cloud(
            femesh_all_2, rotation_matrices, config["rotation_inx"]
        )

        # Calculate RMSD from original mesh
        # Using detached points for this task
        current_mesh_points = femesh_all_2_r["points"].clone().detach()

        # Save the updated mesh before staring forward pass of the simulator
        all_meshes_generated.append(current_mesh_points)

        # Save rmsd values for meshes overtime
        rmsd_orig = tch.sqrt(
            tch.mean((current_mesh_points - original_mesh_points) ** 2)
        )
        rmsd_ref = tch.sqrt(
            tch.mean((current_mesh_points - reference_mesh_points) ** 2)
        )
        rmsd_from_original.append(rmsd_orig.item())
        rmsd_from_reference.append(rmsd_ref.item())
        meshes_overtime.append(current_mesh_points)

        # Calculate RMSD from previous iteration's mesh
        if iter > 0:
            rmsd_prev = tch.sqrt(
                tch.mean((current_mesh_points - previous_mesh_points) ** 2)
            )
            rmsd_from_previous.append(rmsd_prev.item())

        # Update previous_mesh_points for next iteration
        previous_mesh_points = current_mesh_points.clone()

        print("Started forward simulation")
        femesh_all_2_split = split_mesh(femesh_all_2_r)

        # if iter % 10 == 0 or iter == 499:
        #     plot_femesh(femesh_all_2_split)
        # if iter % 5 == 0 or iter == 499:
        #     plot_femesh_plotly_2(femesh_all_2, setup.pde['compartments'], iter, f"{config['experiment_name']}_h", f"{config['experiment_name']}_p", which='h')
        #     plot_point_cloud_plotly(femesh_all_2['points'], iter, f"{config['experiment_name']}_pc_h", f"{config['experiment_name']}_pc_p", which='h')

        neig_max = setup.mf["neig_max"]
        volumes, surface_areas = get_vol_sa(femesh_all_2_split)

        # Individual tet volumes
        _, tet_volumes, _ = get_volume_mesh(
            femesh_all_2_split["points"][0], femesh_all_2_split["elements"][0]
        )

        volumes_history.append(volumes[0])  # Store mean volume
        surface_areas_history.append(surface_areas[0])  # Store mean surface area

        mean_diffusivity = calculate_generalized_mean_diffusivity(
            setup.pde["diffusivity"], volumes
        )
        mean_diffusivity_history.append(mean_diffusivity)

        eiglim = length2eig(setup.mf["length_scale"], mean_diffusivity)

        try:
            lap_eig = compute_laplace_eig_diff(
                femesh_all_2_split, setup, setup.pde, eiglim, neig_max
            )
        except RuntimeError as e:
            optimization_success = False
            print(
                "Error in forward eigen decomposition (Non-hemritian encountered), stopping optimization here:",
                str(e),
            )
            tch.save(current_mesh_points, "mesh_fails_hermitian_property.pth")
            break

        lap_eig["length_scales"] = eig2length(lap_eig["values"], mean_diffusivity)
        mf_signal = solve_mf(femesh_all_2_split, setup, lap_eig)

        if iter % 10 == 0 or iter == config["max_iters"] - 1:
            plot_femesh_plotly_2(
                femesh_all_2_r,
                setup.pde["compartments"],
                iter,
                f"{config['experiment_name']}_h",
                f"{config['experiment_name']}_p",
                which="p",
            )
            plot_point_cloud_plotly(
                femesh_all_2_r["points"],
                iter,
                f"{config['experiment_name']}_pc_h",
                f"{config['experiment_name']}_pc_p",
                which="p",
            )

            plot_femesh_plotly_2(
                femesh_all_2_r,
                setup.pde["compartments"],
                iter,
                f"{config['experiment_name']}_h",
                f"{config['experiment_name']}_p",
                which="h",
            )
            plot_point_cloud_plotly(
                femesh_all_2_r["points"],
                iter,
                f"{config['experiment_name']}_pc_h",
                f"{config['experiment_name']}_pc_p",
                which="h",
            )

        # loss based on difference
        # dif1 = tch.abs(tch.abs(mf_signal['signal_allcmpts']) - tch.abs(mf_signal_orig['signal_allcmpts']))

        # Rewriting computations for clarity
        node = (
            tch.abs(
                tch.abs(
                    tch.abs(mf_signal["signal_allcmpts"])
                    / tch.abs(mf_signal["signal_allcmpts"][0, :, 0])
                    .view(1, -1, 1)
                    .clamp(min=min_value)
                )
            )
            * 100
        )

        # Normalized signal difference
        # Adding clamping to prevent divide by zero
        # dif1 = tch.abs(tch.abs(tch.abs(mf_signal['signal_allcmpts']) / tch.abs(mf_signal['signal_allcmpts'][0, :, 0].view(1, -1, 1)).clamp(min=min_value)) * 100) - (tch.abs(mf_signal_orig['signal_allcmpts']))
        dif1 = node - (tch.abs(mf_signal_orig["signal_allcmpts"]))
        dif2 = tch.sum(1 / tet_volumes)
        # print(tch.allclose(mf_signal['signal_allcmpts'], mf_signal_orig['signal_allcmpts'], rtol=1e-6, atol=1e-8))
        # if tch.allclose(mf_signal['signal_allcmpts'], mf_signal_orig['signal_allcmpts'], rtol=1e-5, atol=1e-7):
        #     print("all close")
        #     break

        # Momentum regularization term
        if update_mask.grad is not None:
            grad_norm = update_mask.grad.norm()
            if grad_norm <= 1e-3:
                momentum_term = (grad_norm - config["momentum"]) ** 2
            else:
                momentum_term = 0
        else:
            momentum_term = 0

        if config["inv_volume"]:
            loss = tch.mean(dif1**2) + dif2
        elif config["momentum"] > 0:
            loss = (config["loss_multiplier"] * tch.mean(dif1**2)) + momentum_term
        else:
            loss = tch.mean(dif1**2)
        loss_overtime.append(loss.item())

        print("Signal loss computed, beginning backward now")
        try:
            loss.backward()
            print("Backward pass successful.")

            # Print gradient norms
            print_grad_norm(update_mask, "update_mask")
            grad_norm = update_mask.grad.norm().item()
            gradient_norms.append(grad_norm)

        except RuntimeError as e:
            print("Error in backward pass:", str(e))

        print(
            f"Iteration {iter}: Loss = {loss.item()}, Learning Rate = {scheduler.get_last_lr()[0]}, Time taken = {time.time() - time_begin}"
        )

        # Log parameter updates
        with tch.no_grad():
            if update_mask.grad is not None:
                param_update_norm = tch.norm(
                    update_mask.grad * scheduler.get_last_lr()[0]
                )
                param_update_norms.append(param_update_norm.item())
                print(f"Parameter update norm: {param_update_norm.item()}")

        if config["gradient_norm"]:
            tch.nn.utils.clip_grad_norm_(update_mask, max_norm=1)
        else:
            tch.nn.utils.clip_grad_value_(update_mask, config["gradient_clip_value"])
        optimizer.step()
        scheduler.step()

    full_range = config["max_iters"]

    if not optimization_success:
        full_range = iter

    tch.save(
        gradient_norms[:full_range],
        f"{config['log_directory']}/gradient_norms_overtime.pth",
    )
    tch.save(
        param_update_norms[:full_range],
        f"{config['log_directory']}/param_update_norms_overtime.pth",
    )
    tch.save(
        rmsd_from_reference[:full_range],
        f"{config['log_directory']}/rmsd_from_reference_overtime.pth",
    )
    tch.save(
        spectral_rmsd_from_original[:full_range],
        f"{config['log_directory']}/spectral_rmsd_from_original_overtime.pth",
    )
    tch.save(
        spectral_rmsd_from_previous[:full_range],
        f"{config['log_directory']}/spectral_rmsd_from_previous_overtime.pth",
    )
    tch.save(
        rmsd_from_reference[:full_range],
        f"{config['log_directory']}/rmsd_from_reference_overtime.pth",
    )
    tch.save(
        rmsd_from_previous[:full_range],
        f"{config['log_directory']}/rmsd_from_previous_overtime.pth",
    )
    tch.save(
        spectral_rmsd_from_original[:full_range],
        f"{config['log_directory']}/spectral_rmsd_from_original_overtime.pth",
    )
    tch.save(loss_overtime[:full_range], f"{config['log_directory']}/loss_overtime.pth")
    tch.save(
        volumes_history[:full_range], f"{config['log_directory']}/volumes_history.pth"
    )
    tch.save(
        meshes_overtime[:full_range], f"{config['log_directory']}/meshes_history.pth"
    )
    tch.save(
        all_latents_generated[:full_range],
        f"{config['log_directory']}/all_latents_generated.pth",
    )

    iterations = [i for i in range(0, full_range)]
    iterations_np = np.array(iterations[:full_range])
    gradient_norms_np = np.array(gradient_norms[:full_range])
    param_update_norms_np = np.array(param_update_norms[:full_range])
    rmsd_from_reference_np = np.array(rmsd_from_reference[:full_range])

    spectral_rmsd_from_original_np = np.array(spectral_rmsd_from_original[:full_range])
    spectral_rmsd_from_previous_np = np.array(spectral_rmsd_from_previous[:full_range])

    rmsd_from_original_np = np.array(rmsd_from_original[:full_range])
    rmsd_from_previous_np = np.array(rmsd_from_previous[:full_range])

    latent_points_history_np = np.array(latent_points_history[:full_range])

    loss_overtime_np = np.array(loss_overtime[:full_range])

    volumes_history_np = np.array([x.cpu().detach() for x in volumes_history])

    def update(frame):
        fig.clear()
        fig.suptitle("Optimization Progress Over Iterations", fontsize=16)

        gs = fig.add_gridspec(4, 3)

        # Plot loss overtime
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(iterations_np[: frame + 1], np.log(loss_overtime_np[: frame + 1]))
        ax1.set_title("Loss Overtime")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Log MSE")

        # Plot gradient norms
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(iterations_np[: frame + 1], np.log(gradient_norms_np[: frame + 1]))
        ax2.set_title("Gradient Norm")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Log Gradient Norm")

        # Plot parameter update norms
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(iterations_np[: frame + 1], np.log(param_update_norms_np[: frame + 1]))
        ax3.set_title("Parameter Update Norm")
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Log Parameter Update Norm")

        # Plot latent dimensions over time
        ax4 = fig.add_subplot(gs[1, 0])
        latent_dim = latent_points_history_np.shape[1]
        for i in range(latent_dim):
            ax4.plot(latent_points_history_np[: frame + 1, i], label=f"Dim {i+1}")
        ax4.set_title("Latent Dimensions Over Time")
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("Latent Value")
        ax4.legend()

        # Plot RMSD from original mesh
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(rmsd_from_original_np[: frame + 1])
        ax5.set_title("RMSD from Original Mesh")
        ax5.set_xlabel("Iteration")
        ax5.set_ylabel("RMSD")

        # Plot RMSD from previous iteration
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(rmsd_from_previous_np[: frame + 1])
        ax6.set_title("RMSD from Previous Iteration")
        ax6.set_xlabel("Iteration")
        ax6.set_ylabel("RMSD")

        # Plot spectral RMSD from original
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.plot(spectral_rmsd_from_original_np[: frame + 1])
        ax7.set_title("Spectral RMSD from Original")
        ax7.set_xlabel("Iteration")
        ax7.set_ylabel("Spectral RMSD")

        # Plot spectral RMSD from previous
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.plot(spectral_rmsd_from_previous_np[: frame + 1])
        ax8.set_title("Spectral RMSD from Previous")
        ax8.set_xlabel("Iteration")
        ax8.set_ylabel("Spectral RMSD")

        # Plot RMSD from reference mesh
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.plot(rmsd_from_reference_np[: frame + 1])
        ax9.set_title("RMSD from Reference Mesh")
        ax9.set_xlabel("Iteration")
        ax9.set_ylabel("RMSD")

        # Plot latent space trajectory
        ax10 = fig.add_subplot(gs[3, 0:2], projection="3d")
        if frame > 0:
            latent_dim = latent_points_history_np.shape[1]
            if latent_dim == 2:
                scatter = ax10.scatter(
                    latent_points_history_np[: frame + 1, 0],
                    latent_points_history_np[: frame + 1, 1],
                    np.zeros(frame + 1),
                    c=range(frame + 1),
                    cmap="viridis",
                )
                ax10.set_title("Latent Space Trajectory (2D)")
                ax10.set_xlabel("Latent Dimension 1")
                ax10.set_ylabel("Latent Dimension 2")
                ax10.set_zlabel("N/A")
            elif latent_dim == 3:
                scatter = ax10.scatter(
                    latent_points_history_np[: frame + 1, 0],
                    latent_points_history_np[: frame + 1, 1],
                    latent_points_history_np[: frame + 1, 2],
                    c=range(frame + 1),
                    cmap="viridis",
                )
                ax10.set_title("Latent Space Trajectory (3D)")
                ax10.set_xlabel("Latent Dimension 1")
                ax10.set_ylabel("Latent Dimension 2")
                ax10.set_zlabel("Latent Dimension 3")
            else:
                n_components = min(3, frame + 1, latent_dim)
                if n_components < 3:
                    ax10.set_title(
                        f"Latent Space Trajectory (PCA)\nNot enough data points yet (n={frame+1})"
                    )
                    scatter = ax10.scatter([], [], [], c=[], cmap="viridis")
                else:
                    pca = PCA(n_components=n_components)
                    latent_points_pca = pca.fit_transform(
                        latent_points_history_np[: frame + 1]
                    )
                    if n_components == 3:
                        scatter = ax10.scatter(
                            latent_points_pca[:, 0],
                            latent_points_pca[:, 1],
                            latent_points_pca[:, 2],
                            c=range(frame + 1),
                            cmap="viridis",
                        )
                    else:
                        scatter = ax10.scatter(
                            latent_points_pca[:, 0],
                            latent_points_pca[:, 1],
                            np.zeros(frame + 1),
                            c=range(frame + 1),
                            cmap="viridis",
                        )
                    ax10.set_title("Latent Space Trajectory (PCA)")
                    ax10.set_xlabel("PC1")
                    ax10.set_ylabel("PC2")
                    ax10.set_zlabel("PC3" if n_components == 3 else "N/A")

            fig.colorbar(scatter, ax=ax10, label="Iteration")
        else:
            ax10.set_title("Latent Space Trajectory\nNot enough data points yet")

        # Volume change over time
        ax11 = fig.add_subplot(gs[3, 2])
        ax11.plot(volumes_history_np[: frame + 1])
        ax11.set_title("Volume Change Overtime")
        ax11.set_xlabel("Iteration")
        ax11.set_ylabel("Volume")

        plt.tight_layout()
        return fig

    def create_video_with_opencv(update_func, num_frames, output_filename, fps=10):
        # Get the first frame to determine the video size
        fig = update_func(0)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        frame = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(canvas.get_width_height()[::-1] + (4,))
        frame = frame[:, :, :3]  # Remove alpha channel

        # Initialize the video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(
            output_filename, fourcc, fps, (frame.shape[1], frame.shape[0])
        )

        for i in range(num_frames):
            # Update the figure
            update_func(i)

            # Convert the figure to an image
            canvas.draw()

            frame = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
            frame = frame.reshape(canvas.get_width_height()[::-1] + (4,))
            frame = frame[:, :, :3]  # Remove alpha channel

            # OpenCV uses BGR color format, so we need to convert from RGB to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Write the frame to the video
            video.write(frame)

            # Clear the figure to free up memory
            fig.clear()

        # Release the video writer
        video.release()
        print(f"Video saved as {output_filename}")

    # Set up the plot
    fig = plt.figure(figsize=(20, 24))

    # In your main function, call the video creation function:
    create_video_with_opencv(
        update,
        len(iterations_np),
        f"{config['experiment_name']}_optimization_progress.mp4",
        fps=5,
    )

    # Display the final plot
    # plt.show()
    def create_video_from_two_folders(
        input_folder1,
        input_folder2,
        ref_image1_path,
        ref_image2_path,
        output_video,
        fps=5,
    ):
        # Get list of PNG files in the input folders
        png_files1 = natsorted(
            [f for f in ops.listdir(input_folder1) if f.endswith(".png")]
        )
        png_files2 = natsorted(
            [f for f in ops.listdir(input_folder2) if f.endswith(".png")]
        )

        if not png_files1 or not png_files2:
            print(f"No PNG files found in one or both folders")
            return

        # Read the first images to get dimensions
        first_image1 = cv2.imread(ops.path.join(input_folder1, png_files1[0]))
        first_image2 = cv2.imread(ops.path.join(input_folder2, png_files2[0]))

        # Read reference images
        ref_image1 = cv2.imread(ref_image1_path)
        ref_image2 = cv2.imread(ref_image2_path)

        # Resize reference images to match the height of the corresponding first images
        ref_image1 = cv2.resize(
            ref_image1,
            (
                int(ref_image1.shape[1] * first_image1.shape[0] / ref_image1.shape[0]),
                first_image1.shape[0],
            ),
        )
        ref_image2 = cv2.resize(
            ref_image2,
            (
                int(ref_image2.shape[1] * first_image2.shape[0] / ref_image2.shape[0]),
                first_image2.shape[0],
            ),
        )

        # Calculate dimensions for the combined frame
        frame_width = ref_image1.shape[1] + max(
            first_image1.shape[1], first_image2.shape[1]
        )
        frame_height = first_image1.shape[0] + first_image2.shape[0]

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

        # Iterate through all PNG files
        for png_file1, png_file2 in zip(png_files1, png_files2):
            image1 = cv2.imread(ops.path.join(input_folder1, png_file1))
            image2 = cv2.imread(ops.path.join(input_folder2, png_file2))

            if image1 is None or image2 is None:
                print(f"Could not read image: {png_file1} or {png_file2}")
                continue

            # Create a frame with white background
            frame = np.full((frame_height, frame_width, 3), 255, dtype=np.uint8)

            # Place reference images on the left
            frame[: ref_image1.shape[0], : ref_image1.shape[1]] = ref_image1
            frame[ref_image1.shape[0] :, : ref_image2.shape[1]] = ref_image2

            # Place the images from folders on the right
            frame[
                : image1.shape[0],
                ref_image1.shape[1] : ref_image1.shape[1] + image1.shape[1],
            ] = image1
            frame[
                image1.shape[0] :,
                ref_image2.shape[1] : ref_image2.shape[1] + image2.shape[1],
            ] = image2

            video.write(frame)

        # Release the video writer
        video.release()

        print(f"Video created successfully: {output_video}")

    # Example usage:
    create_video_from_two_folders(
        f"{config['experiment_name']}_p",
        f"{config['experiment_name']}_pc_p",
        f"reference_cylinder_{config['deformation_bend']}_{config['deformation_twist']}_r{config['rotation_inx']}_p".replace(
            ".", "_"
        )
        + "/femesh_plot_iter_0.png",
        f"reference_cylinder_{config['deformation_bend']}_{config['deformation_twist']}_r{config['rotation_inx']}_pc_p".replace(
            ".", "_"
        )
        + "/femesh_plot_iter_0.png",
        f"{config['experiment_name']}_video.mp4",
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run script with a JSON configuration file."
    )
    parser.add_argument(
        "config_path", type=str, help="Path to the JSON configuration file"
    )
    args = parser.parse_args()

    # Load the config
    config = load_config(args.config_path)

    try:
        main(config)
    finally:
        cleanup()

    print("Optimization process completed, progess plots have been saved")
