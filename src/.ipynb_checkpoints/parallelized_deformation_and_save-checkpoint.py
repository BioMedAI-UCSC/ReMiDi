import torch
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor

def save_files(v, combo, eigenvectors, base_path):
    os.makedirs(base_path, exist_ok=True)
    i, j, u, x = combo
    i_int = int(i.item() * 100)
    j_int = int(j.item())
    u_int = int(u.item() * 100)
    x_int = int(x.item() * 100)
    file_id = f'{u_int:02d}{x_int:03d}{i_int:02d}{j_int:02d}'
    
    # Save spectral coefficients
    spectral_coefficients = torch.matmul(eigenvectors.T, v.T)
    spec_path = os.path.join(base_path, f'mesh{file_id}.coeffs.pth')
    torch.save(spectral_coefficients, spec_path)
    
    # Save JSON info
    json_content = {"id": int(file_id)}
    json_path = os.path.join(base_path, f'mesh{file_id}.infos.json')
    with open(json_path, 'w') as json_file:
        json.dump(json_content, json_file)

def parallelized_deformation_and_save(thin_mesh, fan_single_cylinder_batches, deform_domain, eigenvectors, base_path):
    # Define the parameter ranges
    i_range = torch.arange(0.25, 1, 0.25)
    j_range = torch.arange(0.0, 91.0, 15.0)
    u_range = torch.arange(0, 0.51, 0.04)
    x_range = torch.arange(0, 3.1, 0.04)

    # Create a mesh grid of all parameter combinations
    all_combinations = torch.cartesian_prod(i_range, j_range, u_range, x_range)

    # Move data to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_combinations = all_combinations.to(device)
    thin_mesh_points = thin_mesh['points'].to(device)
    eigenvectors = eigenvectors.to(device)

    # Process in batches to avoid memory issues
    batch_size = 1000  # Adjust based on your GPU memory
    print(thin_mesh_points.shape)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for batch in torch.split(all_combinations, batch_size):
            # Apply fan_single_cylinder_batches to the entire batch at once
            bent_meshes = fan_single_cylinder_batches(thin_mesh_points.unsqueeze(0).expand(batch.shape[0], -1, -1),
                                                      bend_point=batch[:, 0],
                                                      bend_angle=batch[:, 1],
                                                      axis=2)
            
            # Apply deform_domain to each bent mesh
            deformed = [deform_domain(mesh.T, combo[2:].tolist()) for mesh, combo in zip(bent_meshes, batch)]
            
            # Save files in parallel
            futures = [executor.submit(save_files, v.T, combo, eigenvectors, base_path) 
                       for v, combo in zip(deformed, batch)]
            
            # Wait for all saving operations to complete
            for future in futures:
                future.result()

    print("All meshes processed and saved.")

# Example usage:
# base_path = 'sample_testing_deformation_creations'
# parallelized_deformation_and_save(thin_mesh, fan_single_cylinder_batches, deform_domain, eigenvectors, base_path)