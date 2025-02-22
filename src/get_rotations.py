import torch
import roma

def get_rotations():
    # 90 degrees (π/2) in radians
    angle_90 = torch.pi / 2

    # 180 degrees (π) in radians
    angle_180 = torch.pi

    # Rotation vectors for each specified configuration
    # 0. 0 degrees for control
    rotvec_0 = torch.tensor([0.0, 0.0, 0.0])

    # 1. 90 degrees around each axis individually
    rotvec_90_x = angle_90 * torch.tensor([1.0, 0.0, 0.0])
    rotvec_90_y = angle_90 * torch.tensor([0.0, 1.0, 0.0])
    rotvec_90_z = angle_90 * torch.tensor([0.0, 0.0, 1.0])

    # 2. 90 degrees around two axes combined
    rotvec_90_xy = angle_90 * (torch.tensor([1.0, 1.0, 0.0]) / torch.sqrt(torch.tensor(2.0)))
    rotvec_90_xz = angle_90 * (torch.tensor([1.0, 0.0, 1.0]) / torch.sqrt(torch.tensor(2.0)))
    rotvec_90_yz = angle_90 * (torch.tensor([0.0, 1.0, 1.0]) / torch.sqrt(torch.tensor(2.0)))

    # 3. 90 degrees around all three axes
    rotvec_90_xyz = angle_90 * (torch.tensor([1.0, 1.0, 1.0]) / torch.sqrt(torch.tensor(3.0)))

    # 4. 180 degrees around each axis individually
    rotvec_180_x = angle_180 * torch.tensor([1.0, 0.0, 0.0])
    rotvec_180_y = angle_180 * torch.tensor([0.0, 1.0, 0.0])
    rotvec_180_z = angle_180 * torch.tensor([0.0, 0.0, 1.0])

    # List of rotation vectors for convenience
    rotation_vectors = [
        rotvec_0,
        rotvec_90_x, rotvec_90_y, rotvec_90_z,
        rotvec_90_xy, rotvec_90_xz, rotvec_90_yz,
        rotvec_90_xyz,
        rotvec_180_x, rotvec_180_y, rotvec_180_z
    ]

    # Convert each rotation vector to a rotation matrix
    rotation_matrices = [roma.rotvec_to_rotmat(rotvec) for rotvec in rotation_vectors]

    return rotation_matrices