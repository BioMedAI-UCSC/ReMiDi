def rotate_point_cloud(femesh, rotation_matrices, index):
    """
    Rotates the point cloud in femesh using the specified rotation matrix from the list.
    
    Args:
    femesh (dict): A dictionary containing the point cloud under the key 'points'.
    rotation_matrices (list of torch.Tensor): A list of 3x3 tensors representing rotation matrices.
    index (int): Index of the rotation matrix to use from the list.
    
    Returns:
    dict: The updated femesh with rotated points.
    """
    
    # Select the specified rotation matrix
    rotation_matrix = rotation_matrices[index]

    # Apply the rotation matrix to each vertex
    rotated_points = femesh['points'].T @ rotation_matrix.T
    
    return {**femesh, 'points': rotated_points.T}
