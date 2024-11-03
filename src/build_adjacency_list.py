def build_adjacency_list(elements):
    """
    Build an adjacency list from elements of a tetrahedral mesh.
    Elements should be a tensor where each column represents a tetrahedron and contains indices of its vertices.
    """
    adjacency_list = {}
    elements = elements.T  # Transpose to iterate over tetrahedra
    for tetra in elements:
        for i in range(4):  # Assuming 4 vertices per tetrahedron
            vertex_index = tetra[i].item()  # Convert tensor to integer
            if vertex_index not in adjacency_list:
                adjacency_list[vertex_index] = set()
            # Add adjacent vertices, ensuring indices are integers
            adjacency_list[vertex_index].update(
                tetra[j].item() for j in range(4) if i != j
            )
    return adjacency_list
