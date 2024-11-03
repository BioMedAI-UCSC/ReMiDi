from src.flux_matrixP1_3D import flux_matrixP1_3D


def assembleflux_matrix(points, facets):
    ncompartment = len(facets)
    nboundary = len(facets[0]) if ncompartment > 0 else 0

    flux_matrices = []

    for icmpt in range(ncompartment):
        compartment_matrices = []
        for iboundary in range(nboundary):
            boundary = facets[icmpt][iboundary]

            if boundary is not None and boundary.nelement() > 0:
                # Assuming boundary and points are already tensors and transposed as needed
                matrix, _ = flux_matrixP1_3D(
                    boundary.t().contiguous(), points[icmpt].t().contiguous()
                )
                compartment_matrices.append(matrix)

            else:
                compartment_matrices.append(None)  # Or an appropriate placeholder
        flux_matrices.append(compartment_matrices)

    # Returning as a list of lists
    return flux_matrices
