import plotly.graph_objects as go


def plot_femesh_plotly(femesh, compartments, filename=""):
    # colors = ["green", "yellow", "black", "red", "blue", "black", "cyan", "white"]
    colors = [
        "#F5F5F5",  # White Smoke
        "#6495ED",  # Cornflower Blue
        "#8FBC8F",  # Dark Sea Green
        "#F0E68C",  # Khaki
        "#A9A9A9",  # Dark Gray
        "#CD5C5C",  # Red
        "#708090",  # Slate Gray
        "#00CED1",  # Dark Turquoise
    ]

    # Points and facets from femesh
    points = femesh["points"]
    facets = femesh["facets"].T  # Transpose to get facets as a list of triangles

    fig = go.Figure()

    for label in compartments:
        # Determine if compartments are numeric or string labels
        if isinstance(label, str):
            label_index = (
                compartments.index(label) + 1
            )  # Assuming string labels start from 1
        else:
            label_index = label

        # Filter facets for the current compartment
        compartment_facets = facets[femesh["facetmarkers"] == label_index]

        for facet in compartment_facets:
            triangle = points[:, facet.long()].cpu().detach().numpy()
            fig.add_trace(
                go.Mesh3d(
                    x=triangle[0, :],
                    y=triangle[1, :],
                    z=triangle[2, :],
                    color=colors[label_index % len(colors)],
                    opacity=0.7,
                )
            )

    fig.update_layout(
        title="Finite Element Mesh",
        scene=dict(xaxis=dict(title="x"), yaxis=dict(title="y"), zaxis=dict(title="z")),
    )

    if filename:
        fig.write_html(filename)
    else:
        fig.show()
