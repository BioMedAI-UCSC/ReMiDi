import plotly.graph_objects as go
import os

def plot_femesh_plotly_2(femesh, compartments, iter, html_folder="html_plots", png_folder="png_plots", which='a'):
    # Create folders if they don't exist
    if which == 'h':
        os.makedirs(html_folder, exist_ok=True)
    elif which == 'p':
        os.makedirs(png_folder, exist_ok=True)
    else:
        os.makedirs(html_folder, exist_ok=True)
        os.makedirs(png_folder, exist_ok=True)

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
            label_index = compartments.index(label) + 1  # Assuming string labels start from 1
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
        title=f"Finite Element Mesh - Iteration {iter}",
        scene=dict(xaxis=dict(title="x"), yaxis=dict(title="y"), zaxis=dict(title="z")),
    )

    if which == 'h':
        # Save as HTML
        html_filename = os.path.join(html_folder, f"femesh_plot_iter_{iter}.html")
        fig.write_html(html_filename)
    elif which == 'p':
        # Save as PNG
        png_filename = os.path.join(png_folder, f"femesh_plot_iter_{iter}.png")
        fig.write_image(png_filename)
    else:
        html_filename = os.path.join(html_folder, f"femesh_plot_iter_{iter}.html")
        fig.write_html(html_filename)
        png_filename = os.path.join(png_folder, f"femesh_plot_iter_{iter}.png")
        fig.write_image(png_filename)

    # Optionally, you can still show the plot
    # fig.show()