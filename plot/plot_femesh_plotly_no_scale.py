import plotly.graph_objects as go
import numpy as np

def plot_femesh_plotly_no_scale(femesh, compartments, filename=""):
    colors = [
        "#A9A9A9", "#CD5C5C", "#708090", "#00CED1",
        "#F5F5F5", "#6495ED", "#8FBC8F", "#F0E68C"
    ]
    
    points = femesh["points"].cpu().detach().numpy()
    facets = femesh["facets"].T.cpu().detach().numpy()
    
    fig = go.Figure()
    
    for label in compartments:
        label_index = compartments.index(label) + 1 if isinstance(label, str) else label
        compartment_facets = facets[femesh["facetmarkers"].cpu().detach().numpy() == label_index]
        
        for facet in compartment_facets:
            triangle = points[:, facet]
            fig.add_trace(
                go.Mesh3d(
                    x=triangle[0, :],
                    y=triangle[1, :],
                    z=triangle[2, :],
                    color=colors[label_index % len(colors)],
                    opacity=0.7,
                )
            )
    
    # Calculate the range for each axis
    x_range = np.ptp(points[0, :])
    y_range = np.ptp(points[1, :])
    z_range = np.ptp(points[2, :])
    
    # Calculate the center for each axis
    x_center = np.mean(points[0, :])
    y_center = np.mean(points[1, :])
    z_center = np.mean(points[2, :])
    
    # Determine the maximum range to set a consistent scale
    max_range = max(x_range, y_range, z_range)
    
    fig.update_layout(
        title="Finite Element Mesh (True Dimensions)",
        scene=dict(
            xaxis=dict(title="x", range=[x_center - max_range/2, x_center + max_range/2]),
            yaxis=dict(title="y", range=[y_center - max_range/2, y_center + max_range/2]),
            zaxis=dict(title="z", range=[z_center - max_range/2, z_center + max_range/2]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1)
        ),
    )
    
    if filename:
        fig.write_html(filename)
    else:
        fig.show()