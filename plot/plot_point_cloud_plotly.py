import plotly.graph_objects as go
import torch
import os

def plot_point_cloud_plotly(points, iter, html_folder="html_plots", png_folder="png_plots", which='a'):
    # Create folders if they don't exist
    if which == 'h':
        os.makedirs(html_folder, exist_ok=True)
    elif which == 'p':
        os.makedirs(png_folder, exist_ok=True)
    else:
        os.makedirs(html_folder, exist_ok=True)
        os.makedirs(png_folder, exist_ok=True)
    # Sample points uniformly from the surface of the mesh
    x, y, z = points.clone().detach().cpu().unbind(0)    

    # Create the 3D scatter plot
    trace = go.Scatter3d(
        x=x,
        y=y,  # Note: y and z are swapped to match the original orientation
        z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=z,  # Color points based on z-coordinate
            colorscale='Viridis',
            opacity=0.8
        )
    )

    # Set up the layout
    layout = go.Layout(
        title=f"Points of the Mesh - Iteration {iter}",
        scene=dict(
            xaxis_title='x',
            yaxis_title='z',
            zaxis_title='y',
            aspectmode='data'  # This preserves the data aspect ratio
        ),
        width=600,
        height=600,
        margin=dict(r=20, l=10, b=10, t=40)
    )

    # Create the figure and show it
    fig = go.Figure(data=[trace], layout=layout)
    
    # Set the initial camera position to match the original view
    camera = dict(
        eye=dict(x=1.5, y=-1.5, z=0.5)
    )
    fig.update_layout(scene_camera=camera)
    
    if which == 'h':
        # Save as HTML
        html_filename = os.path.join(html_folder, f"femesh_plot_iter_{iter}.html")
        fig.write_html(html_filename)
    elif which == 'p':
        # Save as PNG
        png_filename = os.path.join(png_folder, f"femesh_plot_iter_{iter}.png")
        fig.write_image(png_filename)
    else:
        # Save as HTML
        html_filename = os.path.join(html_folder, f"femesh_plot_iter_{iter}.html")
        fig.write_html(html_filename)
        # Save as PNG
        png_filename = os.path.join(png_folder, f"femesh_plot_iter_{iter}.png")
        fig.write_image(png_filename)
    # fig.show()

# Usage example:
# plot_pointcloud_plotly(mesh, title="My Point Cloud")