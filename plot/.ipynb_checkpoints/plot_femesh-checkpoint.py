import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_femesh(femesh):
    colors = ["r", "b", "g", "y", "k", "c", "m"]

    ncompartment = femesh["ncompartment"]

    for icmpt in range(ncompartment):
        points = femesh["points"][icmpt]
        elements = femesh["elements"][icmpt]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

        # Plot each element as a triangle
        for element in elements.T:
            triangle = points[:, element].cpu().detach().numpy()
            ax.plot_trisurf(
                triangle[0, :],
                triangle[1, :],
                triangle[2, :],
                color=colors[icmpt % len(colors)],
                alpha=0.7,
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.title(f"Compartment {icmpt + 1}")
        plt.show()
