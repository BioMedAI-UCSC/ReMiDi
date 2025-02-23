import torch
import torch
from torchdiffeq import odeint
from setup.directions import directions
from setup.PGSE import PGSE
class Setup1AxonAnalytical:
    def __init__(self):
        # File name to load or store cell description, surface geometry, mesh, and simulation results
        self.name = "mesh_files/cylinders/1axon_analytical"

        # Geometry parameters
        self.geometry = {
            "cell_shape": "cylinder",
            "ncell": 1,
            "rmin": 5,
            "rmax": 5,
            "dmin": 0.2,
            "dmax": 0.3,
            "height": 50,
            "deformation": torch.tensor([0.0, 0.0]),
            "include_in": False,
            "in_ratio": 0.6,
            "ecs_shape": "no_ecs",
            "ecs_ratio": 0.5,
            "refinement": 1
        }

        # PDE parameters
        self.pde = {
            "diffusivity_in": 0.002,
            "diffusivity_out": 0.002,
            "diffusivity_ecs": 0.002,
            "relaxation_in": float('inf'),
            "relaxation_out": float('inf'),
            "relaxation_ecs": float('inf'),
            "initial_density_in": 1.0,
            "initial_density_out": 1.0,
            "initial_density_ecs": 1.0,
            "permeability_in_out": 1e-4,
            "permeability_out_ecs": 1e-4,
            "permeability_in": 0,
            "permeability_out": 0,
            "permeability_ecs": 0
        }

        # Gradient sequences
        self.gradient = {
            "values": torch.arange(0, 10001, 500),
            "values_type": "b",
            "sequences": [PGSE(5000, 10000), PGSE(10000, 100000)],
            "directions": directions("/media/DATA_18_TB_1/shri/DMRI_Code/setup/PointSets/Elec045.txt")
        }

        # BTPDE experiment parameters
        self.btpde = {
            "ode_solver": odeint,  # Using torchdiffeq.odeint for this example
            "reltol": 1e-4,
            "abstol": 1e-6
        }

        # MF experiment parameters
        self.mf = {
            "length_scale": 1,                              # Minimum length scale of eigenfunctions
            "neig_max": 1000,                               # Requested number of eigenvalues
            "ninterval": 500
        }

# Example usage
setup = Setup1AxonAnalytical()