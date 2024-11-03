
import torch
# from torchdiffeq import odeint
from setup_mesh.directions import directions
from setup_mesh.PGSE import PGSE


class Setup1AxonAnalytical_LowRes:
    def __init__(self):
        # File name to load or store cell description, surface geometry, mesh, and simulation results
        # self.name = "mesh_files/spheres/1sphere_analytical"

        # Geometry parameters
        self.geometry = {
            "cell_shape": "cylinder",
            "ncell": 1,
            "rmin": 5,
            "rmax": 5,
            "dmin": 0.1,
            "dmax": 0.2,
            "height": 20,
            "deformation": torch.tensor([0.0, 0.0]),
            "include_in": False,  # True
            "in_ratio": 0.6,
            "ecs_shape": "no_ecs",  # "ecs_shape": "tight_wrap" / "no_ecs"
            "ecs_ratio": 0.5,
            "refinement": 1,
        }

        # PDE parameters - initially set, can be modified later
        self.pde = {
            "diffusivity_in": 0.002,
            "diffusivity_out": 0.002,
            "diffusivity_ecs": 0.002,
            "relaxation_in": float("inf"),
            "relaxation_out": float("inf"),
            "relaxation_ecs": float("inf"),
            "initial_density_in": 1.0,
            "initial_density_out": 1.0,
            "initial_density_ecs": 1.0,
            "permeability_in_out": 1e-4,
            "permeability_out_ecs": 1e-4,
            "permeability_in": 0,
            "permeability_out": 0,
            "permeability_ecs": 0,
        }

        # Gradient sequences
        
        self.gradient = {
            "values": torch.arange(0, 10001, 500),
            "values_type": "b",
            "sequences": [
                PGSE(15000, 25000),
                PGSE(30000, 50000),
                PGSE(10000, 100000)
            ],
            "directions": directions("setup_mesh/PointSets/Elec040.txt")
        }
        

        # BTPDE experiment parameters
        self.btpde = {"reltol": 1e-4, "abstol": 1e-6}

        # MF experiment parameters
        self.mf = {
            "length_scale": 1,  # Minimum length scale of eigenfunctions
            "neig_max": 1000,  # Requested number of eigenvalues
            "ninterval": 500,
        }

    # def solve_ode(self, ode_func, y0, t):
    #     # Solving the ODE using torchdiffeq
    #     solution = odeint(
    #         ode_func, y0, t, rtol=self.btpde["reltol"], atol=self.btpde["abstol"]
    #     )
    #     return solution
