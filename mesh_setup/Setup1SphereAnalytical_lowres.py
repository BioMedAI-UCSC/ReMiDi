import torch
# from torchdiffeq import odeint
from mesh_setup.directions import directions
from mesh_setup.PGSE import PGSE


class Setup1SphereAnalytical_lowres:
    def __init__(self):
        # File name to load or store cell description, surface geometry, mesh, and simulation results
        self.name = "mesh_files/spheres/1sphere_analytical"

        # Geometry parameters
        self.geometry = {
            "cell_shape": "sphere",
            "ncell": 1,
            "rmin": 5,
            "rmax": 5,
            "dmin": 0.1,
            "dmax": 0.2,
            "deformation": torch.tensor([0.0, 0.0]),
            "include_in": False,  # True
            "in_ratio": 0.6,
            "ecs_shape": "no_ecs",  # "ecs_shape": "tight_wrap" / "no_ecs"
            "ecs_ratio": 0.5,
            "refinement": 8,
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
            "values": torch.arange(
                0, 10001, 500
            ),  # torch.arange(0, 10.1, 0.5) / torch.arange(0, 10001, 500), torch.arange(0, 20001, 1000)
            "values_type": "b",
            "sequences": [
                PGSE(5000, 10000),
                PGSE(15000, 25000),
                PGSE(30000, 50000),
            ],  # "sequences": [PGSE(5, 10), PGSE(10, 100)], in ms PGSE(10000, 100000)
            "directions": directions("mesh_setup/PointSets/Elec053.txt"),
            # "directions": torch.tensor([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=torch.float).T # torch.tensor([[1, 0, 0], [-0.5000, 0.8660, 0], [-0.5000, -0.8660, 0]])
        }

        # BTPDE experiment parameters
        self.btpde = {"reltol": 1e-4, "abstol": 1e-6}

        # MF experiment parameters
        self.mf = {
            "length_scale": 2,  # Minimum length scale of eigenfunctions
            "neig_max": 400,  # Requested number of eigenvalues
            "ninterval": 500,
        }

    # def solve_ode(self, ode_func, y0, t):
    #     # Solving the ODE using torchdiffeq
    #     solution = odeint(
    #         ode_func, y0, t, rtol=self.btpde["reltol"], atol=self.btpde["abstol"]
    #     )
    #     return solution
