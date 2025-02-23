from setup_mesh.prepare_pde import prepare_pde


def update_pde(setup):
    # Update the PDE parameters using the prepare_pde function
    setup.pde = prepare_pde(setup)
