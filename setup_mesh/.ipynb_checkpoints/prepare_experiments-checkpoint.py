import torch
# from torchdiffeq import odeint


def prepare_experiments(setup):
    # We assume we are working with water protons
    gamma = 0.2675222005

    # Assign b-values and q-values
    namplitude = len(setup.gradient["values"])
    nsequence = len(setup.gradient["sequences"])
    setup.gradient["bvalues"] = torch.zeros(namplitude, nsequence)
    setup.gradient["qvalues"] = torch.zeros(namplitude, nsequence)

    for iseq in range(nsequence):
        for i, value in enumerate(setup.gradient["values"]):
            bnq = setup.gradient["sequences"][iseq].bvalue_no_q()
            # print(bnq)
            if setup.gradient["values_type"] == "g":
                q = value * gamma / 1e6
                setup.gradient["qvalues"][i, iseq] = q
                setup.gradient["bvalues"][i, iseq] = bnq * (
                    q**2
                )  # Assumedbnq = 1 for simplification earlier
            elif setup.gradient["values_type"] == "q":
                setup.gradient["qvalues"][i, iseq] = value
                setup.gradient["bvalues"][i, iseq] = bnq * (
                    value**2
                )  # Assumed bnq = 1 for simplification earlier
            elif setup.gradient["values_type"] == "b":
                setup.gradient["bvalues"][i, iseq] = value
                setup.gradient["qvalues"][i, iseq] = torch.sqrt(
                    value / bnq
                )  # Assumed bnq = 1 for simplification

    # Normalize gradient directions
    norm = torch.norm(setup.gradient["directions"], p=2, dim=0)
    # print(norm.shape)
    # print(setup.gradient['directions'].shape)
    setup.gradient["directions"] = setup.gradient["directions"] / norm
    # setup.gradient['directions'] = setup.gradient['directions'].T

    # Set ODE solvers for experiments
    # if hasattr(setup, "btpde") and "ode_solver" not in setup.btpde:
    #     setup.btpde["ode_solver"] = odeint  # Using torchdiffeq's general ODE solver
    # if hasattr(setup, "hadc") and "ode_solver" not in setup.hadc:
    #     setup.hadc["ode_solver"] = odeint  # Using torchdiffeq's general ODE solver
