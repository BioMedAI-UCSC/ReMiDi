import torch
from setup.PGSE import PGSE
from setup.DoublePGSE import DoublePGSE
from setup.CosOGSE import CosOGSE
from setup.SinOGSE import SinOGSE


def compute_mf_jn(eigvals, setup):
    """
    Compute the quantity J(lambda_n, f).
    """

    print("Computing the quantity J(lambda_n, f)")

    # Extract experiment parameters
    sequences = setup.gradient["sequences"]
    ninterval = setup.mf["ninterval"]

    # Sizes
    nsequence = len(sequences)
    neig = len(eigvals)

    # Initialize
    mf_jn = torch.zeros(nsequence, neig, dtype=torch.float)

    # Compute Jn
    for iseq in range(nsequence):
        print(f"  Experiment {iseq+1} of {nsequence}")

        # Gradient sequences
        seq = sequences[iseq]

        for ieig in range(neig):
            lambda_val = eigvals[ieig].item()  # Convert to Python scalar
            if abs(lambda_val) < 1e-16:
                tmp = 0.0
            # Assuming seq is an instance of a class that can be compared directly
            # This part needs to be adjusted based on your actual sequence classes
            elif isinstance(seq, (PGSE, DoublePGSE, CosOGSE, SinOGSE)):
                tmp = seq.J(lambda_val)
            else:
                tmp = seq.J(lambda_val, ninterval)

            mf_jn[iseq, ieig] = tmp

    return mf_jn
