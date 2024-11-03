import torch
from torchquad import Trapezoid, set_up_backend
# from torch_fresnel import fresnel

set_up_backend("torch")  # Ensure torchquad uses PyTorch backend

class CosOGSE:
    def __init__(self, delta, Delta, nperiod, echotime=None):
        self.delta = delta
        self.Delta = Delta
        self.nperiod = nperiod
        self.echotime = echotime if echotime else Delta + delta
        self.intg = Trapezoid()  # Initialize the Trapezoidal integrator

    def call(self, t):
        d = self.delta
        D = self.Delta
        n = self.nperiod
        return torch.where((0 <= t) & (t < d), torch.cos(2 * torch.pi * n / d * t),
                           torch.where((D <= t) & (t <= self.echotime), -torch.cos(2 * torch.pi * n / d * (t - D)), 0.0))

    def integral(self, t):
        d = self.delta
        D = self.Delta
        n = self.nperiod
        return (torch.where((0 <= t) & (t < d), torch.sin(2 * torch.pi * n / d * t), 0.0)
                + torch.where((d <= t) & (t <= self.echotime), torch.sin(2 * torch.pi * n), 0.0)
                - torch.where((D <= t) & (t <= self.echotime), torch.sin(2 * torch.pi * n / d * (t - D)), 0.0)) \
               * d / (2 * torch.pi * n)
        
    def bvalue_no_q(self):
        return self.integral_F2()

    def integral_F2(self):
        return self.delta**3 / (4 * torch.pi**2 * self.nperiod**2)

    def diffusion_time(self):
        return 1 / 8 * self.delta / self.nperiod

    def diffusion_time_sta(self):
        d = self.delta
        D = self.Delta
        n = self.nperiod
        S = torch.sin(torch.tensor(2) * torch.pi * n * D / d)
        C = torch.cos(torch.tensor(2) * torch.pi * n * D / d)
        
        # Compute Fresnel integrals using torch_fresnel
        FS1, FC1 = fresnel(torch.tensor(2) * torch.sqrt(n))
        FS2, FC2 = fresnel(torch.tensor(2) * torch.sqrt(D * n / d))
        FS3, FC3 = fresnel(torch.tensor(2) * torch.sqrt((D + d) * n / d))
        FS4, FC4 = fresnel(torch.tensor(2) * torch.sqrt((D - d) * n / d))
        
        out = 3 / 8 / torch.sqrt(n * d) * (
            2 * D * FC2 * C +
            d * FC4 * C -
            d * FC3 * C -
            D * FC3 * C -
            D * FC4 * C +
            2 * D * FS2 * S +
            d * FS4 * S -
            d * FS3 * S -
            D * FS3 * S -
            D * FS4 * S +
            2 * d * FC1
        ) + 9 * torch.sqrt(d) / (32 * torch.pi) / n**(3/2) * (
            2 * FS2 * C -
            FS3 * C -
            FS4 * C -
            2 * FC2 * S +
            FC3 * S +
            FC4 * S +
            2 * FS1
        )
    
        return out**2

    def J(self, lambda_val):
        d = self.delta
        D = self.Delta
        n = self.nperiod
        exp = torch.exp

        # Taylor expansion for lambda close to 0
        taylor_expansion = lambda_val - lambda_val**3 * d**2 * 3 / (4 * (n * torch.pi)**2) \
                           + lambda_val**5 * (12 * D * n**2 * torch.pi**2 * d**3 + 15 * d**4 - 4 * n**2 * torch.pi**2 * d**4) / (48 * n**4 * torch.pi**4) \
                           - lambda_val**6 * D**2 * d**3 / (8 * (n * torch.pi)**2)

        # Complex formula part
        complex_formula = 4 * n**2 * torch.pi**2 * lambda_val * ( \
            -exp(-lambda_val * (D + d)) * lambda_val * d \
            - exp(-lambda_val * (D - d)) * lambda_val * d \
            + 2 * exp(-lambda_val * D) * lambda_val * d \
            + 2 * exp(-lambda_val * d) * lambda_val * d \
            + 4 * n**2 * torch.pi**2 \
            + (lambda_val * d - 2) * lambda_val * d) \
            / (4 * n**2 * torch.pi**2 + lambda_val**2 * d**2)**2

        # Use torch.where to choose between Taylor expansion and the complex formula based on the value of lambda_val
        return torch.where(lambda_val < 1e-7, taylor_expansion, complex_formula)

    def intervals(self):
        timelist = [0, self.delta, self.Delta, self.Delta + self.delta]
        interval_str = ["[0, delta]", "[delta, Delta]", "[Delta, Delta+delta]"]
        funcname = f"cos(2*pi*{self.nperiod}/delta*t)"
        timeprofile_str = [f"f(t) = {funcname}", "f(t) = 0 (constant)", f"f(t) = -{funcname}"]

        if self.delta == self.Delta:
            timelist.pop(2)
            interval_str.pop(1)
            timeprofile_str.pop(1)

        return timelist, interval_str, timeprofile_str