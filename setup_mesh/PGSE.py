import torch
from torchquad import MonteCarlo, set_up_backend, Trapezoid, Simpson, Boole

set_up_backend("torch")  # Ensure torchquad uses PyTorch backend

class PGSE:
    def __init__(self, delta, Delta, echotime=None):
        self.delta = delta
        self.Delta = Delta
        self.echotime = echotime if echotime else Delta + delta
        self.intg = Trapezoid()  # Initialize the Trapezoidal integrator

    def call(self, t):
        # This function is expected to work with scalar values.
        # Adapt it if you plan to work directly with tensors.
        if 0 <= t < self.delta:
            return 1.0
        elif self.Delta <= t <= self.echotime:
            return -1.0
        else:
            return 0.0

    def call_tensor(self, t):
        # Tensor-compatible call function for integration
        return torch.where((0 <= t) & (t < self.delta), 1.0,
               torch.where((self.Delta <= t) & (t <= self.echotime), -1.0, 0.0))

    def integral(self, t):
        # Prepare tensor for t if it's not already a tensor
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], dtype=torch.float32)
        # Integrate the call function over [0, t] for each element in t
        # Assuming t is a tensor
        results = torch.zeros(t.size())
        for i, ti in enumerate(t):
            # Lambda function for integration expects a tensor of samples
            func = lambda x: self.call_tensor(x * ti)
            result = self.intg.integrate(func, dim=1, N=1000, integration_domain=[[0, 1]])
            results[i] = result * ti  # Scale result by ti due to change of variable
        return results

    def integral_F2(self):
        # This method correctly integrates the square of the call function over [0, echotime]
        # Utilize a tensor for echotime if it's not already
        if not isinstance(self.echotime, torch.Tensor):
            echotime_tensor = torch.tensor([self.echotime], dtype=torch.float32)
        else:
            echotime_tensor = self.echotime

        func = lambda s: self.integral(s)**2
        result = self.intg.integrate(func, dim=1, N=1000, integration_domain=[[0, echotime_tensor.item()]])
        return result.item()

    def bvalue_no_q(self):
        return self.integral_F2()
    def diffusion_time(self):
        # Get diffusion time of the PGSE sequence
        return self.Delta - self.delta / 3

    def diffusion_time_sta(self):
        # Get STA diffusion time of the PGSE sequence
        d = self.delta
        D = self.Delta
        out = (4 / 35) * ((D + d)**(7 / 2) + (D - d)**(7 / 2) - 2 * D**(7 / 2) - 2 * d**(7 / 2)) / (d**2 * (D - d/3))
        return out**2

    def J(self, lambda_val):
        d = torch.tensor(self.delta)
        D = torch.tensor(self.Delta)
        exp = torch.exp

        # Taylor expansion for lambda close to 0
        taylor_expansion = lambda_val - lambda_val**2 * D**2 / (2 * (D - d/3)) + \
                           lambda_val**3 * (10*D**3 + 5*D*d**2 - d**3) / (20 * (3*D - d)) - \
                           lambda_val**4 * (D**4 + D**2*d**2) / (8 * (3*D - d)) + \
                           lambda_val**5 * (21*D**5 + 35*D**3*d**2 + 7*D*d**4 - d**5) / (840 * (3*D - d))

        # Complex formula part
        complex_formula = -1 * (exp(-lambda_val * (D + d)) + exp(-lambda_val * (D - d)) - 2 * exp(-lambda_val * d) - 2 * exp(-lambda_val * D) + 2 * (1 - lambda_val * d)) / \
                          (lambda_val**2 * d**2 * (D - d/3))

        # Use torch.where to choose between Taylor expansion and the complex formula based on the value of lambda_val
        if_else = torch.tensor(lambda_val < 1e-7, dtype=bool)
        result = torch.where(if_else, taylor_expansion, complex_formula)

        return result

    def intervals(self):
        # Get intervals of the sequence
        timelist = [0, self.delta, self.Delta, self.Delta+self.delta]
        interval_str = ["[0, delta]", "[delta, Delta]", "[Delta, Delta+delta]"]
        timeprofile_str = ["f(t) = 1 (constant)", "f(t) = 0 (constant)", "f(t) = -1 (constant)"]

        if self.delta == self.Delta:
            # Remove unused interval
            timelist.pop(2)  # Removing the third element which is unnecessary when delta == Delta
            del interval_str[1]  # Removing the second interval description
            del timeprofile_str[1]  # Removing the second time profile description

        return timelist, interval_str, timeprofile_str