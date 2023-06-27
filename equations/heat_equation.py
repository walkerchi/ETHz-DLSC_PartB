import torch 
import numpy as np
PI = np.pi
PI2 = PI * PI
def uniform(a, b, size):
    return (b-a)*torch.rand(size=size) + a

class  HeatEquation:
    """
     Heat Equation
        temperature(scalar) u(t, x, y, mu) is defined by time(scalar) t, position(vec2) x and y, and parameter(vecd) mu 
        PDE:
            du_dt = d^2u_dx^2 + d^2u_dy^2, -1 <= x,y <= 1, 0 <= t <= T, -1^d <= mu <= 1^d
            mu \sim Uniform(-1,1)^d
        Initial Condition:
            u(0, x, y, mu) = - 1/ d \sum_{m=1}^d mu_m * sin(pi * m * x) * sin(pi * m * y) / sqrt(m)
        Boundary Condition:
            u(t, {-1,1}, {-1,1}, mu) = 0
        Solution:
            u = - 1/ d \sum_{m=1}^d mu_m * exp(-2 * pi^2 * m^2 * t) * sin(pi * m * x1) * sin(pi * m * x2) / sqrt(m)
    """

    x_domain = np.array([[-1,1],[-1,1]],dtype=np.float32)

    def __init__(self, d):
        self.d = d 
        self.mu = uniform(-1, 1, size=(d, 1))

    @staticmethod
    def degree_of_freedom(d):
        return d
    
    @property
    def variable(self):
        return self.mu

    def __call__(self, t, x1, x2):
        """
            Parameters:
            -----------
                t: torch.Tensor, shape=(any) or float/int
                x1: torch.Tensor, shape=(any) or float/int
                x2: torch.Tensor, shape=(any) or float/int
            Returns:
            --------
                u: torch.Tensor, shape=(any)
        """
        t  = t [..., None] if isinstance(t , torch.Tensor) else t
        x1 = x1[..., None] if isinstance(x1, torch.Tensor) else x1
        x2 = x2[..., None] if isinstance(x2, torch.Tensor) else x2
        d = self.d 
        m = torch.arange(1, self.d+1).float()
        u = -1 / d * (self.mu * torch.exp(-2 * m * m * PI2 * t) * torch.sin(PI * m * x1) * torch.sin(PI * m * x2) / torch.sqrt(m))
        return u.sum(-1)

