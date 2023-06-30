import torch 
import numpy as np
from typing import Union , Optional
PI = np.pi
def uniform(a, b, size):
    return (b-a)*torch.rand(size=size) + a

class PoissonEquation:
    """
    The solution of the wave equation u(x, y, t)
    Wave Equation (2D):
        PDE:
           -\Delta u = -(u_{xx} + u_{yy}) = f
            x, y \in \mathbb{T}^{2} ([0, 1]^2), t \in [0, T]
            a \sim Uniform(-1, 1)^{K^2}
        Boundary Condition: 
            (no listed explicitly)
            u(0, y, t) = u(1, y, t) = 0
            u(x, 0, t) = u(x, 1, t) = 0
            x \in [0, 1], y \in [0, 1], t \in [0, T]
        Intial Conditions(source function):
            f=\frac{\pi}{K^2} \sum_{i,j=1}^{K} a_{ij} \cdot (i^2 + j^2)^{r} sin(\pi ix) sin(\pi jy), ,\quad \forall (x,y) \in D
        Solution:
            u(x, y) = \frac{1}{\pi\cdot K^2} \sum_{i,j=1}^{K} a_{ij} \cdot (i^2 + j^2)^{r-1} sin(\pi ix) sin(\pi jy),\quad \forall (x,y) \in D
                          
    """

    x_domain = np.array([[0,1],[0,1]],dtype=np.float32)

    def __init__(self, K:int, r:float = 0.85) -> None:
        
        self.r = r
        self.K:int = K
        self.a:torch.Tensor = uniform(-1, 1, (self.K, self.K)).float()

    @staticmethod
    def degree_of_freedom(K:int) -> int:
        return K**2
    
    @property
    def variable(self) -> torch.Tensor:
        return self.a

    def __call__(self, t:Union[float, torch.Tensor], x1:torch.Tensor, x2:torch.Tensor, a:Optional[torch.Tensor] = None) -> torch.Tensor: 
        """
        Args:
            t: time
            x1: x coordinate
            x2: y coordinate
            a: the coefficients of the solution
        Return:
            the solution of the wave equation. The dimension of the solution is 1"""

        
   
        x1 = x1[..., None, None] if isinstance(x1, torch.Tensor) else x1
        x2 = x2[..., None, None] if isinstance(x2, torch.Tensor) else x2
        t  = t[..., None, None]  if isinstance(t, torch.Tensor)  else t

        if a is None:
            a = self.a
        
        r = self.r
        K = self.K
        i, j = torch.meshgrid(
            torch.arange(1, K + 1).float(),
            torch.arange(1, K + 1).float())
        
        temp = (i**2 + j**2)

        if t == 0:
            result = PI / K**2  * torch.sum(a * temp**(r) * torch.sin(PI * i * x1) * torch.sin(PI * j * x2), dim=(-2,-1)) 
        else:
            result =  1 / PI / K**2  * torch.sum(a * temp**(r-1) * torch.sin(PI * i * x1) * torch.sin(PI * j * x2), dim=(-2,-1))
        return result
        

if __name__ == "__main__":
    import os
    from matplotlib.animation  import FuncAnimation
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from tqdm import tqdm
    import matplotlib.pyplot as plt 

    spatial_resolution = 64 * 64
    time_resolution    = 100
    T = 1

    x, y = torch.meshgrid(
        torch.linspace(-1, 1, int(np.sqrt(spatial_resolution))),
        torch.linspace(-1, 1, int(np.sqrt(spatial_resolution)))
    )


    for K in [1,2,4,8,16]:
        fig, ax = plt.subplots(figsize=(6,6))
        he = PoissonEquation(K)
        fig, ax = plt.subplots(ncols=2, figsize=(12,6))
        ax[0].matshow(he(0, x, y).detach().numpy(), cmap='jet')
        ax[1].matshow(he(1, x, y).detach().numpy(), cmap='jet')
        ax[0].set_title("f")
        ax[1].set_title("u")
        os.makedirs("video/poisson", exist_ok=True)
        fig.savefig(f"video/poisson/K={K}.png")
        plt.close(fig=fig)