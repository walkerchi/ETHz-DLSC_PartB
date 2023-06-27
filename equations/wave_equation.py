import torch 
import numpy as np
from typing import Union , Optional
PI = np.pi
def uniform(a, b, size):
    return (b-a)*torch.rand(size=size) + a

class WaveEquation:
    """
    The solution of the wave equation u(x, y, t)
    Wave Equation (2D):
        PDE:
            u_{tt} = c^{2} \cdot (u_{xx} + u_{yy}}
            x, y \in \mathbb{T}^{2} ([0, 1]^2), t \in [0, T]
            a \sim Uniform(-1, 1)^{K^2}
        Boundary Condition: 
            (no listed explicitly)
            u(0, y, t) = u(1, y, t) = 0
            u(x, 0, t) = u(x, 1, t) = 0
            x \in [0, 1], y \in [0, 1], t \in [0, T]
        Intial Conditions:
            u(0, x, y, a) = \frac{\pi}{K^2} \sum_{i,j=1}^{K} a_{ij} \cdot (i^2 + j^2)^{-r} sin(\pi ix) sin(\pi jy) \forall x,y \in [0, 1]
        Solution:
            u(t, x, y, a) = \frac{\pi}{K^2} \sum_{i,j=1}^{K} a_{ij} \cdot (i^2 + j^2)^{-r} sin(\pi ix) sin(\pi jy) cos(c\pi t \sqrt{i^2 + j^2}), \forall x,y \in [0, 1]
                          
    """

    x_domain = np.array([[0,1],[0,1]],dtype=np.float32)

    def __init__(self, K:int, r:float = 0.85, c:float = 0.1) -> None:
        
        self.c = c
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
        c = self.c
        i, j = torch.meshgrid(
            torch.arange(1, K + 1).float(),
            torch.arange(1, K + 1).float())
        
        temp = (i**2 + j**2)

        result =  PI / K**2  * torch.sum(a * temp**(-r) * torch.sin(PI * i * x1) * torch.sin(PI * j * x2) * torch.cos(c * PI * t * (temp)**0.5), dim=(-2,-1))
        return result
        

if __name__ == "__main__":
    we = WaveEquation(4)
    t = torch.tensor([0,0,0]).float()
    x = torch.tensor([0,0,0]).float()
    y = torch.tensor([0,0,0]).float()
    print(we(t, x, y).shape)