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
        self.mu = uniform(-1, 1, size=(d,))

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
                t: torch.Tensor, shape=(any) or float/int
                x1: torch.Tensor, shape=(any) or float/int
                x2: torch.Tensor, shape=(any) or float/int
            Returns:
            --------
                u: torch.Tensor, shape=(any)
                u: torch.Tensor, shape=(any)
        """
        t  = t [..., None] if isinstance(t , torch.Tensor) else t
        x1 = x1[..., None] if isinstance(x1, torch.Tensor) else x1
        x2 = x2[..., None] if isinstance(x2, torch.Tensor) else x2
 
        d = self.d 
        m = torch.arange(1, self.d+1).float()
        u = -1 / d * (self.mu * torch.exp(-2 * m * m * PI2 * t) * torch.sin(PI * m * x1) * torch.sin(PI * m * x2) / torch.sqrt(m))
        return u.sum(-1)


if __name__ == '__main__':
    import os
    from matplotlib.animation  import FuncAnimation
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from tqdm import tqdm
    import matplotlib.pyplot as plt 

    spatial_resolution =  64 * 64
    time_resolution    = 100
    T = 0.01

    x, y = torch.meshgrid(
        torch.linspace(-1, 1, int(np.sqrt(spatial_resolution))),
        torch.linspace(-1, 1, int(np.sqrt(spatial_resolution)))
    )


    for d in [1,2,4,8,16]:
        fig, ax = plt.subplots(figsize=(6,6))
        he = HeatEquation(d)
        mat = ax.matshow(he(0, x, y).detach().numpy(), cmap='jet')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(mat, cax=cax, orientation='vertical')
        def animate(i):
            t = i * T / time_resolution
            u = he(t, x, y)
            ax.set_title(f"heat equation, d={d}, t={t:.5f}")
            mat.set_data(u.detach().numpy())
            mat.set_clim(u.min(),u.max())
            cbar.update_normal(mat)
            return mat,
        
        anim = FuncAnimation(fig, animate, frames=tqdm(range(time_resolution)), interval=100, blit=True)
        os.makedirs("video/heat", exist_ok=True)
        anim.save(f"video/heat/d={d}.mp4")
        plt.close(fig=fig)


    