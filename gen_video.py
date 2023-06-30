"""
    visualize the evolution of the equations over time
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import animation
from matplotlib.animation  import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from itertools import product

from equations import HeatEquation, WaveEquation, EquationLookUp
from config import EQUATION_KEYS, EQUATION_VALUES, EQUATIONS

EQUATION_TIME = {
    "heat": 0.02,
    "wave": 10
}

def make_evolution_video(spatial_resolution = 64 * 64, time_resolution = 100):
    total = len(EQUATIONS) * len(EQUATION_VALUES)
    for i,(equation, v) in enumerate(product(EQUATIONS,EQUATION_VALUES)):
        print(f"making video {i+1}/{total}")
        
        T = EQUATION_TIME[equation]
        Equation = EquationLookUp[equation]
        x, y = torch.meshgrid(
            torch.linspace(-1, 1, int(np.sqrt(spatial_resolution))),
            torch.linspace(-1, 1, int(np.sqrt(spatial_resolution)))
        )
        k = EQUATION_KEYS[equation]
        fig, ax = plt.subplots(figsize=(6,6))
        he = Equation(**{k:v})
        mat = ax.matshow(he(0, x, y).detach().numpy(), cmap='jet')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(mat, cax=cax, orientation='vertical')
        def animate(i):
            t = i * T / time_resolution
            u = he(t, x, y)
            ax.set_title(f"{equation} equation, {k}={v}, t={t:.5f}")
            mat.set_data(u.detach().numpy())
            mat.set_clim(u.min(),u.max())
            cbar.update_normal(mat)
            return mat,
        
        anim = FuncAnimation(fig, animate, frames=tqdm(range(time_resolution), desc="Frame"), interval=100, blit=True)
        os.makedirs(f"videos", exist_ok=True)
        appendix = "mp4" if 'ffmpeg' in animation.writers.list() else 'gif'
        anim.save(f"videos/{equation}_{k}={v}.{appendix}")
        plt.close(fig=fig)

if __name__ == '__main__':
    for equation in EQUATIONS:
        make_evolution_video(equation)
    


   
