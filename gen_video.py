"""
    visualize the evolution of the equations over time
"""
import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import animation
from matplotlib.animation  import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from itertools import product


from equations import HeatEquation, WaveEquation, EquationLookUp
from config import EQUATION_KEY, EQUATION_VALUES, EQUATIONS

EQUATION_TIME = {
    "heat": 0.02,
    "wave": 10,
    "poisson": 1 # it doesn't matter, should be none zero
}

def generate_video(spatial_resolution = 64 * 64, time_resolution = 100, use_gif=False):
    total = len(EQUATIONS) * len(EQUATION_VALUES)
    for i,(equation, v) in enumerate(product(EQUATIONS,EQUATION_VALUES)):
        print(f"\nmaking video {i+1}/{total}")
        
        T = EQUATION_TIME[equation]
        Equation = EquationLookUp[equation]
        x, y = torch.meshgrid(
            torch.linspace(-1, 1, int(np.sqrt(spatial_resolution))),
            torch.linspace(-1, 1, int(np.sqrt(spatial_resolution)))
        )
        k = EQUATION_KEY[equation]
        if equation == "poisson":
            # not dependent on time
            fig, ax = plt.subplots(ncols=2, figsize=(12,6))
            eq = Equation(**{k:v})
            ax[0].matshow(eq(0, x, y).detach().numpy(), cmap='jet')
            ax[1].matshow(eq(T, x, y).detach().numpy(), cmap='jet')
            ax[0].set_title("f")
            ax[1].set_title("u")
            os.makedirs(f"videos", exist_ok=True)
            fig.savefig(f"videos/{equation}_{k}={v}.png")
        else:
            fig, ax = plt.subplots(figsize=(6,6))
            eq = Equation(**{k:v})
            mat = ax.matshow(eq(0, x, y).detach().numpy(), cmap='jet')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(mat, cax=cax, orientation='vertical')
            def animate(i):
                t = i * T / time_resolution
                u = eq(t, x, y)
                ax.set_title(f"{equation} equation, {k}={v}, t={t:.5f}")
                mat.set_data(u.detach().numpy())
                mat.set_clim(u.min(),u.max())
                cbar.update_normal(mat)
                return mat,
            
            anim = FuncAnimation(fig, animate, frames=tqdm(range(time_resolution), desc="Frame"), interval=100, blit=True)
            os.makedirs(f"videos", exist_ok=True)
            appendix = "mp4" if 'ffmpeg' in animation.writers.list() and not use_gif else 'gif'
            anim.save(f"videos/{equation}_{k}={v}.{appendix}")
        plt.close(fig=fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-gif", "--use_gif", action="store_true", help="whether generate video using gif, default is mp4")
    args = parser.parse_args()
    generate_video(use_gif=args.use_gif)
    


   
