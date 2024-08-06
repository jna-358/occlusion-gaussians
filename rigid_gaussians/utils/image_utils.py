#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import matplotlib.cm
import torch
import matplotlib
import numpy as np

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def to_colormap(grayscale, cmap="inferno"):
    # Check if the input is a tensor
    if isinstance(grayscale, torch.Tensor):
        grayscale = grayscale.detach().cpu().numpy()
    grayscale = grayscale.astype(np.float32)

    # Normalize the input
    grayscale = (grayscale - grayscale.min()) / (grayscale.max() - grayscale.min())

    # Apply the colormap
    cmap_mpl = matplotlib.colormaps[cmap]
    rgb = cmap_mpl(grayscale)[:, :, :3]

    return rgb

def discrete_colors(index, colormap="Set1"):
    cmap = matplotlib.colormaps[colormap]
    color = cmap(index % cmap.N)
    return np.array([color[i] for i in range(3)])