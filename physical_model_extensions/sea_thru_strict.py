import os
import numpy as np
import cv2
import natsort
import matplotlib.pyplot as plt
from scipy import optimize
import datetime
import subprocess
import skimage 
import sys
from scipy.optimize import curve_fit

OUTPUT_DIR = "OutputImages"
CURRENT_PREFIX = ""
depth_map = None

class Node(object):
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value

    def printInfo(self):
        print(self.x, self.y, self.value)

def direct_signal_seathru(img_no_b, depth_map, illumination_map):
    """
    Sea-thru Attenuation Restoration:
    J = (I - B) / exp(-beta_D * z)
    """
    # Normalize
    I_minus_B = img_no_b / 255.0
    z = depth_map / 255.0
    
    # Sea-thru suggests beta_D (attenuation) is different from beta_B
    # Here we use the illumination map to guide the range, 
    # but strictly following the exponential decay J = D / e^(-beta*z)
    
    # Simplified estimation of beta_D per channel
    # In a full Sea-thru implementation, this requires iterative optimization
    beta_D = [0.15, 0.08, 0.35] # Typical coeffs for B, G, R in clearish water
    
    J = np.zeros_like(I_minus_B)
    for i in range(3):
        # Apply restoration
        transmission = np.exp(-beta_D[i] * z)
        J[:, :, i] = I_minus_B[:, :, i] / np.clip(transmission, 0.1, 1.0)

    # Apply illumination map scaling (LSAC)
    estill = illumination_map / 255.0
    J = J / np.clip(estill, 0.1, 1.0)
    
    J = np.clip(J, 0, 1) * 255.0
    J = J.astype(np.uint8)
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, CURRENT_PREFIX + "_jc_seathru.jpg"), J)
    return J

def backscatter_seathru(img, depth_map, percent=0.01):
    """
    Implementation inspired by Sea-thru's backscatter estimation.
    Sea-thru estimates backscatter (B) as a function of depth (z):
    B(z) = B_inf * (1 - exp(-beta_B * z))
    """
    height, width, _ = img.shape
    size = height * (width // 10)
    
    # Selecting dark pixels across 10 spatial bins to estimate backscatter
    nodes_list = [[] for _ in range(10)]
    w_step = width // 10
    for i in range(height):
        for b in range(10):
            for j in range(b * w_step, (b + 1) * w_step):
                # Using sum of RGB to find dark pixels (consistent with your original logic)
                val = np.sum(img[i, j].astype(np.float32))
                nodes_list[b].append(Node(i, j, val))

    # Lists to store sampled data for fitting
    sampled_depths = []
    sampled_B_B = [] # Channel 0
    sampled_B_G = [] # Channel 1
    sampled_B_R = [] # Channel 2

    for b in range(10):
        # Sort by value to find the darkest pixels in this spatial bin
        nodes_list[b] = sorted(nodes_list[b], key=lambda node: node.value)
        for i in range(int(percent * size)):
            node = nodes_list[b][i]
            # Normalize to [0, 1] for fitting
            sampled_depths.append(depth_map[node.x, node.y] / 255.0)
            sampled_B_B.append(img[node.x, node.y, 0] / 255.0)
            sampled_B_G.append(img[node.x, node.y, 1] / 255.0) # FIXED: Now appends the pixel value
            sampled_B_R.append(img[node.x, node.y, 2] / 255.0) # FIXED: Now appends the pixel value
    
    # Sea-thru model: B(z) = b_inf * (1 - exp(-beta_b * z))
    def seathru_b_model(z, b_inf, beta_b):
        return b_inf * (1 - np.exp(-beta_b * z))

    def fit_b(z, channel_vals):
        # Initial guess p0=[max_val, 1.0]
        params, _ = optimize.curve_fit(seathru_b_model, z, channel_vals, p0=[np.max(channel_vals), 1.0], maxfev=5000)
        return params

    z_data = np.array(sampled_depths)
    b_params = []
    # Fit each channel (B, G, R)
    b_params.append(fit_b(z_data, np.array(sampled_B_B)))
    b_params.append(fit_b(z_data, np.array(sampled_B_G)))
    b_params.append(fit_b(z_data, np.array(sampled_B_R)))

    # Construct backscatter map
    B_map = np.zeros_like(img, dtype=np.float64)
    z_map = depth_map / 255.0
    for i in range(3):
        B_map[:, :, i] = seathru_b_model(z_map, b_params[i][0], b_params[i][1])

    # J = I - B
    bsrm = (img / 255.0) - B_map
    bsrm = np.clip(bsrm, 0, 1) * 255.0
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, CURRENT_PREFIX + "_seathru_backscatter.jpg"), bsrm)
    return bsrm, b_params