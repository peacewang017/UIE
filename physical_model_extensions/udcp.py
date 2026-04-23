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

# --------------------------------------------------------------------------
# UDCP Model Implementation Functions
# --------------------------------------------------------------------------

def get_udcp_backscatter(img, percent=0.001):
    """
    UDCP: Estimates background light using only Blue and Green channels.
    """
    # Only use B and G channels for dark channel calculation
    bg_img = img[:, :, :2] 
    dark_channel = np.min(bg_img, axis=2)
    
    # Get top 0.1% brightest pixels in the UDCP dark channel
    num_pixels = dark_channel.size
    num_brightest = max(1, int(num_pixels * percent))
    indices = np.argpartition(dark_channel.flatten(), -num_brightest)[-num_brightest:]
    
    # Background light A is the mean of these pixels in the original image
    flat_img = img.reshape(-1, 3)
    A = np.mean(flat_img[indices], axis=0)
    return A

def restore_udcp(img, depth_map):
    """
    UDCP Pipeline: Standard dehazing using UDCP-based background light.
    """
    img_norm = img / 255.0
    A = get_udcp_backscatter(img) / 255.0
    
    # t(z) = exp(-beta * z)
    beta_udcp = 1.0
    t = np.exp(-beta_udcp * (depth_map / 255.0))
    t = np.clip(t, 0.1, 0.9)
    
    res = np.zeros_like(img_norm)
    for i in range(3):
        res[:, :, i] = (img_norm[:, :, i] - A[i]) / t + A[i]
        
    return (np.clip(res, 0, 1) * 255).astype(np.uint8)