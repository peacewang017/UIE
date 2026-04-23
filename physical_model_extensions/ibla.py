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
# IBLA Model Implementation Functions
# --------------------------------------------------------------------------

def estimate_blurriness(img):
    """
    IBLA: Estimates image blurriness using the local variance of Laplacian.
    Blurrier areas often correspond to deeper water.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Calculate local variance as blurriness proxy
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # Use a Gaussian blur to smooth the blurriness map
    blur_map = cv2.GaussianBlur(np.abs(laplacian), (15, 15), 0)
    # Normalize to 0-1
    blur_map = (blur_map - np.min(blur_map)) / (np.max(blur_map) - np.min(blur_map) + 1e-6)
    return 1.0 - blur_map # High value means high blur (potentially far away)

def restore_ibla(img, depth_map):
    """
    IBLA Pipeline: Combines Depth and Blurriness to guide restoration.
    """
    img_norm = img / 255.0
    z = depth_map / 255.0
    blur_map = estimate_blurriness(img)
    
    # Combined factor: In IBLA, both distance and blur guide the correction
    # Higher blur or higher depth = more absorption
    combined_depth = (z + blur_map) / 2.0
    
    # Typical underwater absorption coefficients
    eta = [0.12, 0.18, 0.5] 
    
    res = np.zeros_like(img_norm)
    for i in range(3):
        # Inverse of absorption
        compensation = np.exp(eta[i] * combined_depth)
        res[:, :, i] = img_norm[:, :, i] * compensation
        
    return (np.clip(res, 0, 1) * 255).astype(np.uint8)