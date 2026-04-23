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

def dehaze_wcid(img, depth_map):
    """
    WCID Stage 1: Image Dehazing.
    Uses depth-guided transmission estimation to remove haze caused by scattering.
    """
    img_norm = img / 255.0
    
    # Estimate background light (A) 
    # Simplified: taking the average of the top 0.5% brightest pixels
    A = np.percentile(img_norm, 99.5, axis=(0, 1))
    
    # Estimate transmission t(x) = exp(-beta * depth)
    # beta_dehaze is the scattering coefficient
    beta_dehaze = 1.0 
    t = np.exp(-beta_dehaze * (depth_map / 255.0))
    t = np.clip(t, 0.1, 0.9) # Clamp to avoid noise amplification
    
    dehazed = np.zeros_like(img_norm)
    for i in range(3):
        # Restoration formula: J = (I - A)/t + A
        dehazed[:, :, i] = (img_norm[:, :, i] - A[i]) / t + A[i]
    
    dehazed = np.clip(dehazed, 0, 1) * 255.0
    return dehazed

def wavelength_compensation_wcid(img_dehazed, depth_map):
    """
    WCID Stage 2: Wavelength Compensation.
    Compensates for the energy loss of different wavelengths (mainly Red) due to absorption.
    """
    img_norm = img_dehazed / 255.0
    z = depth_map / 255.0
    
    # Absorption coefficients for Blue, Green, and Red channels
    # Red has the highest attenuation in water
    eta = [0.1, 0.15, 0.4] # Typical coefficients for [B, G, R]
    
    compensated = np.zeros_like(img_norm)
    for i in range(3):
        # Compensation formula: J = I * exp(eta * z)
        # Objects further away (higher z) receive more gain
        compensation_factor = np.exp(eta[i] * z)
        compensated[:, :, i] = img_norm[:, :, i] * compensation_factor
        
    compensated = np.clip(compensated, 0, 1) * 255.0
    return compensated.astype(np.uint8)

np.seterr(over='ignore')