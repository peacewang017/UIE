# -*- coding: utf-8 -*-
"""
LSAC3 with precomputed depth weights.

Main speedup:
- Depth is fixed across iterations, so the 5-point depth weights are
  precomputed once and reused for every LSAC iteration and channel.
"""

import cv2
import numpy as np
import os
import sys


def precompute_depth_weights(depth, sigma_d=0.08):
    """
    Precompute 5-point depth weights for a fixed depth map.
    Returns wc, wu, wd, wl, wr, den, each shape (H, W).
    """
    depth = depth.astype(np.float32)
    depth_pad = cv2.copyMakeBorder(depth, 1, 1, 1, 1, borderType=cv2.BORDER_REFLECT)

    dc = depth_pad[1:-1, 1:-1]
    du = depth_pad[:-2, 1:-1]
    dd = depth_pad[2:, 1:-1]
    dl = depth_pad[1:-1, :-2]
    dr = depth_pad[1:-1, 2:]

    sigma2 = 2.0 * (sigma_d ** 2)
    wc = np.ones_like(dc, dtype=np.float32)
    wu = np.exp(-((dc - du) ** 2) / sigma2).astype(np.float32)
    wd = np.exp(-((dc - dd) ** 2) / sigma2).astype(np.float32)
    wl = np.exp(-((dc - dl) ** 2) / sigma2).astype(np.float32)
    wr = np.exp(-((dc - dr) ** 2) / sigma2).astype(np.float32)
    den = wc + wu + wd + wl + wr + 1e-8
    return wc, wu, wd, wl, wr, den


def depth_weighted_five_point_precomputed(prev, weights):
    wc, wu, wd, wl, wr, den = weights

    center = prev[1:-1, 1:-1]
    up     = prev[:-2, 1:-1]
    down   = prev[2:, 1:-1]
    left   = prev[1:-1, :-2]
    right  = prev[1:-1, 2:]

    num = wc * center + wu * up + wd * down + wl * left + wr * right
    return num / den


def LSAC(Ib, Ig, Ir, weights, blockSize, ab, ag, ar, p):
    print("LSAC start")
    oldab = ab
    oldag = ag
    oldar = ar

    imgbDark = depth_weighted_five_point_precomputed(ab, weights)
    imggDark = depth_weighted_five_point_precomputed(ag, weights)
    imgrDark = depth_weighted_five_point_precomputed(ar, weights)

    imgab = (Ib * p) + (imgbDark * (1 - p))
    imgag = (Ig * p) + (imggDark * (1 - p))
    imgar = (Ir * p) + (imgrDark * (1 - p))

    imgbMiddle = cv2.copyMakeBorder(imgab, 1, 1, 1, 1, borderType=cv2.BORDER_REFLECT)
    imggMiddle = cv2.copyMakeBorder(imgag, 1, 1, 1, 1, borderType=cv2.BORDER_REFLECT)
    imgrMiddle = cv2.copyMakeBorder(imgar, 1, 1, 1, 1, borderType=cv2.BORDER_REFLECT)

    lossb = np.abs(np.sum(imgab) - np.sum(oldab)) / (imgab.shape[0] * imgab.shape[1])
    lossg = np.abs(np.sum(imgag) - np.sum(oldag)) / (imgab.shape[0] * imgab.shape[1])
    lossr = np.abs(np.sum(imgar) - np.sum(oldar)) / (imgab.shape[0] * imgab.shape[1])

    return imgbMiddle, imggMiddle, imgrMiddle, imgab, imgag, imgar, lossb, lossg, lossr


if len(sys.argv) < 2:
    raise ValueError("Usage: python LSAC3_precomputed.py <input_image_path>")

path = sys.argv[1]
img = cv2.imread(path)
if img is None:
    raise ValueError("Could not read image: {}".format(path))

prefix = os.path.splitext(os.path.basename(path))[0]
depth_path = os.path.join("OutputImages", prefix + "_depth_map.jpg")
depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
if depth is None:
    raise ValueError("Depth map not readable")

depth = depth.astype(np.float32) / 255.0
os.makedirs("OutputImages", exist_ok=True)

(Ib, Ig, Ir) = cv2.split(img)
Ib = Ib.astype(np.float32)
Ig = Ig.astype(np.float32)
Ir = Ir.astype(np.float32)

initab = cv2.copyMakeBorder(Ib, 1, 1, 1, 1, cv2.BORDER_REFLECT)
initag = cv2.copyMakeBorder(Ig, 1, 1, 1, 1, cv2.BORDER_REFLECT)
initar = cv2.copyMakeBorder(Ir, 1, 1, 1, 1, cv2.BORDER_REFLECT)

sigma_d = 0.003
weights = precompute_depth_weights(depth, sigma_d=sigma_d)

total_lossb = []
total_lossg = []
total_lossr = []

for i in range(1000):
    if i == 0:
        imgbMiddle, imggMiddle, imgrMiddle, imgab, imgag, imgar, lossb, lossg, lossr = LSAC(
            Ib, Ig, Ir, weights, 2, initab, initag, initar, 0.05
        )
    else:
        imgbMiddle, imggMiddle, imgrMiddle, imgab, imgag, imgar, lossb, lossg, lossr = LSAC(
            Ib, Ig, Ir, weights, 2, imgbMiddle, imggMiddle, imgrMiddle, 0.05
        )

    total_lossb.append(lossb)
    total_lossg.append(lossg)
    total_lossr.append(lossr)

    if max(lossb, lossg, lossr) < 0.001:
        break

imgdark = cv2.merge([imgab, imgag, imgar]).astype(np.uint8)
cv2.imwrite(os.path.join("OutputImages", prefix + "_lsac.jpg"), imgdark)
