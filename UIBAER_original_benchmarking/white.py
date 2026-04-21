# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np


def white_balance_5(img):
    b, g, r = cv2.split(img)

    def con_num(x):
        if x > 0:
            return 1
        if x < 0:
            return -1
        return 0

    yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    (y, u, v) = cv2.split(yuv_img)
    m, n = y.shape
    max_y = np.max(y.flatten())

    avl_u = np.mean(u)
    avl_v = np.mean(v)
    avl_du = np.mean(np.abs(u - avl_u))
    avl_dv = np.mean(np.abs(v - avl_v))

    num_y = np.zeros(y.shape)
    yhistogram = np.zeros(256)
    ysum = 0
    radio = 0.5

    for i in range(m):
        for j in range(n):
            value = 0
            if (
                np.abs(u[i][j] - (avl_u + avl_du * con_num(avl_u))) < radio * avl_du
                or np.abs(v[i][j] - (avl_v + avl_dv * con_num(avl_v))) < radio * avl_dv
            ):
                value = 1
            if value <= 0:
                continue
            num_y[i][j] = y[i][j]
            yhistogram[int(num_y[i][j])] += 1
            ysum += 1

    Y = 255
    num = 0
    key = 0
    while Y >= 0:
        num += yhistogram[Y]
        if num > 0.01 * ysum:
            key = Y
            break
        Y -= 1

    sum_r = sum_g = sum_b = num_rgb = 0
    for i in range(m):
        for j in range(n):
            if num_y[i][j] > key:
                sum_r += r[i][j]
                sum_g += g[i][j]
                sum_b += b[i][j]
                num_rgb += 1

    avl_r = sum_r / num_rgb
    avl_g = sum_g / num_rgb
    avl_b = sum_b / num_rgb

    for i in range(m):
        for j in range(n):
            b_point = int(b[i][j]) * int(max_y) / avl_b
            g_point = int(g[i][j]) * int(max_y) / avl_g
            r_point = int(r[i][j]) * int(max_y) / avl_r
            b[i][j] = np.clip(b_point, 0, 255)
            g[i][j] = np.clip(g_point, 0, 255)
            r[i][j] = np.clip(r_point, 0, 255)

    return cv2.merge([b, g, r])


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not read input image: {input_path}")

    whiteimg = white_balance_5(img)

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = base + "_white" + (ext or ".jpg")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, whiteimg)
