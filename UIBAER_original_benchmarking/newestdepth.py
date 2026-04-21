# -*- coding: utf-8 -*-
import os
import sys
import math
import numpy as np
import cv2
from sklearn.linear_model import LinearRegression

img = None


def getDarkChannel(Ib, Ig, Ir, blockSize, Sb, Sg, Sr, Wb, Wg, Wr):
    global img
    addSize = int((blockSize - 1) / 2)
    newHeight = img.shape[0] + blockSize - 1
    newWidth = img.shape[1] + blockSize - 1

    imgbMiddle = np.ones((newHeight, newWidth))
    imggMiddle = np.ones((newHeight, newWidth))
    imgrMiddle = np.ones((newHeight, newWidth))

    imgbMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = Ib
    imggMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = Ig
    imgrMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = Ir

    imgbDark = np.zeros((img.shape[0], img.shape[1]))
    imggDark = np.zeros((img.shape[0], img.shape[1]))
    imgrDark = np.zeros((img.shape[0], img.shape[1]))

    for i in range(addSize, newHeight - addSize):
        for j in range(addSize, newWidth - addSize):
            localMinb = 1
            localMing = 1
            localMinr = 1
            for k in range(i - addSize, i + addSize + 1):
                for l in range(j - addSize, j + addSize + 1):
                    itemb = imgbMiddle.item((k, l))
                    itemg = imggMiddle.item((k, l))
                    itemr = imgrMiddle.item((k, l))
                    if 1 - Wb * abs(Sb - itemb) < localMinb:
                        localMinb = 1 - Wb * abs(Sb - itemb)
                    elif 1 - Wg * abs(Sg - itemg) < localMing:
                        localMing = 1 - Wg * abs(Sg - itemg)
                    elif 1 - Wr * abs(Sr - itemr) < localMing:
                        localMinr = 1 - Wr * abs(Sr - itemr)
            imgbDark[i - addSize, j - addSize] = localMinb
            imggDark[i - addSize, j - addSize] = localMing
            imgrDark[i - addSize, j - addSize] = localMinr
    return imgbDark, imggDark, imgrDark


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not read input image: {input_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Sobel
    #kernelx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    #kernely = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    #sobelx = cv2.filter2D(gray, -1, kernelx)
    #sobely = cv2.filter2D(gray, -1, kernely)
    #g = abs(sobelx) + abs(sobely)

    # prewitt
    kernelx = np.array([[1, 0, -1],
                        [1, 0, -1],
                        [1, 0, -1]], dtype=np.float32)

    kernely = np.array([[1, 1, 1],
                        [0, 0, 0],
                        [-1, -1, -1]], dtype=np.float32)

    PREWITTX = cv2.filter2D(gray, cv2.CV_32F, kernelx)
    PREWITTY = cv2.filter2D(gray, cv2.CV_32F, kernely)

    g = np.abs(PREWITTX) + np.abs(PREWITTY)
    g = np.clip(g, 0, 255).astype(np.uint8)

    window = np.ones((7, 7), np.uint8)
    window2 = np.ones((8, 8), np.uint8)
    img_dilation = cv2.dilate(g, window, iterations=1)
    img_erosion = cv2.erode(img_dilation, window2, iterations=1)
    img_median = cv2.medianBlur(img_erosion, 5)

    a = np.max(img_median) - np.min(img_median)
    norm = np.zeros(img_median.shape)
    for i in range(img_median.shape[0]):
        for j in range(img_median.shape[1]):
            norm[i][j] = (img_median[i][j] - np.min(img_median)) / a

    Dr = np.around((1 - norm) * 255)

    regDr = np.array(Dr).reshape(Dr.shape[0] * Dr.shape[1], 1) / 255
    (Ib, Ig, Ir) = cv2.split(img)
    regIb = np.array(Ib).reshape(Ib.shape[0] * Ib.shape[1], 1) / 255
    regIg = np.array(Ig).reshape(Ig.shape[0] * Ig.shape[1], 1) / 255
    regIr = np.array(Ir).reshape(Ir.shape[0] * Ir.shape[1], 1) / 255

    modelr = LinearRegression().fit(regDr, regIr)
    modelg = LinearRegression().fit(regDr, regIg)
    modelb = LinearRegression().fit(regDr, regIb)
    Ar = float(modelr.coef_.ravel()[0])
    Ag = float(modelg.coef_.ravel()[0])
    Ab = float(modelb.coef_.ravel()[0])

    Sr = 1 if Ar > 0 else 0
    Sg = 1 if Ag > 0 else 0
    Sb = 1 if Ab > 0 else 0

    Wr = math.tanh(4 * abs(Ar))
    Wg = math.tanh(4 * abs(Ag))
    Wb = math.tanh(4 * abs(Ab))
    imgbdark, imggdark, imgrdark = getDarkChannel(Ib / 255, Ig / 255, Ir / 255, 5, Sb, Sg, Sr, Wb, Wg, Wr)
    imgdark = cv2.merge([imgbdark, imggdark, imgrdark])
    newimgdark = imgdark * 255

    Dd = np.full((img.shape[0], img.shape[1]), 255.0)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(3):
                if newimgdark[i, j, k] < Dd[i, j]:
                    Dd[i, j] = newimgdark[i, j, k]

    Dd = Dd.astype(np.uint8)
    Dd_median = cv2.medianBlur(Dd, 5)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, Dd_median)
