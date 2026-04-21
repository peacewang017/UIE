# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np

img = None


def LSAC(Ib, Ig, Ir, blockSize, ab, ag, ar, p):
    global img
    addSize = int(blockSize / 2)
    newHeight = img.shape[0] + blockSize
    newWidth = img.shape[1] + blockSize

    oldab = ab
    oldag = ag
    oldar = ar
    imgbMiddle = np.zeros((newHeight, newWidth))
    imggMiddle = np.zeros((newHeight, newWidth))
    imgrMiddle = np.zeros((newHeight, newWidth))

    imgbDark = np.zeros((img.shape[0], img.shape[1]))
    imggDark = np.zeros((img.shape[0], img.shape[1]))
    imgrDark = np.zeros((img.shape[0], img.shape[1]))

    imgab = np.zeros((img.shape[0], img.shape[1]))
    imgag = np.zeros((img.shape[0], img.shape[1]))
    imgar = np.zeros((img.shape[0], img.shape[1]))
    for i in range(addSize, newHeight - addSize):
        for j in range(addSize, newWidth - addSize):
            a = 5
            if i - addSize == 0 or i + addSize == newHeight - 1:
                a -= 1
            if j - addSize == 0 or j + addSize == newWidth - 1:
                a -= 1
            imgbDark[i - addSize, j - addSize] = (ab[i, j] + ab[i - 1, j] + ab[i + 1, j] + ab[i, j - 1] + ab[i, j + 1]) / a
            imggDark[i - addSize, j - addSize] = (ag[i, j] + ag[i - 1, j] + ag[i + 1, j] + ag[i, j - 1] + ag[i, j + 1]) / a
            imgrDark[i - addSize, j - addSize] = (ar[i, j] + ar[i - 1, j] + ar[i + 1, j] + ar[i, j - 1] + ar[i, j + 1]) / a
    imgab[:, :] = (Ib[:, :] * p) + (imgbDark[:, :] * (1 - p))
    imgag[:, :] = (Ig[:, :] * p) + (imggDark[:, :] * (1 - p))
    imgar[:, :] = (Ir[:, :] * p) + (imgrDark[:, :] * (1 - p))

    imgbMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = imgab
    imggMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = imgag
    imgrMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = imgar
    lossb = np.abs(np.sum(imgab) - np.sum(oldab)) / (imgab.shape[0] * imgab.shape[1])
    lossg = np.abs(np.sum(imgag) - np.sum(oldag)) / (imgab.shape[0] * imgab.shape[1])
    lossr = np.abs(np.sum(imgar) - np.sum(oldar)) / (imgab.shape[0] * imgab.shape[1])

    return imgbMiddle, imggMiddle, imgrMiddle, imgab, imgag, imgar, lossb, lossg, lossr


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not read input image: {input_path}")

    initab = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
    initag = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
    initar = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
    (Ib, Ig, Ir) = cv2.split(img)

    total_lossb = np.array([])
    total_lossg = np.array([])
    total_lossr = np.array([])
    for i in range(1000):
        if i == 0:
            imgbMiddle, imggMiddle, imgrMiddle, imgab, imgag, imgar, lossb, lossg, lossr = LSAC(Ib, Ig, Ir, 2, initab, initag, initar, 0.001)
        else:
            imgbMiddle, imggMiddle, imgrMiddle, imgab, imgag, imgar, lossb, lossg, lossr = LSAC(Ib, Ig, Ir, 2, imgbMiddle, imggMiddle, imgrMiddle, 0.001)
        total_lossb = np.append(total_lossb, lossb)
        total_lossg = np.append(total_lossg, lossg)
        total_lossr = np.append(total_lossr, lossr)

    imgdark = cv2.merge([imgab, imgag, imgar])
    imgdark = imgdark * 2
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, imgdark)
