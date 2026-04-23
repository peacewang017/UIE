# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 13:43:26 2020

@author: swz84
"""

import cv2
import numpy as np
import os
import sys

def LSAC(Ib, Ig, Ir, blockSize,ab,ag,ar,p):
    print("LSAC start")
    addSize = int(blockSize / 2)
    newHeight = img.shape[0] + blockSize
    newWidth = img.shape[1] + blockSize
    
    # redundant 
    #oldab = np.zeros((newHeight, newWidth))
    #oldag = np.zeros((newHeight, newWidth))
    #oldar = np.zeros((newHeight, newWidth))
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
    #for i in range(addSize, newHeight - addSize):
    #    for j in range(addSize, newWidth - addSize):
    #        a = 5
    #        if i-addSize == 0 or i+addSize == newHeight-1:
    #            a = a-1
    #        if j-addSize == 0 or j+addSize == newWidth-1:
    #            a = a-1      
    #        imgbDark[i - addSize, j - addSize] = (ab[i,j]+ab[i-1,j]+ab[i+1,j]+ab[i,j-1]+ab[i,j+1])/a
    #        imggDark[i - addSize, j - addSize] = (ag[i,j]+ag[i-1,j]+ag[i+1,j]+ag[i,j-1]+ag[i,j+1])/a
    #        imgrDark[i - addSize, j - addSize] = (ar[i,j]+ar[i-1,j]+ar[i+1,j]+ar[i,j-1]+ar[i,j+1])/a
    center_b = ab[1:-1, 1:-1]
    up_b     = ab[:-2, 1:-1]
    down_b   = ab[2:, 1:-1]
    left_b   = ab[1:-1, :-2]
    right_b  = ab[1:-1, 2:]

    center_g = ag[1:-1, 1:-1]
    up_g     = ag[:-2, 1:-1]
    down_g   = ag[2:, 1:-1]
    left_g   = ag[1:-1, :-2]
    right_g  = ag[1:-1, 2:]

    center_r = ar[1:-1, 1:-1]
    up_r     = ar[:-2, 1:-1]
    down_r   = ar[2:, 1:-1]
    left_r   = ar[1:-1, :-2]
    right_r  = ar[1:-1, 2:]

    imgbDark[:, :] = (center_b + up_b + down_b + left_b + right_b) / 5.0
    imggDark[:, :] = (center_g + up_g + down_g + left_g + right_g) / 5.0
    imgrDark[:, :] = (center_r + up_r + down_r + left_r + right_r) / 5.0
    
    imgab[:,:] = (Ib[:,:]*p)+(imgbDark[:,:]*(1-p))
    imgag[:,:] = (Ig[:,:]*p)+(imggDark[:,:]*(1-p))
    imgar[:,:] = (Ir[:,:]*p)+(imgrDark[:,:]*(1-p))
    
    imgbMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = imgab
    imggMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = imgag
    imgrMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = imgar
    lossb = np.abs(np.sum(imgab)-np.sum(oldab))/(imgab.shape[0]*imgab.shape[1])
    lossg = np.abs(np.sum(imgag)-np.sum(oldag))/(imgab.shape[0]*imgab.shape[1])
    lossr = np.abs(np.sum(imgar)-np.sum(oldar))/(imgab.shape[0]*imgab.shape[1])
    
    return imgbMiddle,imggMiddle,imgrMiddle,imgab,imgag,imgar,lossb,lossg,lossr

#img = cv2.imread('D:/DCP/InputImages/LFT_3374.jpg')
if len(sys.argv) < 2:
    raise ValueError("Usage: python LSAC2.py <input_image_path>")

path = sys.argv[1]
img = cv2.imread(path)
if img is None:
    raise ValueError("Could not read image: {}".format(path))

prefix = os.path.splitext(os.path.basename(path))[0]
os.makedirs("OutputImages", exist_ok=True)

initab = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
initag = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
initar = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
(Ib,Ig,Ir) = cv2.split(img)
s = 488
sig = 0.08
p = 1/((sig**2)*(s**2)+1)

# TODO lists instead of arrays will speed this part up
total_lossb = []#np.array([])
total_lossg = []#np.array([])
total_lossr = []#np.array([])
for i in range(1000):
    if i == 0:
        imgbMiddle,imggMiddle,imgrMiddle,imgab,imgag,imgar,lossb,lossg,lossr = LSAC(Ib,Ig,Ir,2,initab,initag,initar,0.001)
        #total_lossb = np.array([lossb])
    else:
        imgbMiddle,imggMiddle,imgrMiddle,imgab,imgag,imgar,lossb,lossg,lossr = LSAC(Ib,Ig,Ir,2,imgbMiddle,imggMiddle,imgrMiddle,0.001)
    total_lossb.append(lossb)
    total_lossg.append(lossg)
    total_lossr.append(lossr)
    #print(total_lossb,total_lossg,total_lossr)
total_lossb = np.array(total_lossb)
total_lossg = np.array(total_lossg)
total_lossr = np.array(total_lossr)
    
imgdark = cv2.merge([imgab,imgag,imgar])
imgdark = imgdark*2
#cv2.imwrite('D:/DCP/LSAregg3374.jpg',imgdark)
cv2.imwrite(os.path.join("OutputImages", prefix + "_lsac.jpg"), imgdark) # To allow us to be end-to-end