import cv2
import numpy as np
import os
import sys

# Weight the averaging window by the depth map
def depth_weighted_five_point(prev, depth, sigma_d=0.08):
    # pad depth to match prev
    depth_pad = cv2.copyMakeBorder(depth, 1, 1, 1, 1, borderType=cv2.BORDER_REFLECT)

    center = prev[1:-1, 1:-1]
    up     = prev[:-2, 1:-1]
    down   = prev[2:, 1:-1]
    left   = prev[1:-1, :-2]
    right  = prev[1:-1, 2:]

    dc = depth_pad[1:-1, 1:-1]
    du = depth_pad[:-2, 1:-1]
    dd = depth_pad[2:, 1:-1]
    dl = depth_pad[1:-1, :-2]
    dr = depth_pad[1:-1, 2:]

    wc = np.ones_like(dc, dtype=np.float32)
    wu = np.exp(-((dc - du) ** 2) / (2 * sigma_d ** 2))
    wd = np.exp(-((dc - dd) ** 2) / (2 * sigma_d ** 2))
    wl = np.exp(-((dc - dl) ** 2) / (2 * sigma_d ** 2))
    wr = np.exp(-((dc - dr) ** 2) / (2 * sigma_d ** 2))

    num = wc * center + wu * up + wd * down + wl * left + wr * right
    den = wc + wu + wd + wl + wr + 1e-8
    return num / den

def LSAC(Ib, Ig, Ir, depth, blockSize,ab,ag,ar,p, sigma_d = 0.08):
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

    #imgbDark[:, :] = (center_b + up_b + down_b + left_b + right_b) / 5.0
    #imggDark[:, :] = (center_g + up_g + down_g + left_g + right_g) / 5.0
    #imgrDark[:, :] = (center_r + up_r + down_r + left_r + right_r) / 5.0
    
    # First improvement: use depth map to weight averaging 
    imgbDark[:, :] = depth_weighted_five_point(ab, depth, sigma_d=0.03)
    imggDark[:, :] = depth_weighted_five_point(ag, depth, sigma_d=0.03)
    imgrDark[:, :] = depth_weighted_five_point(ar, depth, sigma_d=0.03)


    imgab[:,:] = (Ib[:,:]*p)+(imgbDark[:,:]*(1-p))
    imgag[:,:] = (Ig[:,:]*p)+(imggDark[:,:]*(1-p))
    imgar[:,:] = (Ir[:,:]*p)+(imgrDark[:,:]*(1-p))
    
    imgbMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = imgab
    imggMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = imgag
    imgrMiddle[addSize:newHeight - addSize, addSize:newWidth - addSize] = imgar

    # second improvement: reflect pad to avoid dark borders. Everythign outside actual image is black so during averaging on borders, neighbors are zero valued. 
    # This artificially pulls numbers down and since there are 1000 iterations the bias diffuses inward
    imgbMiddle = cv2.copyMakeBorder(imgab, 1, 1, 1, 1, borderType=cv2.BORDER_REFLECT)
    imggMiddle = cv2.copyMakeBorder(imgag, 1, 1, 1, 1, borderType=cv2.BORDER_REFLECT)
    imgrMiddle = cv2.copyMakeBorder(imgar, 1, 1, 1, 1, borderType=cv2.BORDER_REFLECT)

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
# Load depth map, we can weigh the moving average filter 
depth_path = os.path.join("OutputImages", prefix + "_depth_map.jpg")
depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
if depth is None:
    raise ValueError("Depth map not readable")

depth = depth.astype(np.float32)/255.0

os.makedirs("OutputImages", exist_ok=True)

initab = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
initag = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
initar = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
(Ib,Ig,Ir) = cv2.split(img)
# Fourth potential improvement: do not start from zero padding, use image instead so LSAC loop doesn't propogate down
initab = cv2.copyMakeBorder(Ib.astype(np.float32), 1,1,1,1, cv2.BORDER_REFLECT)
initag = cv2.copyMakeBorder(Ig.astype(np.float32), 1,1,1,1, cv2.BORDER_REFLECT)
initar = cv2.copyMakeBorder(Ir.astype(np.float32), 1,1,1,1, cv2.BORDER_REFLECT)

s = 488
sig = 0.08
p = 1/((sig**2)*(s**2)+1)

# TODO lists instead of arrays will speed this part up
total_lossb = []#np.array([])
total_lossg = []#np.array([])
total_lossr = []#np.array([])
for i in range(1000):
    if i == 0:
        imgbMiddle,imggMiddle,imgrMiddle,imgab,imgag,imgar,lossb,lossg,lossr = LSAC(Ib,Ig,Ir, depth, 2,initab,initag,initar,0.05, sigma_d = 0.003)
        if max(lossb, lossg, lossr) < 0.001: # another improvement: Instead of arbitrary 1000 loops, use the loss
            break
        #total_lossb = np.array([lossb])
    else:
        imgbMiddle,imggMiddle,imgrMiddle,imgab,imgag,imgar,lossb,lossg,lossr = LSAC(Ib,Ig,Ir, depth, 2,imgbMiddle,imggMiddle,imgrMiddle,0.05, sigma_d = 0.003)
        if max(lossb, lossg, lossr) < 0.001: # another improvement: Instead of arbitrary 1000 loops, use the loss
            break
    total_lossb.append(lossb)
    total_lossg.append(lossg)
    total_lossr.append(lossr)
    #print(total_lossb,total_lossg,total_lossr)
total_lossb = np.array(total_lossb)
total_lossg = np.array(total_lossg)
total_lossr = np.array(total_lossr)
    
imgdark = cv2.merge([imgab,imgag,imgar])
#imgdark = imgdark*2 Not sure why they just multiply by 2
#cv2.imwrite('D:/DCP/LSAregg3374.jpg',imgdark)
cv2.imwrite(os.path.join("OutputImages", prefix + "_lsac.jpg"), imgdark)