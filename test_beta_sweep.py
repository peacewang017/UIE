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
import time

OUTPUT_DIR = "OutputImages"
CURRENT_PREFIX = ""
depth_map = None
CURRENT_VARIANT_SUFFIX = ""

GUIDED_BETAS = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80]


class Node(object):
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value

    def printInfo(self):
        print(self.x, self.y, self.value)


def format_beta(beta):
    return f"{beta:.2f}"


def variant_tag(beta):
    return f"_gbeta_{format_beta(beta)}"


def file_with_variant(prefix, stem, ext, beta):
    return os.path.join(OUTPUT_DIR, f"{prefix}_{stem}{variant_tag(beta)}.{ext}")


def backscatter(img, percent=0.01):
    print(img.shape)
    height = img.shape[0]
    width = int(img.shape[1] / 10)
    size = height * width

    nodes1 = []
    nodes2 = []
    nodes3 = []
    nodes4 = []
    nodes5 = []
    nodes6 = []
    nodes7 = []
    nodes8 = []
    nodes9 = []
    nodes10 = []

    for i in range(0, height):
        for j in range(0, width):
            oneNode = Node(i, j, sum(tuple(img[i, j])))
            nodes1.append(oneNode)
        for j in range(width, width * 2):
            oneNode = Node(i, j, sum(tuple(img[i, j])))
            nodes2.append(oneNode)
        for j in range(width * 2, width * 3):
            oneNode = Node(i, j, sum(tuple(img[i, j])))
            nodes3.append(oneNode)
        for j in range(width * 3, width * 4):
            oneNode = Node(i, j, sum(tuple(img[i, j])))
            nodes4.append(oneNode)
        for j in range(width * 4, width * 5):
            oneNode = Node(i, j, sum(tuple(img[i, j])))
            nodes5.append(oneNode)
        for j in range(width * 5, width * 6):
            oneNode = Node(i, j, sum(tuple(img[i, j])))
            nodes6.append(oneNode)
        for j in range(width * 6, width * 7):
            oneNode = Node(i, j, sum(tuple(img[i, j])))
            nodes7.append(oneNode)
        for j in range(width * 7, width * 8):
            oneNode = Node(i, j, sum(tuple(img[i, j])))
            nodes8.append(oneNode)
        for j in range(width * 8, width * 9):
            oneNode = Node(i, j, sum(tuple(img[i, j])))
            nodes9.append(oneNode)
        for j in range(width * 9, width * 10):
            oneNode = Node(i, j, sum(tuple(img[i, j])))
            nodes10.append(oneNode)

    nodes1 = sorted(nodes1, key=lambda node: node.value, reverse=False)
    nodes2 = sorted(nodes2, key=lambda node: node.value, reverse=False)
    nodes3 = sorted(nodes3, key=lambda node: node.value, reverse=False)
    nodes4 = sorted(nodes4, key=lambda node: node.value, reverse=False)
    nodes5 = sorted(nodes5, key=lambda node: node.value, reverse=False)
    nodes6 = sorted(nodes6, key=lambda node: node.value, reverse=False)
    nodes7 = sorted(nodes7, key=lambda node: node.value, reverse=False)
    nodes8 = sorted(nodes8, key=lambda node: node.value, reverse=False)
    nodes9 = sorted(nodes9, key=lambda node: node.value, reverse=False)
    nodes10 = sorted(nodes10, key=lambda node: node.value, reverse=False)

    imgR = []
    imgG = []
    imgB = []
    depth = []
    for i in range(0, int(percent * size)):
        imgB.append(img[nodes1[i].x, nodes1[i].y, 0])
        imgB.append(img[nodes2[i].x, nodes2[i].y, 0])
        imgB.append(img[nodes3[i].x, nodes3[i].y, 0])
        imgB.append(img[nodes4[i].x, nodes4[i].y, 0])
        imgB.append(img[nodes5[i].x, nodes5[i].y, 0])
        imgB.append(img[nodes6[i].x, nodes6[i].y, 0])
        imgB.append(img[nodes7[i].x, nodes7[i].y, 0])
        imgB.append(img[nodes8[i].x, nodes8[i].y, 0])
        imgB.append(img[nodes9[i].x, nodes9[i].y, 0])
        imgB.append(img[nodes10[i].x, nodes10[i].y, 0])
        imgG.append(img[nodes1[i].x, nodes1[i].y, 1])
        imgG.append(img[nodes2[i].x, nodes2[i].y, 1])
        imgG.append(img[nodes3[i].x, nodes3[i].y, 1])
        imgG.append(img[nodes4[i].x, nodes4[i].y, 1])
        imgG.append(img[nodes5[i].x, nodes5[i].y, 1])
        imgG.append(img[nodes6[i].x, nodes6[i].y, 1])
        imgG.append(img[nodes7[i].x, nodes7[i].y, 1])
        imgG.append(img[nodes8[i].x, nodes8[i].y, 1])
        imgG.append(img[nodes9[i].x, nodes9[i].y, 1])
        imgG.append(img[nodes10[i].x, nodes10[i].y, 1])
        imgR.append(img[nodes1[i].x, nodes1[i].y, 2])
        imgR.append(img[nodes2[i].x, nodes2[i].y, 2])
        imgR.append(img[nodes3[i].x, nodes3[i].y, 2])
        imgR.append(img[nodes4[i].x, nodes4[i].y, 2])
        imgR.append(img[nodes5[i].x, nodes5[i].y, 2])
        imgR.append(img[nodes6[i].x, nodes6[i].y, 2])
        imgR.append(img[nodes7[i].x, nodes7[i].y, 2])
        imgR.append(img[nodes8[i].x, nodes8[i].y, 2])
        imgR.append(img[nodes9[i].x, nodes9[i].y, 2])
        imgR.append(img[nodes10[i].x, nodes10[i].y, 2])
        depth.append(depth_map[nodes1[i].x, nodes1[i].y])
        depth.append(depth_map[nodes2[i].x, nodes2[i].y])
        depth.append(depth_map[nodes3[i].x, nodes3[i].y])
        depth.append(depth_map[nodes4[i].x, nodes4[i].y])
        depth.append(depth_map[nodes5[i].x, nodes5[i].y])
        depth.append(depth_map[nodes6[i].x, nodes6[i].y])
        depth.append(depth_map[nodes7[i].x, nodes7[i].y])
        depth.append(depth_map[nodes8[i].x, nodes8[i].y])
        depth.append(depth_map[nodes9[i].x, nodes9[i].y])
        depth.append(depth_map[nodes10[i].x, nodes10[i].y])

    imgB = np.array(imgB) / 255
    imgG = np.array(imgG) / 255
    imgR = np.array(imgR) / 255
    depth = np.array(depth) / 255

    ab, bb, cb, db = nls(depth, imgB)
    ag, bg, cg, dg = nls(depth, imgG)
    ar, br, cr, dr = nls(depth, imgR)

    bsrm = np.zeros(img.shape)
    bsrm = np.float64(bsrm)

    for i in range(0, 3):
        if i == 0:
            bsrm[:, :, i] = img[:, :, i] / 255 - (ab * (1 - np.exp(-(bb) * (depth_map[:, :] / 255))) + cb * np.exp(-(db)))
        if i == 1:
            bsrm[:, :, i] = img[:, :, i] / 255 - (ag * (1 - np.exp(-(bg) * (depth_map[:, :] / 255))) + cg * np.exp(-(dg)))
        if i == 2:
            bsrm[:, :, i] = img[:, :, i] / 255 - (ar * (1 - np.exp(-(br) * (depth_map[:, :] / 255))) + cr * np.exp(-(dr)))

    bsrm = np.array(bsrm) * 255
    out_path = os.path.join(OUTPUT_DIR, CURRENT_PREFIX + CURRENT_VARIANT_SUFFIX + "_backscatter.jpg")
    cv2.imwrite(out_path, bsrm)
    return bsrm


def nls(depth, img):
    p0 = [1, 5, 1, 5]
    s = "Test the number of iteration"
    Para = optimize.leastsq(test_err, p0, args=(depth, img, s), maxfev=10000)
    a, b, c, d = Para[0]
    return a, b, c, d


def nls2(depth, img):
    p1 = [1, -1, 1, -1]
    s = "Test the number of iteration"
    Para = optimize.leastsq(test_err2, p1, args=(depth, img, s), maxfev=10000)
    a, b, c, d = Para[0]
    return a, b, c, d


def test_err2(p, x, y, s):
    return fit1(p, x) - y


def test_func(p, x):
    a, b, c, d = p
    return a * (1 - np.exp(-(b) * x)) + c * np.exp(-(d))


def test_err(p, x, y, s):
    return test_func(p, x) - y


def fit(x, a, b, c, d):
    return a * np.exp(b * x) + c * np.exp(d * x)


def fit1(p, x):
    a, b, c, d = p
    return a * np.exp(b * x) + c * np.exp(d * x)


def direct_signal(img, Ec, depths):
    Jc = np.zeros(img.shape)
    img = img / 255.0
    depths = depths / 255 * 30
    Ec = Ec / 255
    Ec = np.clip(Ec, 1e-6, 1.0)
    Jc = img / Ec
    Jc = np.clip(Jc, 0.0, 1.0)
    Jc = (Jc * 255.0).astype(np.uint8)

    out_path = os.path.join(OUTPUT_DIR, CURRENT_PREFIX + CURRENT_VARIANT_SUFFIX + "_jc.jpg")
    cv2.imwrite(out_path, Jc)
    return Jc


def benchmark_line(handle, file_name, step_name, elapsed):
    line = f"{file_name}\t{step_name}\t{elapsed:.6f}s\n"
    print(line.strip())
    handle.write(line)
    handle.flush()


np.seterr(over='ignore')

if __name__ == '__main__':
    start_total = time.perf_counter()
    starttime = datetime.datetime.now()

    input_dir = "InputImages"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    benchmark_path = "benchmark.txt"

    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    files = natsort.natsorted(files)

    with open(benchmark_path, "w") as benchmark_file:
        benchmark_file.write("Benchmark Results\n")
        benchmark_file.write("=================\n")

        for file in files:
            filepath = os.path.join(input_dir, file)
            prefix = os.path.splitext(file)[0]

            print("Running file :", file)

            # Compute depth exactly once per image
            t0 = time.perf_counter()
            subprocess.run([sys.executable, "newestdepth.py", filepath], check=True)
            benchmark_line(benchmark_file, file, "depth", time.perf_counter() - t0)
            print("Successfully generated depth images")

            img = cv2.imread(filepath)
            depth_map = cv2.imread(os.path.join(OUTPUT_DIR, prefix + "_depth_map.jpg"), cv2.IMREAD_GRAYSCALE)

            if img is None:
                print("Skipping {}, could not load input image".format(file))
                continue
            if depth_map is None:
                print("Skipping {}, missing depth map".format(file))
                continue

            CURRENT_PREFIX = prefix

            for beta in GUIDED_BETAS:
                beta_str = format_beta(beta)
                CURRENT_VARIANT_SUFFIX = variant_tag(beta)

                print(f"Generating outputs for guided_beta={beta_str}")

                # Stage 2: LSAC illumination map
                t0 = time.perf_counter()
                subprocess.run(
                    [
                        sys.executable,
                        "LSAC_guided_luma_regularizer_beta_sweep.py",
                        filepath,
                        beta_str,
                        CURRENT_VARIANT_SUFFIX,
                    ],
                    check=True,
                )
                benchmark_line(benchmark_file, file, f"lsac_beta_{beta_str}", time.perf_counter() - t0)

                estill = cv2.imread(file_with_variant(prefix, "lsac", "jpg", beta))
                if estill is None:
                    print(f"Skipping beta={beta_str}, missing LSAC map")
                    continue

                # Stage 3: backscatter removal
                t0 = time.perf_counter()
                testDC = backscatter(img, 0.01)
                benchmark_line(benchmark_file, file, f"backscatter_beta_{beta_str}", time.perf_counter() - t0)

                # Stage 4: attenuation restoration
                t0 = time.perf_counter()
                direct_signal(testDC, estill, depth_map)
                benchmark_line(benchmark_file, file, f"attenuation_restoration_beta_{beta_str}", time.perf_counter() - t0)

                # Stage 5: white balance
                t0 = time.perf_counter()
                subprocess.run(
                    [
                        sys.executable,
                        "white.py",
                        os.path.join(OUTPUT_DIR, prefix + CURRENT_VARIANT_SUFFIX + "_jc.jpg"),
                    ],
                    check=True
                )
                benchmark_line(benchmark_file, file, f"white_balance_beta_{beta_str}", time.perf_counter() - t0)
                print(f"Completed guided_beta={beta_str}")

        benchmark_line(benchmark_file, "ALL", "total", time.perf_counter() - start_total)

    Endtime = datetime.datetime.now()
    Time = Endtime - starttime
    print('Time', Time)
