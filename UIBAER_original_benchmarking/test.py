import os
import time
import numpy as np
import cv2
import natsort
from scipy import optimize
import subprocess
import sys

OUTPUT_DIR = "OutputImages"
depth_map = None
CURRENT_PREFIX = ""

class Node(object):
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value

    def printInfo(self):
        print(self.x, self.y, self.value)


def backscatter(img, percent=0.01):
    global depth_map
    height = img.shape[0]
    width = int(img.shape[1] / 10)
    size = height * width

    nodes = [[] for _ in range(10)]
    for i in range(height):
        for band in range(10):
            start = band * width
            end = (band + 1) * width
            for j in range(start, end):
                nodes[band].append(Node(i, j, sum(tuple(img[i, j]))))

    nodes = [sorted(group, key=lambda node: node.value) for group in nodes]

    img_r = []
    img_g = []
    img_b = []
    depth = []
    sample_count = int(percent * size)

    for i in range(sample_count):
        for group in nodes:
            n = group[i]
            img_b.append(img[n.x, n.y, 0])
            img_g.append(img[n.x, n.y, 1])
            img_r.append(img[n.x, n.y, 2])
            depth.append(depth_map[n.x, n.y])

    img_b = np.array(img_b) / 255.0
    img_g = np.array(img_g) / 255.0
    img_r = np.array(img_r) / 255.0
    depth = np.array(depth) / 255.0

    ab, bb, cb, db = nls(depth, img_b)
    ag, bg, cg, dg = nls(depth, img_g)
    ar, br, cr, dr = nls(depth, img_r)

    bsrm = np.zeros(img.shape, dtype=np.float64)
    bsrm[:, :, 0] = img[:, :, 0] / 255.0 - (ab * (1 - np.exp(-bb * (depth_map / 255.0))) + cb * np.exp(-db))
    bsrm[:, :, 1] = img[:, :, 1] / 255.0 - (ag * (1 - np.exp(-bg * (depth_map / 255.0))) + cg * np.exp(-dg))
    bsrm[:, :, 2] = img[:, :, 2] / 255.0 - (ar * (1 - np.exp(-br * (depth_map / 255.0))) + cr * np.exp(-dr))

    bsrm = np.clip(bsrm * 255.0, 0, 255).astype(np.uint8)
    return bsrm


def nls(depth, img):
    p0 = [1, 5, 1, 5]
    params = optimize.leastsq(test_err, p0, args=(depth, img, "benchmark"), maxfev=10000)
    a, b, c, d = params[0]
    return a, b, c, d


def test_func(p, x):
    a, b, c, d = p
    return a * (1 - np.exp(-b * x)) + c * np.exp(-d)


def test_err(p, x, y, s):
    return test_func(p, x) - y


def direct_signal(img, ill, depths):
    ill = np.clip(ill.astype(np.float64) / 255.0, 1e-6, 1.0)
    img = img.astype(np.float64) / 255.0
    jc = np.clip(img / ill, 0.0, 1.0)
    return (jc * 255.0).astype(np.uint8)


def benchmark_line(handle, file_name, step_name, elapsed):
    line = f"{file_name}\t{step_name}\t{elapsed:.6f}s\n"
    print(line.strip())
    handle.write(line)
    handle.flush()


if __name__ == '__main__':
    np.seterr(over='ignore')
    start_total = time.perf_counter()

    input_dir = "InputImages"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    benchmark_path = "benchmark.txt"

    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    files = natsort.natsorted(files)

    with open(benchmark_path, "w", encoding="utf-8") as bench:
        bench.write("file\tstep\tseconds\n")

        for file in files:
            filepath = os.path.join(input_dir, file)
            prefix = os.path.splitext(file)[0]
            print(f"Running file: {file}")

            t0 = time.perf_counter()
            subprocess.run(
                [sys.executable, "newestdepth.py", filepath, os.path.join(OUTPUT_DIR, f"{prefix}_depth_map.jpg")],
                check=True,
            )
            benchmark_line(bench, file, "depth", time.perf_counter() - t0)

            t0 = time.perf_counter()
            subprocess.run(
                [sys.executable, "LSAC2.py", filepath, os.path.join(OUTPUT_DIR, f"{prefix}_lsac.jpg")],
                check=True,
            )
            benchmark_line(bench, file, "lsac", time.perf_counter() - t0)

            img = cv2.imread(filepath)
            depth_map = cv2.imread(os.path.join(OUTPUT_DIR, f"{prefix}_depth_map.jpg"), cv2.IMREAD_GRAYSCALE)
            estill = cv2.imread(os.path.join(OUTPUT_DIR, f"{prefix}_lsac.jpg"))

            if img is None or depth_map is None or estill is None:
                print(f"Skipping {file}, missing required input/output image")
                continue

            t0 = time.perf_counter()
            test_dc = backscatter(img, 0.01)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{prefix}_backscatter.jpg"), test_dc)
            benchmark_line(bench, file, "backscatter", time.perf_counter() - t0)

            t0 = time.perf_counter()
            jc = direct_signal(test_dc, estill, depth_map)
            jc_path = os.path.join(OUTPUT_DIR, f"{prefix}_jc.jpg")
            cv2.imwrite(jc_path, jc)
            benchmark_line(bench, file, "attenuation_restoration", time.perf_counter() - t0)

            t0 = time.perf_counter()
            subprocess.run(
                [sys.executable, "white.py", jc_path, os.path.join(OUTPUT_DIR, f"{prefix}_result.jpg")],
                check=True,
            )
            benchmark_line(bench, file, "white_balance", time.perf_counter() - t0)

        benchmark_line(bench, "ALL", "total", time.perf_counter() - start_total)
