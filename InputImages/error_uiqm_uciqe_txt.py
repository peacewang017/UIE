
import numpy as np
import cv2
import sys
import os

RGT_0207 = (
    (813,431),(816,399),(885,405),(883,437),
    (820,361),(817,393),(886,339),(889,368),
    (821,355),(824,324),(893,331),(891,362),
    (828,288),(825,318),(894,325),(897,294),
    (829,282),(833,252),(901,258),(897,289),
    (834,246),(837,217),(903,225),(901,252),
)
RGT_0216 = (
    (273,719),(299,702),(308,766),(334,749),
    (303,699),(330,682),(366,728),(340,746),
    (334,678),(360,662),(396,709),(370,726),
    (365,658),(391,641),(426,689),(401,706),
    (395,638),(420,622),(457,670),(431,685),
    (426,619),(450,604),(461,666),(484,649),
)
T_S04882 = (
    (675,1332),(706,1315),(782,1344),(749,1361),
    (712,1312),(745,1295),(821,1324),(789,1341),
    (750,1292),(781,1277),(855,1305),(825,1321),
    (788,1274),(816,1260),(888,1286),(860,1302),
    (821,1258),(848,1244),(919,1270),(893,1284),
    (853,1242),(879,1229),(949,1254),(924,1268),
)
LFT_3403 = (
    (628,776),(639,796),(639,797),(628,797),
    (641,777),(652,777),(652,797),(640,797),
    (654,777),(665,777),(665,798),(653,798),
    (666,777),(677,777),(677,798),(666,798),
    (679,777),(691,777),(691,798),(678,798),
    (692,777),(703,777),(703,798),(692,798),
)

DATASETS = {
    "RGT_0207": ("RGT_0207_result.png", RGT_0207),
    "RGT_0216": ("RGT_0216_result.png", RGT_0216),
    "T_S04882": ("T_S04882_result.png", T_S04882),
    "LFT_3403": ("LFT_3403_result.png", LFT_3403),
}


def angle_error_region(img, quad_coords):
    pts = np.array([(x, y) for x, y in quad_coords], dtype=np.int32)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    ys, xs = np.where(mask == 255)
    if len(xs) == 0:
        return 0.0

    pixels = img[ys, xs].astype(np.float64)
    num = np.sum(pixels, axis=1)
    denom = np.sqrt(3) * np.linalg.norm(pixels, axis=1)

    valid = denom > 0
    angles = np.arccos(np.clip(num[valid] / denom[valid], -1.0, 1.0))
    return float(np.mean(angles)) if len(angles) > 0 else 0.0


def avg_error(img, coords):
    total = 0.0
    for i in range(0, 24, 4):
        quad = coords[i:i+4]
        total += angle_error_region(img, quad)
    return total / 6


def _alpha_trimmed_stats(x, alpha=0.1):
    x = np.sort(x.reshape(-1).astype(np.float64))
    n = len(x)
    if n == 0:
        return 0.0, 0.0
    trim = int(alpha * n)
    if 2 * trim >= n:
        trimmed = x
    else:
        trimmed = x[trim:n-trim]
    if trimmed.size == 0:
        trimmed = x
    mu = float(np.mean(trimmed))
    var = float(np.var(trimmed))
    return mu, var


def uicm(img_rgb):
    img = img_rgb.astype(np.float64)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    rg = R - G
    yb = (R + G) / 2.0 - B

    mu_rg, var_rg = _alpha_trimmed_stats(rg, alpha=0.1)
    mu_yb, var_yb = _alpha_trimmed_stats(yb, alpha=0.1)

    return (-0.0268 * np.sqrt(mu_rg ** 2 + mu_yb ** 2)
            + 0.1586 * np.sqrt(var_rg + var_yb))


def eme(channel, block_size=8):
    channel = channel.astype(np.float64) + 1e-6
    h, w = channel.shape
    k1 = h // block_size
    k2 = w // block_size
    if k1 == 0 or k2 == 0:
        cmax = np.max(channel)
        cmin = np.min(channel)
        return 20.0 * np.log(cmax / cmin) if cmin > 0 and cmax > cmin else 0.0

    val = 0.0
    count = 0
    for i in range(k1):
        for j in range(k2):
            block = channel[i * block_size:(i + 1) * block_size,
                            j * block_size:(j + 1) * block_size]
            cmax = np.max(block)
            cmin = np.min(block)
            if cmin > 0 and cmax > cmin:
                val += np.log(cmax / cmin)
            count += 1
    return 2.0 * val / max(count, 1)


def uism(img_rgb):
    img = img_rgb.astype(np.uint8)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    sobel_r = cv2.Sobel(R, cv2.CV_64F, 1, 1, ksize=3)
    sobel_g = cv2.Sobel(G, cv2.CV_64F, 1, 1, ksize=3)
    sobel_b = cv2.Sobel(B, cv2.CV_64F, 1, 1, ksize=3)

    edge_r = np.abs(sobel_r) * R
    edge_g = np.abs(sobel_g) * G
    edge_b = np.abs(sobel_b) * B

    return 0.299 * eme(edge_r) + 0.587 * eme(edge_g) + 0.114 * eme(edge_b)


def uiconm(img_rgb, block_size=8):
    gray = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64) + 1e-6
    h, w = gray.shape
    k1 = h // block_size
    k2 = w // block_size
    if k1 == 0 or k2 == 0:
        cmax = np.max(gray)
        cmin = np.min(gray)
        if cmax + cmin == 0:
            return 0.0
        return (cmax - cmin) / (cmax + cmin)

    total = 0.0
    count = 0
    for i in range(k1):
        for j in range(k2):
            block = gray[i * block_size:(i + 1) * block_size,
                         j * block_size:(j + 1) * block_size]
            cmax = np.max(block)
            cmin = np.min(block)
            if cmax + cmin > 0:
                total += (cmax - cmin) / (cmax + cmin)
            count += 1
    return total / max(count, 1)


def uiqm(img_rgb):
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    return c1 * uicm(img_rgb) + c2 * uism(img_rgb) + c3 * uiconm(img_rgb)


def uciqe(img_bgr):
    img = img_bgr.astype(np.float32) / 255.0
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    L = lab[:, :, 0] / 100.0
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    chroma = np.sqrt(a * a + b * b)
    sigma_c = np.std(chroma)

    con_l = np.percentile(L, 99) - np.percentile(L, 1)

    saturation = chroma / np.sqrt(chroma * chroma + L * L + 1e-12)
    mu_s = np.mean(saturation)

    return 0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s


def evaluate_dataset(name):
    img_path, coords = DATASETS[name]
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image '{img_path}'")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    angular_error = avg_error(img_rgb, coords)
    quality_uiqm = uiqm(img_rgb)
    quality_uciqe = uciqe(img)

    return {
        "dataset": name,
        "image": img_path,
        "angular_error_deg": float(np.degrees(angular_error)),
        "angular_error_rad": float(angular_error),
        "uiqm": float(quality_uiqm),
        "uciqe": float(quality_uciqe),
    }


def write_results(results, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Angular Error, UIQM, and UCIQE Results\n")
        f.write("======================================\n\n")
        for result in results:
            f.write(f"Dataset: {result['dataset']}\n")
            f.write(f"Image: {result['image']}\n")
            f.write(f"Average angular error: {result['angular_error_deg']:.4f} deg  ({result['angular_error_rad']:.6f} rad)\n")
            f.write(f"UIQM: {result['uiqm']:.6f}\n")
            f.write(f"UCIQE: {result['uciqe']:.6f}\n")
            f.write("\n")


def main():
    output_path = "error_metrics.txt"

    if len(sys.argv) == 2 and sys.argv[1] in DATASETS:
        names = [sys.argv[1]]
    elif len(sys.argv) == 2 and sys.argv[1].lower() == "all":
        names = list(DATASETS.keys())
    else:
        print(f"Usage: {sys.argv[0]} <dataset|all>")
        print(f"Available datasets: {', '.join(DATASETS.keys())}")
        sys.exit(1)

    results = []
    for name in names:
        results.append(evaluate_dataset(name))

    write_results(results, output_path)

    for result in results:
        print(f"[{result['dataset']}] Average angular error: {result['angular_error_deg']:.4f} deg  ({result['angular_error_rad']:.6f} rad)")
        print(f"[{result['dataset']}] UIQM: {result['uiqm']:.6f}")
        print(f"[{result['dataset']}] UCIQE: {result['uciqe']:.6f}")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
