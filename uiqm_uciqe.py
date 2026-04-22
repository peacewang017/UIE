
import numpy as np
import cv2
import sys
import os



DATASETS = {
    "RGT_0207": ("RGT_0207.png"),
    "RGT_0216": ("RGT_0216.png"),
    "T_S04882": ("T_S04882.png"),
    "LFT_3403": ("LFT_3403.png"),
    "T_S04866": ("T_S04866.png"),
    "18_img_": ("18_img_.png"),
    "178_img_": ("178_img_.png"),
    "254_img_": ("254_img_.png"),
    "365_img_": ("365_img_.png"),
    "173_img_": ("173_img_.png"),
    "206_img_": ("206_img_.png"),
    "241_img_": ("241_img_.png"),
    "141_img_": ("141_img_.png"),
    #"8_img_": ("8_img_.png"),
}






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
    img_path = DATASETS[name]
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image '{img_path}'")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #angular_error = avg_error(img_rgb, coords)
    quality_uiqm = uiqm(img_rgb)
    quality_uciqe = uciqe(img)

    return {
        "dataset": name,
        "image": img_path,
        #"angular_error_deg": float(np.degrees(angular_error)),
        #"angular_error_rad": float(angular_error),
        "uiqm": float(quality_uiqm),
        "uciqe": float(quality_uciqe),
    }


def write_results(results, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("UIQM, and UCIQE Results\n")
        f.write("======================================\n\n")
        for result in results:
            f.write(f"Dataset: {result['dataset']}\n")
            f.write(f"Image: {result['image']}\n")
            #f.write(f"Average angular error: {result['angular_error_deg']:.4f} deg  ({result['angular_error_rad']:.6f} rad)\n")
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
        #print(f"[{result['dataset']}] Average angular error: {result['angular_error_deg']:.4f} deg  ({result['angular_error_rad']:.6f} rad)")
        print(f"[{result['dataset']}] UIQM: {result['uiqm']:.6f}")
        print(f"[{result['dataset']}] UCIQE: {result['uciqe']:.6f}")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
