import cv2
import numpy as np
import os
import sys


def guided_filter_channel(guide, src, radius, eps):
    """
    Apply guided filtering to one channel. Prefer OpenCV ximgproc if available;
    otherwise fall back to a bilateral filter on the source channel.
    Inputs are float32 in [0, 1].
    """
    if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'guidedFilter'):
        return cv2.ximgproc.guidedFilter(
            guide=guide,
            src=src,
            radius=radius,
            eps=eps,
        )

    # Fallback: bilateral on the source. Not the same as guided filtering,
    # but still edge-aware and available in base OpenCV.
    return cv2.bilateralFilter(src, d=9, sigmaColor=0.08, sigmaSpace=max(3, radius // 2))


def estimate_lsac_guided_depth(img_bgr, depth_gray=None, radius=35, eps=1e-3, depth_alpha=0.15):
    img = img_bgr.astype(np.float32) / 255.0

    # Use luminance-like grayscale as the guide to preserve major scene edges.
    guide = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Coarse illumination seed: broad smooth version of the image.
    seed = cv2.GaussianBlur(img, (0, 0), sigmaX=15, sigmaY=15)

    # Mild depth modulation only; do not let depth dominate illumination.
    if depth_gray is not None:
        depth = depth_gray.astype(np.float32) / 255.0
        depth_weight = 1.0 + depth_alpha * depth
        seed = np.clip(seed * depth_weight[..., None], 0.0, 1.0)

    lsac = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        lsac[:, :, c] = guided_filter_channel(guide, seed[:, :, c], radius, eps)

    # Gentle final smoothing to suppress residual speckle and keep the field stable.
    lsac = cv2.GaussianBlur(lsac, (0, 0), sigmaX=1.2, sigmaY=1.2)
    lsac = np.clip(lsac, 0.05, 1.0)
    return (lsac * 255.0).astype(np.uint8)


def main():
    if len(sys.argv) < 2:
        raise ValueError("Usage: python LSAC_guided.py <input_image_path>")

    path = sys.argv[1]
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    prefix = os.path.splitext(os.path.basename(path))[0]
    os.makedirs("OutputImages", exist_ok=True)

    depth_path = os.path.join("OutputImages", prefix + "_depth_map.jpg")
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if depth is None:
        # Depth is optional for this variant; continue without it.
        print(f"Warning: could not read depth map: {depth_path}. Continuing without depth guidance.")

    lsac = estimate_lsac_guided_depth(
        img_bgr=img,
        depth_gray=depth,
        radius=35,
        eps=1e-3,
        depth_alpha=0.15,
    )

    out_path = os.path.join("OutputImages", prefix + "_lsac.jpg")
    cv2.imwrite(out_path, lsac)
    print(f"Saved guided LSAC to {out_path}")


if __name__ == '__main__':
    main()
