import cv2
import numpy as np
import os
import sys


def estimate_lsac_luminance(img_bgr, depth_gray=None, radius=60, eps=1e-2,
                            seed_sigma=15.0, post_sigma=2.5, depth_alpha=0.0):
    """
    Luminance-only LSAC estimate.

    - Builds a single-channel illumination map from grayscale/luminance.
    - Uses guided filtering when available, otherwise bilateral filtering.
    - Applies an extra blur to remove residual structure.
    - Replicates the final map to 3 channels so the rest of the pipeline
      can keep reading OutputImages/<prefix>_lsac.jpg as before.

    depth_alpha is kept at 0 by default so depth does not dominate the
    luminance estimate. You can raise it slightly (e.g. 0.05-0.10) later.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Smooth seed: illumination should be low-frequency
    seed = cv2.GaussianBlur(gray, (0, 0), sigmaX=seed_sigma, sigmaY=seed_sigma)

    # Optional mild depth modulation only
    if depth_gray is not None and depth_alpha > 0.0:
        depth = depth_gray.astype(np.float32) / 255.0
        seed = np.clip(seed * (1.0 + depth_alpha * depth), 0.0, 1.0)

    # Guided filter if available; otherwise fall back to bilateral filter
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        illum = cv2.ximgproc.guidedFilter(
            guide=gray,
            src=seed,
            radius=radius,
            eps=eps,
        )
    else:
        illum = cv2.bilateralFilter(seed.astype(np.float32), d=9,
                                    sigmaColor=0.08, sigmaSpace=21)

    # Extra blur to keep illumination smooth and not too edge-aware
    illum = cv2.GaussianBlur(illum, (0, 0), sigmaX=post_sigma, sigmaY=post_sigma)

    # Keep downstream division stable
    illum = np.clip(illum, 0.08, 1.0)

    # Repeat single luminance map into 3 channels for compatibility
    lsac = np.repeat(illum[:, :, None], 3, axis=2)
    return (lsac * 255.0).astype(np.uint8)


def main():
    if len(sys.argv) < 2:
        raise ValueError("Usage: python LSAC_guided_luma.py <input_image_path>")

    path = sys.argv[1]
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    prefix = os.path.splitext(os.path.basename(path))[0]
    os.makedirs("OutputImages", exist_ok=True)

    depth_path = os.path.join("OutputImages", prefix + "_depth_map.jpg")
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if depth is None:
        depth = None

    lsac = estimate_lsac_luminance(
        img_bgr=img,
        depth_gray=depth,
        radius=60,
        eps=1e-2,
        seed_sigma=15.0,
        post_sigma=2.5,
        depth_alpha=0.0,
    )

    cv2.imwrite(os.path.join("OutputImages", prefix + "_lsac.jpg"), lsac)


if __name__ == "__main__":
    main()
