import cv2
import numpy as np
import os
import sys


def guided_filter_channel(guide, src, radius, eps):
    """
    Use OpenCV guided filter if available; otherwise fall back to bilateral, let's us use different py versions :)
    guide/src are float32 in [0, 1]
    """
    if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'guidedFilter'):
        return cv2.ximgproc.guidedFilter(
            guide=guide,
            src=src,
            radius=radius,
            eps=eps,
        )

    return cv2.bilateralFilter(src, d=9, sigmaColor=0.08, sigmaSpace=max(3, radius // 2))


def estimate_lsac_guided_rgb(img_bgr, radius=35, eps=1e-3):
    """
    One-shot guided LSAC with separate channel guides.

    - no depth this time, each output channel is guided by its own input channel
    """
    img = img_bgr.astype(np.float32) / 255.0

    # Coarse illumination seed: broad smooth version of the RGB image.
    seed = cv2.GaussianBlur(img, (0, 0), sigmaX=15, sigmaY=15)

    lsac = np.zeros_like(img, dtype=np.float32)

    # Separate channel guides: B->B, G->G, R->R
    for c in range(3):
        guide_c = img[:, :, c]
        src_c = seed[:, :, c]
        lsac[:, :, c] = guided_filter_channel(guide_c, src_c, radius, eps)

    # Gentle final smoothing to suppress residual speckle and keep the field stable.
    lsac = cv2.GaussianBlur(lsac, (0, 0), sigmaX=1.2, sigmaY=1.2)
    lsac = np.clip(lsac, 0.05, 1.0)
    return (lsac * 255.0).astype(np.uint8)


def main():
    if len(sys.argv) < 2:
        raise ValueError("Usage: python LSAC_guided_rgb_separate_guides.py <input_image_path>")

    path = sys.argv[1]
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    prefix = os.path.splitext(os.path.basename(path))[0]
    os.makedirs("OutputImages", exist_ok=True)

    lsac = estimate_lsac_guided_rgb(
        img_bgr=img,
        radius=35,
        eps=1e-3,
    )

    out_path = os.path.join("OutputImages", prefix + "_lsac.jpg")
    cv2.imwrite(out_path, lsac)
    print(f"Saved RGB guided LSAC with separate channel guides to {out_path}")


if __name__ == '__main__':
    main()
