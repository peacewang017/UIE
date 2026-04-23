# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import sys


def guided_filter_channel(guide, src, radius, eps):
    """
    Use OpenCV guided filter if available; otherwise fall back to bilateral, let's us use different py versions :)
    guide/src are float32 in [0, 1]
    """
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        return cv2.ximgproc.guidedFilter(
            guide=guide,
            src=src,
            radius=radius,
            eps=eps,
        )
    return cv2.bilateralFilter(
        src,
        d=9,
        sigmaColor=0.08,
        sigmaSpace=max(3, radius // 2),
    )


def local_avg_5pt(u):
    """
    5-point local averaging on an image padded by 1 pixel on each side.
    Input shape: (H+2, W+2)
    Output shape: (H, W)
    """
    center = u[1:-1, 1:-1]
    up     = u[:-2, 1:-1]
    down   = u[2:, 1:-1]
    left   = u[1:-1, :-2]
    right  = u[1:-1, 2:]
    return (center + up + down + left + right) / 5.0


def pad1(x):
    """
    Pad with 1-pixel reflect border.
    """
    return np.pad(x, ((1, 1), (1, 1)), mode="reflect")


def lsac2_guided_regularizer(
    img_bgr,
    p=0.001,
    max_iters=1000,
    loss_threshold=1e-5,
    guided_radius=16,
    guided_eps=1e-3,
    guided_beta=0.20,
    post_sigma=0.0,
    init_mode="image",
    verbose=True,
):

    img = img_bgr.astype(np.float32) / 255.0
    H, W, _ = img.shape

    # grayscale guide
    guide = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    if init_mode == "zero":
        current = np.zeros_like(img, dtype=np.float32)
    elif init_mode == "image":
        current = img.copy()
    else:
        raise ValueError("init_mode must be 'image' or 'zero'")

    loss_history = []

    for i in range(max_iters):
        updated = np.zeros_like(current, dtype=np.float32)

        for c in range(3):
            # LSAC2-style local diffusion
            local = local_avg_5pt(pad1(current[:, :, c]))

            # Guided regularizer
            guided = guided_filter_channel(
                guide=guide,
                src=current[:, :, c],
                radius=guided_radius,
                eps=guided_eps,
            )

            # Blend local diffusion with guided regularization
            regularized = (1.0 - guided_beta) * local + guided_beta * guided

            # Anchor to original image, LSAC2-style
            updated[:, :, c] = p * img[:, :, c] + (1.0 - p) * regularized

        if post_sigma > 0:
            updated = cv2.GaussianBlur(
                updated,
                (0, 0),
                sigmaX=post_sigma,
                sigmaY=post_sigma,
            )

        updated = np.clip(updated, 0.0, 1.0)

        loss = float(np.mean(np.abs(updated - current)))
        loss_history.append(loss)
        current = updated

        if verbose and (i < 10 or (i + 1) % 50 == 0):
            print(f"iter={i+1:4d}, loss={loss:.8f}")

        if loss < loss_threshold:
            if verbose:
                print(f"Converged at iter {i+1} with loss {loss:.8f}")
            break

    out_img = (current * 255.0).astype(np.uint8)
    return out_img, loss, i + 1, np.array(loss_history, dtype=np.float32)


def main():
    if len(sys.argv) < 2:
        raise ValueError(
            "Usage: python LSAC_guided_regularizer.py <input_image_path> [guided_beta] [suffix]"
        )

    path = sys.argv[1]
    guided_beta = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.20
    suffix = sys.argv[3] if len(sys.argv) >= 4 else ""

    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    prefix = os.path.splitext(os.path.basename(path))[0]
    os.makedirs("OutputImages", exist_ok=True)

    out_img, final_loss, n_iters, loss_history = lsac2_guided_regularizer(
        img_bgr=img,
        p=0.001,
        max_iters=100,
        loss_threshold=1e-5,
        guided_radius=16,
        guided_eps=1e-3,
        guided_beta=guided_beta,
        post_sigma=0.0,
        init_mode="image",
        verbose=True,
    )

    tag = suffix if suffix else f"_gbeta_{guided_beta:.2f}"

    out_path = os.path.join("OutputImages", prefix + "_lsac" + tag + ".jpg")
    cv2.imwrite(out_path, out_img)

    # keep compatibility (optional)
    if suffix == "":
        cv2.imwrite(os.path.join("OutputImages", prefix + "_lsac.jpg"), out_img)

    print(f"Saved LSAC (beta={guided_beta}) to {out_path}")


if __name__ == "__main__":
    main()