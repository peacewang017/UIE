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


def estimate_lsac_guided_rgb_iterative(
    img_bgr,
    radius=35,
    eps=1e-3,
    post_sigma=0.0,
    max_iters=50,
    loss_threshold=1e-5,
    init_mode="image",
):
    """
    Iterative RGB guided LSAC with separate channel guides.

    init_mode: We used image but the original paper used zero
      "image" -> initialize current estimate from the input image
      "zero"  -> initialize from zeros
    """
    img = img_bgr.astype(np.float32) / 255.0

    if init_mode == "zero":
        current = np.zeros_like(img, dtype=np.float32)
    elif init_mode == "image":
        current = img.copy()
    else:
        raise ValueError("init_mode must be 'image' or 'zero'")

    final_loss = None
    num_iters = 0
    loss_history = []

    for i in range(max_iters):
        updated = np.zeros_like(current, dtype=np.float32)

        # Separate channel guides: B->B, G->G, R->R
        for c in range(3):
            guide_c = img[:, :, c]
            updated[:, :, c] = guided_filter_channel(
                guide=guide_c,
                src=current[:, :, c],
                radius=radius,
                eps=eps,
            )

        updated = np.clip(updated, 0.05, 1.0)

        loss = float(np.mean(np.abs(updated - current)))
        loss_history.append(loss)
        final_loss = loss
        num_iters = i + 1
        current = updated

        if loss < loss_threshold:
            break

    return (current * 255.0).astype(np.uint8), final_loss, num_iters, np.array(loss_history, dtype=np.float32)


def main():
    if len(sys.argv) < 2:
        raise ValueError("Usage: python LSAC_guided_rgb_iterative.py <input_image_path>")

    path = sys.argv[1]
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    prefix = os.path.splitext(os.path.basename(path))[0]
    os.makedirs("OutputImages", exist_ok=True)

    lsac, loss, n_iters, loss_history = estimate_lsac_guided_rgb_iterative(
        img_bgr=img,
        radius=35,
        eps=1e-3,
        post_sigma=0.0,
        max_iters=50,
        loss_threshold=1e-5,
        init_mode="image",
    )

    out_path = os.path.join("OutputImages", prefix + "_lsac.jpg")
    cv2.imwrite(out_path, lsac)
    np.save(os.path.join("OutputImages", prefix + "_lsac_guided_rgb_iter_loss.npy"), loss_history)

    print(f"Saved iterative RGB guided LSAC to {out_path}")
    print(f"Iterations: {n_iters}, final loss: {loss:.6f}")


if __name__ == '__main__':
    main()
