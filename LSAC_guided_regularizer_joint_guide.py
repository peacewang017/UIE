# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import sys


def guided_filter_channel(guide, src, radius, eps):
    """
    Use OpenCV guided filter if available; otherwise fall back to bilateral.
    guide/src are float32 in [0, 1].
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
    center = u[1:-1, 1:-1]
    up     = u[:-2, 1:-1]
    down   = u[2:, 1:-1]
    left   = u[1:-1, :-2]
    right  = u[1:-1, 2:]
    return (center + up + down + left + right) / 5.0


def pad1(x):
    return np.pad(x, ((1, 1), (1, 1)), mode="reflect")


def make_joint_guide(img_bgr, depth_gray=None, guide_mix=0.5):
    """
    Joint guide from intensity and depth:
        guide = guide_mix * gray + (1 - guide_mix) * depth
    guide_mix=1.0 -> intensity only
    guide_mix=0.0 -> depth only
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    if depth_gray is None:
        return gray

    depth = depth_gray.astype(np.float32) / 255.0
    guide = guide_mix * gray + (1.0 - guide_mix) * depth
    return np.clip(guide, 0.0, 1.0)


def lsac2_guided_regularizer(
    img_bgr,
    depth_gray=None,
    p=0.001,
    max_iters=1000,
    loss_threshold=1e-5,
    guided_radius=16,
    guided_eps=1e-3,
    guided_beta=0.20,
    guide_mix=0.5,
    post_sigma=0.0,
    init_mode="image",
    verbose=True,
):
    """
    Hybrid LSAC2 + joint-guided regularization.

    Update:
        local = 5-point diffusion
        guided = guided_filter(joint_guide, current)
        regularized = (1 - guided_beta) * local + guided_beta * guided
        next = p * I + (1 - p) * regularized

    guide_mix controls the joint guide:
        1.0 = intensity only
        0.0 = depth only
        0.5 = equal mix
    """
    img = img_bgr.astype(np.float32) / 255.0
    guide = make_joint_guide(img_bgr, depth_gray=depth_gray, guide_mix=guide_mix)

    if init_mode == "zero":
        current = np.zeros_like(img, dtype=np.float32)
    elif init_mode == "image":
        current = img.copy()
    else:
        raise ValueError("init_mode must be 'image' or 'zero'")

    loss_history = []
    final_loss = None
    n_iters = 0

    for i in range(max_iters):
        updated = np.zeros_like(current, dtype=np.float32)

        for c in range(3):
            local = local_avg_5pt(pad1(current[:, :, c]))

            guided = guided_filter_channel(
                guide=guide,
                src=current[:, :, c],
                radius=guided_radius,
                eps=guided_eps,
            )

            regularized = (1.0 - guided_beta) * local + guided_beta * guided
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
        final_loss = loss
        n_iters = i + 1
        current = updated

        if verbose and (i < 10 or (i + 1) % 50 == 0):
            print(f"iter={i+1:4d}, loss={loss:.8f}")

        if loss < loss_threshold:
            if verbose:
                print(f"Converged at iter {i+1} with loss {loss:.8f}")
            break

    out_img = (current * 255.0).astype(np.uint8)
    return out_img, final_loss, n_iters, np.array(loss_history, dtype=np.float32)


def main():
    if len(sys.argv) < 2:
        raise ValueError(
            "Usage: python LSAC_guided_regularizer.py <input_image_path> "
            "[guided_beta] [suffix] [guide_mix]"
        )

    path = sys.argv[1]
    guided_beta = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.20
    suffix = sys.argv[3] if len(sys.argv) >= 4 else ""
    guide_mix = float(sys.argv[4]) if len(sys.argv) >= 5 else 0.5

    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    prefix = os.path.splitext(os.path.basename(path))[0]
    os.makedirs("OutputImages", exist_ok=True)

    depth_path = os.path.join("OutputImages", prefix + "_depth_map.jpg")
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if depth is None:
        depth = None

    out_img, final_loss, n_iters, loss_history = lsac2_guided_regularizer(
        img_bgr=img,
        depth_gray=depth,
        p=0.001,
        max_iters=100,
        loss_threshold=1e-5,
        guided_radius=16,
        guided_eps=1e-3,
        guided_beta=guided_beta,
        guide_mix=guide_mix,
        post_sigma=0.0,
        init_mode="image",
        verbose=True,
    )

    tag = suffix if suffix else f"_gbeta_{guided_beta:.2f}"

    out_path = os.path.join("OutputImages", prefix + "_lsac" + tag + ".jpg")
    cv2.imwrite(out_path, out_img)

    if suffix == "":
        cv2.imwrite(os.path.join("OutputImages", prefix + "_lsac.jpg"), out_img)

    np.save(
        os.path.join("OutputImages", prefix + f"_lsac2_guided_loss{tag}.npy"),
        loss_history,
    )

    print(f"Saved LSAC (beta={guided_beta}, guide_mix={guide_mix}) to {out_path}")
    print(f"Iterations: {n_iters}")
    print(f"Final loss: {final_loss:.8f}")


if __name__ == "__main__":
    main()
