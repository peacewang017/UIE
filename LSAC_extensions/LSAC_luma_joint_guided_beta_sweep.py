import cv2
import numpy as np
import os
import sys


def guided_filter_channel(guide, src, radius, eps):
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


def estimate_lsac_luminance_iterative(
    img_bgr,
    depth_gray=None,
    p=0.001,
    max_iters=100,
    loss_threshold=1e-5,
    guided_radius=16,
    guided_eps=1e-3,
    guided_beta=0.20,
    post_sigma=0.0,
    depth_alpha=0.0,
    init_mode="image",
    verbose=True,
):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    guide = gray.copy()

    gray_anchor = gray.copy()
    if depth_gray is not None and depth_alpha > 0.0:
        depth = depth_gray.astype(np.float32) / 255.0
        gray_anchor = np.clip(gray_anchor * (1.0 + depth_alpha * depth), 0.0, 1.0)

    if init_mode == "zero":
        current = np.zeros_like(gray, dtype=np.float32)
    elif init_mode == "image":
        current = gray.copy()
    else:
        raise ValueError("init_mode must be 'image' or 'zero'")

    loss_history = []
    final_loss = None
    n_iters = 0

    for i in range(max_iters):
        local = local_avg_5pt(pad1(current))

        guided = guided_filter_channel(
            guide=guide,
            src=current,
            radius=guided_radius,
            eps=guided_eps,
        )

        regularized = (1.0 - guided_beta) * local + guided_beta * guided
        updated = p * gray_anchor + (1.0 - p) * regularized

        if post_sigma > 0:
            updated = cv2.GaussianBlur(
                updated,
                (0, 0),
                sigmaX=post_sigma,
                sigmaY=post_sigma,
            )

        updated = np.clip(updated, 0.08, 1.0)

        loss = float(np.mean(np.abs(updated - current)))
        loss_history.append(loss)
        final_loss = loss
        n_iters = i + 1
        current = updated

        if verbose and (i < 10 or (i + 1) % 25 == 0):
            print(f"iter={i+1:4d}, loss={loss:.8f}")

        if loss < loss_threshold:
            if verbose:
                print(f"Converged at iter {i+1} with loss {loss:.8f}")
            break

    lsac = np.repeat(current[:, :, None], 3, axis=2)
    return (lsac * 255.0).astype(np.uint8), final_loss, n_iters, np.array(loss_history, dtype=np.float32)


def parse_args(argv):
    if len(argv) < 2:
        raise ValueError(
            "Usage: python LSAC_guided_luma_regularizer.py <input_image_path> "
            "[guided_beta] [output_suffix]"
        )

    path = argv[1]
    guided_beta = float(argv[2]) if len(argv) >= 3 else 0.20
    output_suffix = argv[3] if len(argv) >= 4 else ""
    return path, guided_beta, output_suffix


def main():
    path, guided_beta, output_suffix = parse_args(sys.argv)

    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    prefix = os.path.splitext(os.path.basename(path))[0]
    os.makedirs("OutputImages", exist_ok=True)

    depth_path = os.path.join("OutputImages", prefix + "_depth_map.jpg")
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if depth is None:
        depth = None

    lsac, final_loss, n_iters, loss_history = estimate_lsac_luminance_iterative(
        img_bgr=img,
        depth_gray=depth,
        p=0.001,
        max_iters=100,
        loss_threshold=1e-5,
        guided_radius=16,
        guided_eps=1e-3,
        guided_beta=guided_beta,
        post_sigma=0.0,
        depth_alpha=0.0,
        init_mode="image",
        verbose=True,
    )

    suffix = output_suffix if output_suffix else f"_gbeta_{guided_beta:.2f}"

    out_path = os.path.join("OutputImages", prefix + f"_lsac{suffix}.jpg")
    cv2.imwrite(out_path, lsac)

    # keep pipeline compatibility if no custom suffix was given
    if output_suffix == "":
        cv2.imwrite(os.path.join("OutputImages", prefix + "_lsac.jpg"), lsac)

    npy_path = os.path.join("OutputImages", prefix + f"_lsac_luma_loss{suffix}.npy")
    np.save(npy_path, loss_history)

    print(f"Saved pure-iterative guided-luma LSAC to {out_path}")
    print(f"Saved loss history to {npy_path}")
    print(f"guided_beta: {guided_beta:.2f}")
    print(f"Iterations: {n_iters}")
    print(f"Final loss: {final_loss:.8f}")


if __name__ == "__main__":
    main()
