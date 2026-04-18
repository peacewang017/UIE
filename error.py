import numpy as np
import cv2
import sys

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
    "RGT_0207": ("RGT_0207_result.jpg", RGT_0207),
    "RGT_0216": ("RGT_0216_result.jpg", RGT_0216),
    "T_S04882": ("T_S04882_result.jpg", T_S04882),
    "LFT_3403": ("LFT_3403_result.jpg", LFT_3403),
}

def angle_error_region(img, quad_coords):
    """
    Compute mean angular error over all pixels inside a quadrilateral mask.
    quad_coords: 4 (x,y) tuples defining the quadrilateral corners.
    """
    pts = np.array([(x, y) for x, y in quad_coords], dtype=np.int32)

    # Build a mask for the quadrilateral region
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    # Get all pixel locations inside the mask (returns row,col = y,x)
    ys, xs = np.where(mask == 255)
    if len(xs) == 0:
        return 0.0

    # Sample pixels: img is indexed [row, col] = [y, x]
    pixels = img[ys, xs].astype(np.float64)  # shape (N, 3)

    num = np.sum(pixels, axis=1)                        # R+G+B per pixel
    denom = np.sqrt(3) * np.linalg.norm(pixels, axis=1) # sqrt(3)*|v| per pixel

    # Avoid division by zero for black pixels
    valid = denom > 0
    angles = np.arccos(np.clip(num[valid] / denom[valid], -1.0, 1.0))

    return float(np.mean(angles)) if len(angles) > 0 else 0.0

def avg_error(img, coords):
    """Average angular error across all 6 quadrilaterals."""
    total = 0.0
    for i in range(0, 24, 4):
        quad = coords[i:i+4]
        total += angle_error_region(img, quad)
    return total / 6

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in DATASETS:
        print(f"Usage: {sys.argv[0]} <dataset>")
        print(f"Available datasets: {', '.join(DATASETS.keys())}")
        sys.exit(1)

    name = sys.argv[1]
    img_path, coords = DATASETS[name]

    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: could not load image '{img_path}'")
        sys.exit(1)

    # Convert BGR -> RGB so channel sum R+G+B is meaningful
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    error = avg_error(img_rgb, coords)
    print(f"[{name}] Average angular error: {np.degrees(error):.4f} deg  ({error:.6f} rad)")

if __name__ == "__main__":
    main()
