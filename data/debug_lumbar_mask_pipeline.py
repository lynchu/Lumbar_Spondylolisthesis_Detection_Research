import os
import cv2
import numpy as np


def get_boundary(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_min, y_min, x_max, y_max = float("inf"), float("inf"), 0, 0

    if contours:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

    return x_min, y_min, x_max, y_max


def keep_largest_component(mask):
    mask = mask.astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if stats.shape[0] <= 1:
        return mask

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    filtered_mask = np.zeros_like(mask)
    filtered_mask[labels == largest_label] = 1
    return filtered_mask


def visualize_one_case(img_path, mask_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    org_img = cv2.imread(img_path)
    if org_img is None:
        raise ValueError(f"Failed to read image: {img_path}")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read mask: {mask_path}")

    # your current dataset assumption: 255 = vertebra+sacrum, 0 = background
    mask = mask.astype(np.uint8)
    mask[mask != 255] = 0
    mask[mask == 255] = 1

    mask = cv2.resize(mask, (org_img.shape[1], org_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    average_size = np.mean(stats[1:, cv2.CC_STAT_AREA]) if stats.shape[0] > 1 else 0

    kernel_size = max(int(average_size * 0.025), 100)
    kernel = np.ones((kernel_size * 2, kernel_size), np.uint8)
    expanded_mask = cv2.dilate(mask, kernel, iterations=1)
    filtered_mask = keep_largest_component(expanded_mask)

    coverage = float((filtered_mask[mask == 1] > 0).mean()) if mask.sum() > 0 else 1.0

    masked_img = cv2.bitwise_and(org_img, org_img, mask=filtered_mask)

    x_min, y_min, x_max, y_max = get_boundary(filtered_mask)
    if x_max > x_min and y_max > y_min:
        cropped_img = org_img[y_min:y_max, x_min:x_max]
        cropped_masked_img = masked_img[y_min:y_max, x_min:x_max]
    else:
        cropped_img = org_img.copy()
        cropped_masked_img = masked_img.copy()

    # overlays
    overlay_original = org_img.copy()
    overlay_original[mask == 1] = [0, 255, 0]  # green = original mask

    overlay_filtered = org_img.copy()
    overlay_filtered[filtered_mask == 1] = [255, 0, 0]  # blue = filtered mask

    overlay_loss = org_img.copy()
    overlay_loss[mask == 1] = [0, 255, 0]  # green = original mask
    overlay_loss[(mask == 1) & (filtered_mask == 0)] = [0, 0, 255]  # red = lost area

    stem = os.path.splitext(os.path.basename(img_path))[0]

    cv2.imwrite(os.path.join(out_dir, f"{stem}_01_original.png"), org_img)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_02_mask.png"), mask * 255)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_03_expanded_mask.png"), expanded_mask * 255)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_04_filtered_mask.png"), filtered_mask * 255)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_05_overlay_original_mask.png"), overlay_original)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_06_overlay_filtered_mask.png"), overlay_filtered)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_07_overlay_loss_check.png"), overlay_loss)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_08_masked_img.png"), masked_img)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_09_cropped_img.png"), cropped_img)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_10_cropped_masked_img.png"), cropped_masked_img)

    print(f"[INFO] Saved debug outputs to: {out_dir}")
    print(f"[INFO] kernel_size={kernel_size}, average_size={average_size:.2f}, coverage={coverage:.4f}")
    print(f"[INFO] crop box = ({x_min}, {y_min}) -> ({x_max}, {y_max})")


if __name__ == "__main__":
    img_path = "./dataset/train/lumbar_crop/0001-Lat_lumbar.png"
    mask_path = "./dataset/train/lumbar_crop_mask/0001-Lat_lumbar.png"
    out_dir = "./debug_lumbar_mask"

    visualize_one_case(img_path, mask_path, out_dir)