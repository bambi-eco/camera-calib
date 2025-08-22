import os
from typing import Optional
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from utils import read_images
from intrinsics import IntrinsicsPair, Intrinsics


# --- utility function ---
def imshow(image, *args, **kwargs):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image, *args, **kwargs)
    plt.axis("off")


# --- main overlay function ---
def overlay_and_save(images, calibrations: IntrinsicsPair, out_dir: Optional[str] = None, target: str = "RGB", show_individual_images: bool = False):
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    target_camera_matrix = np.array(calibrations[target]["K"])
    dst_size = np.array([calibrations[target].cx*2, calibrations[target].cy*2], dtype=int)

    num_frames = len(images[target])
    for idx in range(num_frames):
        for cam in images.keys():
            calib = calibrations[cam.lower()]
            
            camera_matrix = calib.mtx
            dist_coefs = calib.dist
            img = images[cam][idx]

            mapx, mapy = cv2.initUndistortRectifyMap(
                camera_matrix, dist_coefs, None,
                target_camera_matrix, dst_size, cv2.CV_32FC1
            )
            dst = cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)

            if cam == "RGB":
                dst = cv2.Canny(dst.copy(), 100, 200)
                rgb_edge = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)
            else:
                thermal_img = dst.copy()
                cmp_img = thermal_img.copy()

        cmp_img[:, :, 1] = rgb_edge[:, :, 1]

        if show_individual_images:
            plt.figure(figsize=(10, 10))
            imshow(cmp_img)
            plt.show()

        if out_dir is not None:
            out_name = f"overlay_{idx:06d}.png"
            out_path = os.path.join(out_dir, out_name)

            cv2.imwrite(out_path, cmp_img)
            print(f"[OK] saved {out_path}")


def main():

    # test the intrinsics/calibration here

    #input paths
    rgb_folder = r"./data/rgb"
    thermal_folder = r"./data/thermal"
    output_folder = r"./results/overlays"

    # read images
    images, rgb_files, th_files = read_images(rgb_folder, thermal_folder)

    gt_calibrations = IntrinsicsPair.load_json("./data/initial_calibration.json")

    # run overlay
    overlay_and_save(images, gt_calibrations, output_folder, target = "RGB", show_individual_images=False)


if __name__ == "__main__":
    main()
