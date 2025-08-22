import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob


gt_calibrations = {
    "Thermal": {
        "K": [
            [1508.1604853684528, 0.0, 640.0],
            [0.0, 1511.9953924859087, 512.0],
            [0.0, 0.0, 1.0],
        ],
        # COLMAP OPENCV order: k1, k2, p1, p2  (no k3 in your file) → pad k3=0 for OpenCV
        "dist": [[-0.2936949101751592, 0.039961485318537757, 0.00014209738493819476, 0.0012181059706456816, 0.0]],
    },
    "RGB": {
        "K": [
            [2867.2449371834909, 0.0, 2000.0],
            [0.0, 2872.0170866340036, 1500.0],
            [0.0, 0.0, 1.0],
        ],
        "dist": [[0.12983522280449861, -0.23263871750808079, 0.00025880967117184307, -0.0026594579685052587, 0.0]],
    },
}


# --- utility function ---
def imshow(image, *args, **kwargs):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image, *args, **kwargs)
    plt.axis("off")


# --- read paired images ---
def read_images(rgb_dir: str, th_dir: str):
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))
    th_files = sorted(glob.glob(os.path.join(th_dir, "*.jpg")))

    if len(rgb_files) == 0 or len(th_files) == 0:
        raise ValueError("No images found in provided folders.")

    if len(rgb_files) != len(th_files):
        print(f"WARNING: mismatch in counts. RGB={len(rgb_files)}, Thermal={len(th_files)}")
        min_len = min(len(rgb_files), len(th_files))
        rgb_files = rgb_files[:min_len]
        th_files = th_files[:min_len]

    rgb_imgs = [cv2.imread(f) for f in rgb_files]
    th_imgs = [cv2.imread(f) for f in th_files]

    return {"RGB": rgb_imgs, "Thermal": th_imgs}, rgb_files, th_files


# --- undistortion canvas setup ---
dst_size = (2024, 1024)
f = min(*dst_size)
new_camera_matrix = np.array(
    [[f, 0.0, dst_size[0] / 2 - 0.5],
     [0.0, f, dst_size[1] / 2 - 0.5],
     [0.0, 0.0, 1.0]]
)


# --- main overlay function ---
def overlay_and_save(images, calibrations, out_dir, rgb_files, show_individual_images=False):
    os.makedirs(out_dir, exist_ok=True)

    num_frames = len(images["RGB"])
    for idx in range(num_frames):
        for cam in images.keys():
            calib = calibrations.get(cam)
            assert calib is not None, f"No calibration for {cam}"

            camera_matrix = np.array(calib["K"])
            dist_coefs = np.array(calib["dist"])
            img = images[cam][idx]

            mapx, mapy = cv2.initUndistortRectifyMap(
                camera_matrix, dist_coefs, None,
                new_camera_matrix, dst_size, cv2.CV_32FC1
            )
            dst = cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)

            if cam == "RGB":
                dst = cv2.Canny(dst.copy(), 100, 200)
                rgb_edge = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)
            else:
                thermal_img = dst.copy()
                cmp_img = thermal_img.copy()

        cmp_img[:, :, 1] = rgb_edge[:, :, 1]


        out_name = f"overlay_{idx:06d}.png"
        out_path = os.path.join(out_dir, out_name)

        cv2.imwrite(out_path, cmp_img)
        print(f"[OK] saved {out_path}")


#input paths
rgb_folder = r"./data/rgb"
thermal_folder = r"./data/thermal"
output_folder = r"./results/overlays"

# read images
images, rgb_files, th_files = read_images(rgb_folder, thermal_folder)

# run overlay
overlay_and_save(images, gt_calibrations, output_folder, rgb_files,  show_individual_images=False)