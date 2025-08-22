"""Utility functions for image processing and camera calibration."""

from typing import Any, Optional
import os
import glob

import cv2
import numpy as np
import numpy.typing as npt

# --- read paired images ---
def read_images(rgb_dir: str, th_dir: str):
    # Get both jpg and png files
    rgb_jpg_files = glob.glob(os.path.join(rgb_dir, "*.jpg"))
    rgb_png_files = glob.glob(os.path.join(rgb_dir, "*.png"))
    rgb_files = sorted(rgb_jpg_files + rgb_png_files)
    
    th_jpg_files = glob.glob(os.path.join(th_dir, "*.jpg"))
    th_png_files = glob.glob(os.path.join(th_dir, "*.png"))
    th_files = sorted(th_jpg_files + th_png_files)

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




def equal_images(frame1: npt.NDArray[Any], frame2: npt.NDArray[Any]) -> bool:
    """
    Method allowing to check if two images are equal
    :param frame1: reference
    :param frame2: to be compared
    :return: true iff images are equal in all channels
    """
    difference = cv2.subtract(frame1, frame2)
    channels = cv2.split(difference)
    for channel in channels:
        if cv2.countNonZero(channel) != 0:
            return False
    return True


def image_equality(frame1: npt.NDArray[Any], frame2: npt.NDArray[Any]) -> float:
    """
    Method allowing to check how equal images are (in percentage)
    :param frame1: reference
    :param frame2: to be compared
    :return: 0 iff images are equal in all channels; and 1 if not one pixel is equal
    """
    difference = cv2.subtract(frame1, frame2)
    channels = cv2.split(difference)
    different_pixels = 0
    for channel in channels:
        different_pixels += cv2.countNonZero(channel)

    total_pixels = 1
    for shape in frame1.shape:
        total_pixels *= shape

    return different_pixels / total_pixels


def image_equality_check(
    frame1: npt.NDArray[Any], frame2: npt.NDArray[Any], equality_threshold: float = 0.05
) -> bool:
    """
    Method allowing to check how equal images are (in percentage)
    :param frame1: reference
    :param frame2: to be compared
    :param equality_threshold: Threshold used to determine if images are equal. Default: 5%
    :return: true iff equality_threshold is not exceeded
    """
    return image_equality(frame1, frame2) < equality_threshold


def ssd(frame1: npt.NDArray[Any], frame2: npt.NDArray[Any]) -> int:
    """
    Method for calculating the sum of squared error of two images
    :param frame1: reference
    :param frame2: to be compared
    :return: Sum of squared error
    """
    if len(frame1.shape) == 3:
        channels = frame1.shape[2]
        return np.sum((frame1[:, :, 0:channels] - frame2[:, :, 0:channels]) ** 2)
    else:
        return np.sum((frame1[:, :] - frame2[:, :]) ** 2)


def ssd_equal_images(frame1: npt.NDArray[Any], frame2: npt.NDArray[Any]) -> bool:
    """
    Method allowing to check if two images are equal based on the sum of squared error metric
    :param frame1: reference
    :param frame2: to be compared
    :return: true iff images are equal in all channels
    """
    return ssd(frame1, frame2) == 0


def image_resize(
    image: npt.NDArray[Any],
    width: Optional[int] = None,
    height: Optional[int] = None,
    inter: int = cv2.INTER_AREA,
) -> npt.NDArray[Any]:
    """
    Resizes an image to a given width and height. If only one of the two is specified, the image is resized proportionally.
    :param image (npt.NDArray[Any]): image to be resized
    :param width (Optional[int], optional): width of the resized image. Defaults to None.
    :param height (Optional[int], optional): height of the resized image. Defaults to None.
    :param inter (cv2.INTER_AREA, optional): interpolation method. Defaults to cv2.INTER_AREA.
    :return: resized image
    """
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then just return the original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None and height is not None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    elif width is not None and height is None:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    else:
        # this should not happen
        raise ValueError("resize_image: something went wrong!")

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


# from : https://gist.github.com/woolpeeker/d7e1821e1b5c556b32aafe10b7a1b7e8

# drawMatches numpy version
def draw_matches(img1, kp1, img2, kp2, matches, color=None): 
    """Draws lines between matching keypoints of two images.  
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles 
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: ndarray [n1, 2]
        img2: An openCV image ndarray of the same format and with the same 
        element type as img1.
        kp2: ndarray [n2, 2]
        matches: ndarray [n_match, 2]
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.  
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.  
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))  
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    
    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 15
    thickness = 2
    if color:
        c = color
    for m in matches:
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color: 
            c = np.random.randint(0,256,3) if len(img1.shape) == 3 else np.random.randint(0,256)
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(kp1[m[0]]).astype(int))
        end2 = tuple(np.round(kp2[m[1]]).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(new_img, end1, end2, c, thickness)
        cv2.circle(new_img, end1, r, c, thickness)
        cv2.circle(new_img, end2, r, c, thickness)
    return new_img
