import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Any, Optional, List
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from utils import image_resize, read_images


class AnnoImage:
    def __init__(self, filename, filepath, _display_resolutions=None):
        self.img = None  # the opened image (OpenCV)
        self.scaled_img = None  # the scaled image (OpenCV)
        self.name = filename
        self.filepath = filepath
        if _display_resolutions is None:
            self._display_resolutions = (1200, 1200)
        else:
            self._display_resolutions = _display_resolutions

        # running things
        self.windowname = None  # the window name for OpenCV
        self.annotations = {}  # the annotations

        # verify file exists
        if not os.path.isfile(os.path.join(self.filepath, self.name)):
            raise Exception("File does not exist")

        # verify if the display resolutions are valid
        if self._display_resolutions[0] < 0 or self._display_resolutions[1] < 0:
            raise Exception("Display resolutions must be positive")

    # get image
    @property
    def display_image(self):
        # make sure that the image is loaded
        tmp = self.original_image

        # draw annotations
        for name, (x, y) in self.annotations.items():
            _x, _y = self.original_to_scaled(x, y)

            color = (0, 255, 0) if tmp.shape[2] == 3 else 255
            # draw circle
            cv2.circle(
                self.scaled_img,
                (_x, _y),
                5,
                color,
            )

            # draw a cross
            cv2.line(
                self.scaled_img,
                (_x - 5, _y),
                (_x + 5, _y),
                255,
            )
            cv2.line(
                self.scaled_img,
                (_x, _y - 5),
                (_x, _y + 5),
                255,
            )

            # draw text
            cv2.putText(
                self.scaled_img,
                str(name),
                (_x, _y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )

        return self.scaled_img

    @property
    def original_image(self):
        if self.img is None:
            self.img = cv2.imread(os.path.join(self.filepath, self.name))
            # resize image to fit display height
            self.scaled_img = image_resize(
                self.img, width=None, height=self._display_resolutions[1]
            )
        return self.img

    @property
    def display_resolutions(self):
        return self._display_resolutions

    @display_resolutions.setter
    def display_resolutions(self, value):
        self._display_resolutions = value
        self.scaled_img = image_resize(
            self.img, width=None, height=self._display_resolutions[1]
        )

    @property
    def width(self):
        return self.original_image.shape[1]

    @property
    def height(self):
        return self.original_image.shape[0]

    def scaled_to_original(self, x, y):
        w, h = self.width, self.height
        if self.img is not None:
            # get scaling factor
            scale_x = w / self.scaled_img.shape[1]
            scale_y = h / self.scaled_img.shape[0]
            return (int(x * scale_x), int(y * scale_y))
        else:
            return (-1, -1)

    def original_to_scaled(self, x, y):
        w, h = self.width, self.height

        if self.img is not None:
            # get scaling factor
            scale_x = self.scaled_img.shape[1] / w
            scale_y = self.scaled_img.shape[0] / h
            return (int(x * scale_x), int(y * scale_y))
        else:
            return (-1, -1)

    def addAnnotation(self, x, y, scaled=True, name=None):
        if name is None:
            name = len(self.annotations)
        self.annotations[name] = self.scaled_to_original(x, y) if scaled else (x, y)


class AnnoPair:
    def __init__(self, w_image, t_image):
        self.w_image = w_image
        self.t_image = t_image

    def __getitem__(self, key):
        if key == "W":
            return self.w_image
        elif key == "T":
            return self.t_image
        else:
            raise Exception("Invalid key")

    # items
    def items(self):
        return [("W", self.w_image), ("T", self.t_image)]

    # keys
    def keys(self):
        return ["W", "T"]

    # values
    def values(self):
        return [self.w_image, self.t_image]

    def __repr__(self):
        return "AnnoPair(W: {}, T: {})".format(self.w_image.name, self.t_image.name)

    @property
    def paired_annotations(self):
        return {
            name: {
                "W": self.w_image.annotations[name],
                "T": self.t_image.annotations[name],
            }
            for name in self.w_image.annotations.keys()
            # & self.t_image.annotations.keys()
        }


def cmap_per_label(labels, cmap="jet"):
    cmap = mpl.colormaps[cmap](np.linspace(0, 1, len(labels)))
    return {label: color for label, color in zip(sorted(labels, key=int), cmap)}


def plot_annotations(image, ax=plt):
    # get a color map
    cmap = cmap_per_label(image.annotations.keys())
    for name, (x, y) in image.annotations.items():
        ax.scatter(x, y, marker="+", color=cmap[name])
        ax.text(x - 2, y - 2, name, color=cmap[name])  # this adds colored shadow
        ax.text(x, y, name, color="white")


from matplotlib.patches import ConnectionPatch


def plot_pair(pair, ax=plt, draw_matches=False):
    axes = {"W": None, "T": None}
    for i, (TW, image) in enumerate(pair.items()):
        axes[TW] = ax.subplot(1, 2, i + 1)
        ax.imshow(cv2.cvtColor(image.original_image, cv2.COLOR_RGB2BGR))
        ax.title(image.name)
        plot_annotations(image, axes[TW])

    if draw_matches:
        cmap = cmap_per_label(image.annotations.keys())
        for label, anno_pair in pair.paired_annotations.items():

            con = ConnectionPatch(
                xyA=anno_pair["W"],
                coordsA=axes["W"].transData,
                xyB=anno_pair["T"],
                coordsB=axes["T"].transData,
                color=cmap[label],
            )

            plt.figure(1).add_artist(con)


# write annotations to XML file
# <annotation>
#  <image id="3" name="DJI_20230222081841_0001_T_frame0322.png" width="3840" height="2160">
#   <points label="C2" occluded="0" source="manual" points="1483.33,1922.18" z_order="0">
#    </points>
#    <points label="door" occluded="0" source="manual" points="2457.67,1735.01;2563.81,1738.33" z_order="0">
#    </points>


def annotation_to_xml(image, w_or_t=None):
    """Converts an image annotation to XML"""
    # create the file structure
    image_xml = ET.Element("image")
    # parse  id from frame DJI_20230222081841_0001_T_frame0322.png to 322
    # parse DJI_20230222080936_0007_V_frame000.png
    if w_or_t is None:
        T_or_W = image.name.split("_")[-2]
    else:
        T_or_W = w_or_t
    try:
        frame_id = int(image.name.split("_")[-1].split(".")[0][5:])
    except:
        # assume a file name like "0240.png"
        frame_id = int(image.name.split(".")[-2])
    video_id = 0
    try:
        video_id = int(image.name.split("_")[-3])
    except:
        pass
    id = video_id * 1000 + frame_id
    image_xml.set("id", f"{T_or_W}{id}")
    image_xml.set("name", image.name)
    image_xml.set("width", str(image.width))
    image_xml.set("height", str(image.height))

    for label, (x, y) in image.annotations.items():
        points = ET.SubElement(image_xml, "points")
        points.set("label", str(label))
        points.set("occluded", "0")
        points.set("source", "manual")
        points.set("points", f"{x},{y}")
        points.set("z_order", "0")

    return image_xml


def annotations_to_xml(pairs):
    """Converts a list of image annotations to XML"""
    # create the file structure
    annotation_xml = ET.Element("annotations")
    for pair in pairs:
        for w_or_t, image in pair.items():
            image_xml = annotation_to_xml(image, w_or_t)
            annotation_xml.append(image_xml)

    return annotation_xml


def save_annotations(pairs, filename):
    xml = annotations_to_xml(pairs)

    # write xml to file
    xmlstr = minidom.parseString(ET.tostring(xml)).toprettyxml(indent="   ")
    with open(filename, "w") as f:
        f.write(xmlstr)


# -----------------------------------------------------------------------------
# internal function(s) --------------------------------------------------------
def _retrieve_id_WT(xml_image):
    sid: str = xml_image.get("id", " ")
    name: str = xml_image.get("name", "")
    id = -1
    T_or_W = sid[0]
    # check if id starts with W or T e.g. "W36"
    if sid[0] == "W" or sid[0] == "T":
        id = int(sid[1:])  # remove W or T from id
    else:
        # assume we have a continous number for id and "frame0240T.png" as name
        if name == "":
            raise ValueError("Image name not present")

        # check if name ends with T or W
        T_or_W = name.split(".")[0][-1]
        if T_or_W != "W" and T_or_W != "T":
            raise ValueError("Image name does not end with W or T")

        # get the id from the name
        id = int(name.split("frame")[1].split(T_or_W + ".")[0])

    return id, T_or_W


def _add_points(xml_image, img, label_prefix=""):
    xml_points = (xml_point for xml_point in xml_image if xml_point.tag == "points")
    for points in xml_points:
        label = points.get("label")
        if points.get("outside", "0") == "0" and points.get("occluded", "0") == "0":
            pts = [
                float(i)
                for _2Dpts in points.get("points", "").split(";")
                for i in _2Dpts.split(",")
            ]
            pts = np.array(pts).reshape(-1, 2)
            pts = _sort_keypoints(pts).tolist()
            label_suffix = ""
            for i, pt in enumerate(pts):
                x, y = pt
                if len(pts) > 1:
                    label_suffix = f"_{i}"
                img.addAnnotation(
                    x, y, scaled=False, name=label_prefix + label + label_suffix
                )


def _sort_keypoints(points):
    """sort corner points of a rectangle in  the following order: top-left, top-right, bottom-left, bottom-right

    Args:
        points (list): a list of 4 points, each point is a list of 2 elements, [x, y]

    Returns:
        list: a list of  sorted points, each point is a list of 2 elements, [x, y]
    """

    if len(points) <= 1:
        return np.array(points)

    _points = np.array(points)

    # find the middle point
    middle_point = np.mean(_points, axis=0)

    if len(points) == 4:
        for pt in points:
            if pt[0] < middle_point[0] and pt[1] < middle_point[1]:
                # top left
                _points[0] = pt
            elif pt[0] > middle_point[0] and pt[1] < middle_point[1]:
                # top right
                _points[1] = pt
            elif pt[0] < middle_point[0] and pt[1] > middle_point[1]:
                # bottom left
                _points[2] = pt
            elif pt[0] > middle_point[0] and pt[1] > middle_point[1]:
                # bottom right
                _points[3] = pt
    else:  # 2,3, 5,6,...
        # sort by x coordinate
        _points = _points[_points[:, 0].argsort()]

    return _points


# load the annotations from the XML file
def load_annotations(filename, w_filepath, t_filepath):
    tree = ET.parse(filename)
    root = tree.getroot()

    pairs = {}
    # retrieve all xml image tags
    xml_images = (xml_image for xml_image in root if xml_image.tag == "image")
    for xml_image in xml_images:
        name = xml_image.get("name")
        id, T_or_W = _retrieve_id_WT(xml_image)
        img = AnnoImage(name, w_filepath if T_or_W == "W" else t_filepath)

        # check that the image exists and dimensions match
        if img.width != int(xml_image.get("width")) or img.height != int(
            xml_image.get("height")
        ):
            raise ValueError("Image dimensions do not match")

        # retrieve all tag==points children
        _add_points(xml_image, img)

        # furthermore there might be some skeleton nodes in the XML
        # retrieve all tag==skeleton children
        xml_skeletons = (
            xml_skeleton for xml_skeleton in xml_image if xml_skeleton.tag == "skeleton"
        )
        for xml_skeleton in xml_skeletons:
            _add_points(
                xml_skeleton,
                img,
                label_prefix=xml_skeleton.get("label", "skeleton") + "_",
            )

        pair = pairs.get(id, AnnoPair(None, None))
        if T_or_W == "W":
            pair.w_image = img
        elif T_or_W == "T":
            pair.t_image = img

        pairs[id] = pair

    # verify that all pairs have both images
    for id, pair in pairs.items():
        if pair.w_image is None or pair.t_image is None:
            raise ValueError(f"Pair {id} is missing an image")

    # convert dict to list
    return list(pairs.values())


def create_annotation_pairs(w_filepath: str, t_filepath: str, sampling: Optional[int] = None, display_resolutions: Optional[tuple] = None) -> List[AnnoPair]:
    """
    Create pairs of wide (RGB) and thermal image files for annotation purposes.
    
    Uses read_images internally to get matching files and creates AnnoPair objects with 
    AnnoImage instances based on sampling strategy. Files are matched by ensuring 
    their suffixes match (e.g., "frame_000022.jpg").
    
    :param w_filepath: Path to the directory containing wide (RGB) images
    :param t_filepath: Path to the directory containing thermal images  
    :param sampling: Number of pairs to create. If None, uses all available files if < 10,
                    otherwise uses half of available files. Must be > 0 if specified.
    :param display_resolutions: Optional tuple for display resolutions (width, height). 
                               If None, uses default (1200, 1200).
    :return: List of AnnoPair objects containing AnnoImage instances
    :raises ValueError: If directories don't contain matching files or sampling is invalid
    """
    # Use read_images to get the file lists
    _, w_files, t_files = read_images(w_filepath, t_filepath)
    
    # Extract just the filenames from the full paths
    w_filenames = [os.path.basename(f) for f in w_files]
    t_filenames = [os.path.basename(f) for f in t_files]
    
    if len(w_filenames) != len(t_filenames):
        raise ValueError(f"W {w_filepath} and T {t_filepath} files do not match")
    
    # Determine sampling strategy
    if sampling is None:
        sampling = len(w_filenames)
    elif sampling <= 0:
        raise ValueError("Sampling must be greater than 0")
    else:
        sampling = min(sampling, len(w_filenames))
    
    # Create pairs with sampling
    pairs: List[AnnoPair] = []
    step_size = max(1, len(w_filenames) // sampling)
    
    for idx in range(0, len(w_filenames), step_size):
        if len(pairs) >= sampling:
            break
            
        w_file = w_filenames[idx]
        t_file = t_filenames[idx]
        
        # Ensure that the last 10 characters (e.g., "_frame0270.png") are the same
        if w_file[-10:] != t_file[-10:]:
            raise ValueError(f"W {w_file} and T {t_file} files do not match")
        
        # Create AnnoImage objects
        w_image = AnnoImage(w_file, w_filepath, display_resolutions)
        t_image = AnnoImage(t_file, t_filepath, display_resolutions)
        
        # Create AnnoPair object
        pair = AnnoPair(w_image, t_image)
        pairs.append(pair)
    
    return pairs


def points_from_pairs(pairs):
    """
    Extract points from a list of AnnoPair objects.
    
    :param pairs: List of AnnoPair objects
    :return: List of points
    """
    # construct points with numpy
    T_points = np.zeros([0,2])
    W_points = np.zeros([0,2])
    TW_names = []

    for i, pair in enumerate(pairs):

        T = pair["T"]
        W = pair["W"]

        # get the annotations
        for label, (tx, ty) in T.annotations.items():

            wx, wy = W.annotations[label]

            # add points
            # W (wide)
            W_points = np.concatenate((W_points, [[wx, wy]]))
            # T (thermal)
            T_points = np.concatenate((T_points, [[tx, ty]]))

            # add names
            TW_names.append(T.name + "_" + label)

    # reshape the points to (n, 1, 2)
    T_points = T_points.reshape(-1, 1, 2)
    W_points = W_points.reshape(-1, 1, 2)

    return T_points, W_points, TW_names
