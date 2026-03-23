"""
Camera Calibration Web Application - Backend
FastAPI server for thermal/RGB camera pair calibration.
"""

import os
import io
import json
import glob
import uuid
import shutil
import base64
import traceback
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Optional, List, Dict, Any
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.intrinsics import Intrinsics, IntrinsicsPair

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Camera Calibration Tool")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_ROOT = os.environ.get("DATA_ROOT", "/app/data")
os.makedirs(DATA_ROOT, exist_ok=True)


def project_dir(project_id: str) -> str:
    return os.path.join(DATA_ROOT, project_id)


# ---------------------------------------------------------------------------
# Project management
# ---------------------------------------------------------------------------
@app.post("/api/projects")
async def create_project(name: str = Form("Untitled")):
    pid = uuid.uuid4().hex[:12]
    pdir = project_dir(pid)
    os.makedirs(os.path.join(pdir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(pdir, "thermal"), exist_ok=True)
    os.makedirs(os.path.join(pdir, "results"), exist_ok=True)
    meta = {"id": pid, "name": name, "pairs": [], "annotations": {}, "initial_calibration": None}
    with open(os.path.join(pdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


@app.get("/api/projects")
async def list_projects():
    projects = []
    if not os.path.isdir(DATA_ROOT):
        return projects
    for entry in sorted(os.listdir(DATA_ROOT)):
        meta_path = os.path.join(DATA_ROOT, entry, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                projects.append(json.load(f))
    return projects


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    meta_path = os.path.join(project_dir(project_id), "meta.json")
    if not os.path.isfile(meta_path):
        raise HTTPException(404, "Project not found")
    with open(meta_path) as f:
        return json.load(f)


@app.patch("/api/projects/{project_id}")
async def rename_project(project_id: str, body: dict):
    meta_path = os.path.join(project_dir(project_id), "meta.json")
    if not os.path.isfile(meta_path):
        raise HTTPException(404, "Project not found")
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "Name must not be empty")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["name"] = name
    _save_meta(project_id, meta)
    return meta


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    pdir = project_dir(project_id)
    if os.path.isdir(pdir):
        shutil.rmtree(pdir)
    return {"ok": True}


def _save_meta(project_id: str, meta: dict):
    with open(os.path.join(project_dir(project_id), "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# File upload (images)
# ---------------------------------------------------------------------------
@app.post("/api/projects/{project_id}/upload-images")
async def upload_images(
    project_id: str,
    channel: str = Form(...),  # "rgb" or "thermal"
    files: List[UploadFile] = File(...),
):
    pdir = project_dir(project_id)
    meta_path = os.path.join(pdir, "meta.json")
    if not os.path.isfile(meta_path):
        raise HTTPException(404, "Project not found")

    target_dir = os.path.join(pdir, channel)
    os.makedirs(target_dir, exist_ok=True)

    saved = []
    for f in files:
        ext = os.path.splitext(f.filename or "img.jpg")[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            continue
        dest = os.path.join(target_dir, f.filename)
        with open(dest, "wb") as out:
            content = await f.read()
            out.write(content)
        saved.append(f.filename)

    # rebuild pairs
    _rebuild_pairs(project_id)
    return {"saved": saved}


# ---------------------------------------------------------------------------
# File upload (videos) — extract central frame
# ---------------------------------------------------------------------------
@app.post("/api/projects/{project_id}/upload-video")
async def upload_video(
    project_id: str,
    channel: str = Form(...),
    file: UploadFile = File(...),
):
    pdir = project_dir(project_id)
    meta_path = os.path.join(pdir, "meta.json")
    if not os.path.isfile(meta_path):
        raise HTTPException(404, "Project not found")

    # Save video to temp location
    tmp_path = os.path.join(pdir, f"_tmp_{channel}_{file.filename}")
    with open(tmp_path, "wb") as out:
        content = await file.read()
        out.write(content)

    # Extract central frame
    try:
        frame_path = _extract_central_frame(tmp_path, pdir, channel, file.filename)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    _rebuild_pairs(project_id)
    return {"extracted_frame": os.path.basename(frame_path)}


def _extract_central_frame(video_path: str, pdir: str, channel: str, original_name: str) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(400, f"Cannot open video: {original_name}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise HTTPException(400, f"Video has no frames: {original_name}")

    central_idx = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, central_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise HTTPException(400, f"Failed to read central frame from: {original_name}")

    stem = Path(original_name).stem
    out_name = f"{stem}_frame{central_idx:06d}.jpg"
    out_path = os.path.join(pdir, channel, out_name)
    cv2.imwrite(out_path, frame)
    return out_path


# ---------------------------------------------------------------------------
# Upload video pairs (both at once)
# ---------------------------------------------------------------------------
@app.post("/api/projects/{project_id}/upload-video-pair")
async def upload_video_pair(
    project_id: str,
    rgb_file: UploadFile = File(...),
    thermal_file: UploadFile = File(...),
):
    pdir = project_dir(project_id)
    meta_path = os.path.join(pdir, "meta.json")
    if not os.path.isfile(meta_path):
        raise HTTPException(404, "Project not found")

    results = {}
    for channel, f in [("rgb", rgb_file), ("thermal", thermal_file)]:
        tmp_path = os.path.join(pdir, f"_tmp_{channel}_{f.filename}")
        with open(tmp_path, "wb") as out:
            content = await f.read()
            out.write(content)
        try:
            frame_path = _extract_central_frame(tmp_path, pdir, channel, f.filename)
            results[channel] = os.path.basename(frame_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    _rebuild_pairs(project_id)
    return results


# ---------------------------------------------------------------------------
# Pair management
# ---------------------------------------------------------------------------
def _rebuild_pairs(project_id: str):
    """Match RGB and thermal images by sorted filename order and update meta."""
    pdir = project_dir(project_id)
    meta_path = os.path.join(pdir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    rgb_dir = os.path.join(pdir, "rgb")
    th_dir = os.path.join(pdir, "thermal")

    rgb_files = sorted(
        [fn for fn in os.listdir(rgb_dir) if _is_image(fn)]
    )
    th_files = sorted(
        [fn for fn in os.listdir(th_dir) if _is_image(fn)]
    )

    n = min(len(rgb_files), len(th_files))
    pairs = []
    for i in range(n):
        pairs.append({"rgb": rgb_files[i], "thermal": th_files[i], "id": i})

    meta["pairs"] = pairs
    _save_meta(project_id, meta)


def _is_image(fn: str) -> bool:
    return fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))


@app.get("/api/projects/{project_id}/pairs")
async def get_pairs(project_id: str):
    meta_path = os.path.join(project_dir(project_id), "meta.json")
    if not os.path.isfile(meta_path):
        raise HTTPException(404, "Project not found")
    with open(meta_path) as f:
        meta = json.load(f)
    return meta.get("pairs", [])


@app.delete("/api/projects/{project_id}/pairs/{pair_id}")
async def delete_pair(project_id: str, pair_id: int):
    pdir = project_dir(project_id)
    meta_path = os.path.join(pdir, "meta.json")
    if not os.path.isfile(meta_path):
        raise HTTPException(404, "Project not found")
    with open(meta_path) as f:
        meta = json.load(f)

    pair = next((p for p in meta.get("pairs", []) if p["id"] == pair_id), None)
    if pair is None:
        raise HTTPException(404, "Pair not found")

    # Remove image files (ignore if already missing)
    for channel in ("rgb", "thermal"):
        img_path = os.path.join(pdir, channel, pair[channel])
        if os.path.isfile(img_path):
            os.remove(img_path)

    # Re-key annotations: pairs with id > pair_id shift down by 1
    old_ann = meta.get("annotations", {})
    new_ann: Dict[str, Any] = {}
    for k, v in old_ann.items():
        try:
            k_int = int(k)
        except ValueError:
            new_ann[k] = v
            continue
        if k_int == pair_id:
            pass  # deleted
        elif k_int > pair_id:
            new_ann[str(k_int - 1)] = v
        else:
            new_ann[k] = v
    meta["annotations"] = new_ann
    _save_meta(project_id, meta)

    # Rebuild pairs list from remaining files
    _rebuild_pairs(project_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Serve images
# ---------------------------------------------------------------------------
@app.get("/api/projects/{project_id}/image/{channel}/{filename}")
async def get_image(project_id: str, channel: str, filename: str):
    path = os.path.join(project_dir(project_id), channel, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "Image not found")
    return FileResponse(path)


@app.get("/api/projects/{project_id}/image-info/{channel}/{filename}")
async def get_image_info(project_id: str, channel: str, filename: str):
    path = os.path.join(project_dir(project_id), channel, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "Image not found")
    img = cv2.imread(path)
    h, w = img.shape[:2]
    return {"width": w, "height": h, "filename": filename, "channel": channel}


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------
@app.get("/api/projects/{project_id}/annotations")
async def get_annotations(project_id: str):
    meta_path = os.path.join(project_dir(project_id), "meta.json")
    if not os.path.isfile(meta_path):
        raise HTTPException(404, "Project not found")
    with open(meta_path) as f:
        meta = json.load(f)
    return meta.get("annotations", {})


@app.post("/api/projects/{project_id}/annotations")
async def save_annotations(project_id: str, body: dict):
    """
    body: {
        "annotations": {
            "<pair_id>": {
                "points": [
                    {"label": "0", "rgb": [x, y], "thermal": [x, y]},
                    ...
                ]
            },
            ...
        }
    }
    """
    meta_path = os.path.join(project_dir(project_id), "meta.json")
    if not os.path.isfile(meta_path):
        raise HTTPException(404, "Project not found")
    with open(meta_path) as f:
        meta = json.load(f)

    meta["annotations"] = body.get("annotations", {})
    _save_meta(project_id, meta)

    # Also write XML for compatibility
    _write_annotations_xml(project_id, meta)

    return {"ok": True}


@app.post("/api/projects/{project_id}/upload-annotations")
async def upload_annotations_xml(project_id: str, file: UploadFile = File(...)):
    """Import annotations from a previously exported pairs.xml file.
    Points are matched across channels by label and merged into meta.json.
    Existing annotations for pairs not present in the XML are preserved.
    """
    meta_path = os.path.join(project_dir(project_id), "meta.json")
    if not os.path.isfile(meta_path):
        raise HTTPException(404, "Project not found")
    with open(meta_path) as f:
        meta = json.load(f)

    content = await file.read()
    try:
        root = ET.fromstring(content)
    except ET.ParseError as e:
        raise HTTPException(400, f"Invalid XML: {e}")

    # Parse image elements — id prefix "W" = rgb, "T" = thermal
    rgb_pts:     Dict[str, Dict[str, List[float]]] = {}
    thermal_pts: Dict[str, Dict[str, List[float]]] = {}

    for img_el in root.findall("image"):
        img_id = img_el.get("id", "")
        if img_id.startswith("W"):
            pair_id, target = img_id[1:], rgb_pts.setdefault(img_id[1:], {})
        elif img_id.startswith("T"):
            pair_id, target = img_id[1:], thermal_pts.setdefault(img_id[1:], {})
        else:
            continue

        for pt_el in img_el.findall("points"):
            label = pt_el.get("label", "")
            pts_str = pt_el.get("points", "")
            try:
                x, y = (float(v) for v in pts_str.split(","))
                target[label] = [x, y]
            except (ValueError, TypeError):
                continue

    # Build annotation objects — only include points present in BOTH channels
    new_annotations: Dict[str, Any] = {}
    for pair_id in set(rgb_pts) | set(thermal_pts):
        rgb_map     = rgb_pts.get(pair_id, {})
        thermal_map = thermal_pts.get(pair_id, {})
        labels = sorted(set(rgb_map) & set(thermal_map),
                        key=lambda l: int(l) if l.isdigit() else l)
        if not labels:
            continue
        new_annotations[pair_id] = {
            "points": [
                {"label": lbl, "rgb": rgb_map[lbl], "thermal": thermal_map[lbl]}
                for lbl in labels
            ]
        }

    # Merge: replace only pairs present in the XML; keep everything else
    merged = meta.get("annotations", {})
    merged.update(new_annotations)
    meta["annotations"] = merged
    _save_meta(project_id, meta)
    _write_annotations_xml(project_id, meta)

    return {"annotations": meta["annotations"], "imported": len(new_annotations)}


def _write_annotations_xml(project_id: str, meta: dict):
    """Write annotations in the original XML format for compatibility."""
    pdir = project_dir(project_id)
    root = ET.Element("annotations")

    pairs = meta.get("pairs", [])
    annotations = meta.get("annotations", {})

    for pair in pairs:
        pair_id = str(pair["id"])
        if pair_id not in annotations:
            continue
        pts = annotations[pair_id].get("points", [])
        if not pts:
            continue

        # Get image dimensions
        for channel, tw_key in [("rgb", "W"), ("thermal", "T")]:
            img_path = os.path.join(pdir, channel, pair[channel])
            if not os.path.isfile(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            img_el = ET.SubElement(root, "image")
            img_el.set("id", f"{tw_key}{pair_id}")
            img_el.set("name", pair[channel])
            img_el.set("width", str(w))
            img_el.set("height", str(h))

            coord_key = "rgb" if channel == "rgb" else "thermal"
            for pt in pts:
                coord = pt.get(coord_key)
                if not coord:   # skip half-placed points missing this channel
                    continue
                points_el = ET.SubElement(img_el, "points")
                points_el.set("label", str(pt["label"]))
                points_el.set("occluded", "0")
                points_el.set("source", "manual")
                points_el.set("points", f"{coord[0]},{coord[1]}")
                points_el.set("z_order", "0")

    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with open(os.path.join(pdir, "pairs.xml"), "w") as f:
        f.write(xmlstr)


# ---------------------------------------------------------------------------
# Initial calibration upload
# ---------------------------------------------------------------------------
@app.post("/api/projects/{project_id}/initial-calibration")
async def upload_initial_calibration(project_id: str, body: dict):
    meta_path = os.path.join(project_dir(project_id), "meta.json")
    if not os.path.isfile(meta_path):
        raise HTTPException(404, "Project not found")
    with open(meta_path) as f:
        meta = json.load(f)

    meta["initial_calibration"] = body
    _save_meta(project_id, meta)
    return {"ok": True}


@app.delete("/api/projects/{project_id}/initial-calibration")
async def delete_initial_calibration(project_id: str):
    meta_path = os.path.join(project_dir(project_id), "meta.json")
    if not os.path.isfile(meta_path):
        raise HTTPException(404, "Project not found")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["initial_calibration"] = None
    _save_meta(project_id, meta)
    return {"ok": True}


@app.get("/api/projects/{project_id}/initial-calibration")
async def get_initial_calibration(project_id: str):
    meta_path = os.path.join(project_dir(project_id), "meta.json")
    if not os.path.isfile(meta_path):
        raise HTTPException(404, "Project not found")
    with open(meta_path) as f:
        meta = json.load(f)
    return meta.get("initial_calibration")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def _estimate_intrinsics_from_image(pdir: str, pairs: list, channel: str) -> Intrinsics:
    """Estimate reasonable default intrinsics from image dimensions.

    Assumes a typical drone camera with ~70-80° horizontal FoV for RGB
    and ~40-50° for thermal.  The principal point is placed at the image
    centre, distortion is initialised to zero.
    """
    import math

    fname = pairs[0]["rgb"] if channel == "rgb" else pairs[0]["thermal"]
    img_path = os.path.join(pdir, channel, fname)
    img = cv2.imread(img_path)
    if img is None:
        raise HTTPException(400, f"Cannot read image: {fname}")
    h, w = img.shape[:2]

    # Heuristic: assume ~73° hFoV for RGB, ~45° for thermal
    hfov_deg = 73.0 if channel == "rgb" else 45.0
    fx = (w / 2.0) / math.tan(math.radians(hfov_deg / 2.0))
    fy = fx  # square pixels
    cx, cy = w / 2.0, h / 2.0

    return Intrinsics(
        mtx=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64),
        dist=np.zeros(5, dtype=np.float64),
        name="Thermal" if channel == "thermal" else "Wide",
    )


@app.post("/api/projects/{project_id}/calibrate")
async def run_calibration(project_id: str):
    pdir = project_dir(project_id)
    meta_path = os.path.join(pdir, "meta.json")
    if not os.path.isfile(meta_path):
        raise HTTPException(404, "Project not found")
    with open(meta_path) as f:
        meta = json.load(f)

    pairs = meta.get("pairs", [])
    annotations = meta.get("annotations", {})

    if not pairs:
        raise HTTPException(400, "No image pairs in project")

    # Collect all point correspondences
    T_points_list = []
    W_points_list = []
    names = []

    for pair in pairs:
        pair_id = str(pair["id"])
        if pair_id not in annotations:
            continue
        pts = annotations[pair_id].get("points", [])
        for pt in pts:
            rgb_xy = pt.get("rgb")
            th_xy = pt.get("thermal")
            if rgb_xy and th_xy:
                W_points_list.append(rgb_xy)
                T_points_list.append(th_xy)
                names.append(f"{pair['rgb']}_{pt['label']}")

    if len(T_points_list) < 4:
        raise HTTPException(400, f"Need at least 4 point correspondences, got {len(T_points_list)}")

    T_points = np.array(T_points_list, dtype=np.float64).reshape(-1, 1, 2)
    W_points = np.array(W_points_list, dtype=np.float64).reshape(-1, 1, 2)

    # --- Build or load initial calibration ---
    initial_calib_data = meta.get("initial_calibration")
    used_auto_estimate = False

    if initial_calib_data:
        try:
            initial_calib = IntrinsicsPair.from_dict(initial_calib_data)
        except Exception as e:
            raise HTTPException(400, f"Invalid initial calibration: {e}")
    else:
        # Auto-estimate from image dimensions
        try:
            rgb_intr = _estimate_intrinsics_from_image(pdir, pairs, "rgb")
            th_intr = _estimate_intrinsics_from_image(pdir, pairs, "thermal")
            initial_calib = IntrinsicsPair(th_intr, rgb_intr)
            used_auto_estimate = True
        except Exception as e:
            raise HTTPException(400, f"Cannot auto-estimate intrinsics: {e}")

    # --- Homography ---
    M, mask = cv2.findHomography(W_points, T_points, cv2.RANSAC, 15)
    if M is None:
        raise HTTPException(400, "Homography computation failed")

    inlier_mask = mask.ravel().tolist()
    W_warped = cv2.perspectiveTransform(W_points, M)
    homography_mse = float(np.sum((W_warped - T_points) ** 2) / T_points.shape[0])

    # --- Optimization ---
    new_camera_matrix = initial_calib["RGB"].mtx.copy()
    cameras = ["Wide", "Thermal"]

    def warp_points(pts, camera_matrix, dist_coefs=None, new_cm=None):
        if dist_coefs is None:
            dist_coefs = np.zeros((1, 5))
        if new_cm is None:
            new_cm = camera_matrix
        pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.undistortPoints(pts, camera_matrix, dist_coefs, P=new_cm).reshape(-1, 1, 2)

    def mse_func(x):
        th_fx, th_fy, th_cx, th_cy = x[:4]
        th_dist = np.zeros(5)
        for i, v in enumerate(x[4:9]):
            th_dist[i] = v

        new_pts = {}
        for camera in cameras:
            if camera == "Thermal":
                cm = np.array([[th_fx, 0.0, th_cx], [0.0, th_fy, th_cy], [0.0, 0.0, 1.0]])
                dc = th_dist.reshape(1, -1)
                img_pts = T_points
            else:
                cm = initial_calib["RGB"].mtx
                dc = initial_calib["RGB"].dist.reshape(1, -1)
                img_pts = W_points
            new_pts[camera] = warp_points(img_pts, cm, dc, new_camera_matrix)

        stacked = np.concatenate(list(new_pts.values()), axis=1)
        return float(np.sum(np.diff(stacked, axis=1) ** 2, axis=2).mean())

    # Initial guess from provided or estimated calibration
    tic = initial_calib["Thermal"]
    x0 = [tic.fx, tic.fy, tic.cx, tic.cy] + list(tic.dist[:5])
    # Pad to at least 9 params (4 intrinsics + 5 dist)
    while len(x0) < 9:
        x0.append(0.0)

    # Run optimization
    n_iters = 10
    for _ in range(n_iters):
        res = minimize(mse_func, x0, method="Nelder-Mead",
                       options={"maxiter": 50000, "disp": False}, tol=1e-4)
        x0 = res.x.tolist()

    opt_mse = float(res.fun)

    # Build optimized calibration
    opt_calib = initial_calib.copy()
    opt_calib["Thermal"].set(
        fx=res.x[0], fy=res.x[1], cx=res.x[2], cy=res.x[3],
        dist=res.x[4:9],
    )

    result_path = os.path.join(pdir, "results", "calibration.json")
    opt_calib.save_json(result_path)

    # Generate overlays
    overlay_images = _generate_overlays(pdir, pairs, opt_calib)

    result = {
        "homography": M.tolist(),
        "homography_mse": homography_mse,
        "inlier_mask": inlier_mask,
        "optimized_mse": opt_mse,
        "optimized_calibration": opt_calib.to_dict(),
        "initial_calibration_used": initial_calib.to_dict(),
        "used_auto_estimate": used_auto_estimate,
        "num_points": len(T_points_list),
        "overlay_images": overlay_images,
    }

    # Save result summary
    with open(os.path.join(pdir, "results", "result.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def _generate_overlays(pdir: str, pairs: list, calibrations: IntrinsicsPair) -> list:
    """Generate overlay images showing alignment quality."""
    target_cm = calibrations["RGB"].mtx
    dst_size = np.array([int(calibrations["RGB"].cx * 2), int(calibrations["RGB"].cy * 2)], dtype=int)
    overlay_names = []

    for pair in pairs:
        rgb_path = os.path.join(pdir, "rgb", pair["rgb"])
        th_path = os.path.join(pdir, "thermal", pair["thermal"])
        if not os.path.isfile(rgb_path) or not os.path.isfile(th_path):
            continue

        images = {
            "RGB": cv2.imread(rgb_path),
            "Thermal": cv2.imread(th_path),
        }

        results = {}
        for cam, img in images.items():
            if img is None:
                continue
            calib = calibrations[cam]
            mapx, mapy = cv2.initUndistortRectifyMap(
                calib.mtx, calib.dist.reshape(1, -1), None,
                target_cm, dst_size, cv2.CV_32FC1,
            )
            results[cam] = cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)

        if "RGB" in results and "Thermal" in results:
            rgb_edge = cv2.cvtColor(cv2.Canny(results["RGB"], 100, 200), cv2.COLOR_GRAY2RGB)
            overlay = results["Thermal"].copy()
            overlay[:, :, 1] = rgb_edge[:, :, 1]

            out_name = f"overlay_{pair['id']:04d}.jpg"
            cv2.imwrite(os.path.join(pdir, "results", out_name), overlay)
            overlay_names.append(out_name)

    return overlay_names


@app.get("/api/projects/{project_id}/result-image/{filename}")
async def get_result_image(project_id: str, filename: str):
    path = os.path.join(project_dir(project_id), "results", filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "Result image not found")
    return FileResponse(path)


@app.get("/api/projects/{project_id}/calibration-result")
async def get_calibration_result(project_id: str):
    path = os.path.join(project_dir(project_id), "results", "result.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


@app.get("/api/projects/{project_id}/download-calibration")
async def download_calibration(project_id: str):
    path = os.path.join(project_dir(project_id), "results", "calibration.json")
    if not os.path.isfile(path):
        raise HTTPException(404, "No calibration result yet")
    return FileResponse(path, filename="calibration.json", media_type="application/json")


@app.get("/api/projects/{project_id}/download-calibration/{camera}")
async def download_single_calibration(project_id: str, camera: str):
    """Download a single camera's calibration as a flat Intrinsics JSON.

    camera: 'thermal' or 'rgb'
    """
    path = os.path.join(project_dir(project_id), "results", "calibration.json")
    if not os.path.isfile(path):
        raise HTTPException(404, "No calibration result yet")

    try:
        calib_pair = IntrinsicsPair.load_json(path)
    except Exception as e:
        raise HTTPException(500, f"Failed to read calibration: {e}")

    cam_lower = camera.lower()
    if cam_lower in ("thermal", "t"):
        intr = calib_pair.thermal
        filename = "calibration_thermal.json"
    elif cam_lower in ("rgb", "wide", "w"):
        intr = calib_pair.wide
        filename = "calibration_rgb.json"
    else:
        raise HTTPException(400, f"Unknown camera '{camera}'. Use 'thermal' or 'rgb'.")

    return JSONResponse(
        content=intr.to_dict(),
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/projects/{project_id}/download-annotations")
async def download_annotations(project_id: str):
    path = os.path.join(project_dir(project_id), "pairs.xml")
    if not os.path.isfile(path):
        raise HTTPException(404, "No annotations yet")
    return FileResponse(path, filename="pairs.xml", media_type="application/xml")


# ---------------------------------------------------------------------------
# Point-distribution heatmap
# ---------------------------------------------------------------------------
@app.get("/api/projects/{project_id}/point-heatmap")
async def point_heatmap(project_id: str, rows: int = 8, cols: int = 8):
    """
    Return a grid heatmap of point density for each channel (rgb / thermal).
    Coordinates are normalised to [0, 1] before binning so heatmaps are
    resolution-independent and comparable across pairs.

    Returns:
        {
          "rows": int, "cols": int,
          "rgb":     { "grid": [[...], ...], "max": int, "total": int },
          "thermal": { "grid": [[...], ...], "max": int, "total": int },
          "points_rgb":     [[nx, ny], ...],   # normalised coords
          "points_thermal": [[nx, ny], ...],
        }
    """
    meta_path = os.path.join(project_dir(project_id), "meta.json")
    if not os.path.isfile(meta_path):
        raise HTTPException(404, "Project not found")
    with open(meta_path) as f:
        meta = json.load(f)

    pdir = project_dir(project_id)
    pairs = meta.get("pairs", [])
    annotations = meta.get("annotations", {})

    rows = max(1, min(rows, 64))
    cols = max(1, min(cols, 64))

    # Cache image dimensions per pair so we can normalise
    def _img_dims(pair, channel):
        fname = pair["rgb"] if channel == "rgb" else pair["thermal"]
        path = os.path.join(pdir, channel, fname)
        if not os.path.isfile(path):
            return None
        img = cv2.imread(path)
        if img is None:
            return None
        h, w = img.shape[:2]
        return (w, h)

    dims_cache: Dict[str, Any] = {}

    def get_dims(pair, channel):
        key = f"{pair['id']}_{channel}"
        if key not in dims_cache:
            dims_cache[key] = _img_dims(pair, channel)
        return dims_cache[key]

    result: Dict[str, Any] = {}
    for channel in ("rgb", "thermal"):
        grid = [[0] * cols for _ in range(rows)]
        norm_points = []

        for pair in pairs:
            pair_id = str(pair["id"])
            if pair_id not in annotations:
                continue
            dims = get_dims(pair, channel)
            if dims is None:
                continue
            w, h = dims

            for pt in annotations[pair_id].get("points", []):
                coord = pt.get(channel)
                if coord is None:
                    continue
                nx = coord[0] / w  # normalised x  [0..1]
                ny = coord[1] / h  # normalised y  [0..1]
                norm_points.append([round(nx, 5), round(ny, 5)])
                ci = min(int(nx * cols), cols - 1)
                ri = min(int(ny * rows), rows - 1)
                grid[ri][ci] += 1

        flat = [v for row in grid for v in row]
        result[channel] = {
            "grid": grid,
            "max": max(flat) if flat else 0,
            "total": sum(flat),
        }
        result[f"points_{channel}"] = norm_points

    result["rows"] = rows
    result["cols"] = cols
    return result


# ---------------------------------------------------------------------------
# Single-camera calibration (checkerboard)
# ---------------------------------------------------------------------------
SINGLE_CALIB_ROOT = os.path.join(DATA_ROOT, "_single")


def _single_project_dir(project_id: str) -> str:
    return os.path.join(SINGLE_CALIB_ROOT, project_id)


def _save_single_meta(project_id: str, meta: dict):
    pdir = _single_project_dir(project_id)
    with open(os.path.join(pdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


@app.post("/api/single-calib/projects")
async def create_single_project(name: str = Form("Single Calib")):
    pid = uuid.uuid4().hex[:12]
    pdir = _single_project_dir(pid)
    os.makedirs(os.path.join(pdir, "images"), exist_ok=True)
    meta = {
        "id": pid, "name": name,
        "images": [],
        "calibration": None,
    }
    with open(os.path.join(pdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


@app.get("/api/single-calib/projects")
async def list_single_projects():
    projects = []
    if not os.path.isdir(SINGLE_CALIB_ROOT):
        return projects
    for entry in sorted(os.listdir(SINGLE_CALIB_ROOT)):
        meta_path = os.path.join(SINGLE_CALIB_ROOT, entry, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                projects.append(json.load(f))
    return projects


@app.get("/api/single-calib/projects/{project_id}")
async def get_single_project(project_id: str):
    meta_path = os.path.join(_single_project_dir(project_id), "meta.json")
    if not os.path.isfile(meta_path):
        raise HTTPException(404, "Project not found")
    with open(meta_path) as f:
        return json.load(f)


@app.patch("/api/single-calib/projects/{project_id}")
async def rename_single_project(project_id: str, body: dict):
    pdir = _single_project_dir(project_id)
    meta_path = os.path.join(pdir, "meta.json")
    if not os.path.isfile(meta_path):
        raise HTTPException(404, "Project not found")
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "Name must not be empty")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["name"] = name
    _save_single_meta(project_id, meta)
    return meta


@app.delete("/api/single-calib/projects/{project_id}")
async def delete_single_project(project_id: str):
    pdir = _single_project_dir(project_id)
    if os.path.isdir(pdir):
        shutil.rmtree(pdir)
    return {"ok": True}


@app.post("/api/single-calib/projects/{project_id}/upload")
async def upload_single_images(
    project_id: str,
    files: List[UploadFile] = File(...),
):
    pdir = _single_project_dir(project_id)
    meta_path = os.path.join(pdir, "meta.json")
    if not os.path.isfile(meta_path):
        raise HTTPException(404, "Project not found")
    with open(meta_path) as f:
        meta = json.load(f)

    _IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    _VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".mts", ".m4v", ".wmv")

    saved = []
    for f in files:
        ext = os.path.splitext(f.filename or "img.jpg")[1].lower()
        if ext in _IMAGE_EXTS:
            dest = os.path.join(pdir, "images", f.filename)
            with open(dest, "wb") as out:
                content = await f.read()
                out.write(content)
            if f.filename not in meta["images"]:
                meta["images"].append(f.filename)
            saved.append(f.filename)
        elif ext in _VIDEO_EXTS:
            tmp_path = os.path.join(pdir, f"_tmp_{f.filename}")
            with open(tmp_path, "wb") as out:
                content = await f.read()
                out.write(content)
            try:
                cap = cv2.VideoCapture(tmp_path)
                if not cap.isOpened():
                    continue
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames <= 0:
                    cap.release()
                    continue
                central_idx = total_frames // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, central_idx)
                ret, frame = cap.read()
                cap.release()
                if not ret or frame is None:
                    continue
                out_name = f"{Path(f.filename).stem}_frame{central_idx:06d}.jpg"
                cv2.imwrite(os.path.join(pdir, "images", out_name), frame)
                if out_name not in meta["images"]:
                    meta["images"].append(out_name)
                saved.append(out_name)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    _save_single_meta(project_id, meta)
    return {"saved": saved, "images": meta["images"]}


@app.get("/api/single-calib/projects/{project_id}/image/{filename}")
async def get_single_image(project_id: str, filename: str):
    path = os.path.join(_single_project_dir(project_id), "images", filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "Image not found")
    return FileResponse(path)


@app.delete("/api/single-calib/projects/{project_id}/image/{filename}")
async def delete_single_image(project_id: str, filename: str):
    pdir = _single_project_dir(project_id)
    meta_path = os.path.join(pdir, "meta.json")
    if not os.path.isfile(meta_path):
        raise HTTPException(404, "Project not found")

    img_path = os.path.join(pdir, "images", filename)
    if not os.path.isfile(img_path):
        raise HTTPException(404, "Image not found")
    os.remove(img_path)

    with open(meta_path) as f:
        meta = json.load(f)
    meta["images"] = [fn for fn in meta.get("images", []) if fn != filename]
    meta["calibration"] = None  # invalidate stale calibration
    _save_single_meta(project_id, meta)
    return {"ok": True, "images": meta["images"]}


def _sfm_calibrate(image_paths: list, w: int, h: int) -> dict:
    """
    SfM-based intrinsic calibration powered by COLMAP (via pycolmap).

    Pipeline
    --------
    1. SIFT feature extraction  (OPENCV camera model: fx,fy,cx,cy,k1,k2,p1,p2)
    2. Exhaustive feature matching
    3. Incremental reconstruction with global bundle adjustment
    4. Extract calibrated K + distortion; compute per-image reprojection error
    """
    import pycolmap
    import tempfile
    import shutil
    from pathlib import Path

    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        image_dir = tmpdir / "images"
        image_dir.mkdir()
        db_path = str(tmpdir / "database.db")
        sparse_dir = tmpdir / "sparse"
        sparse_dir.mkdir()

        # Copy images; prefix duplicates to avoid name collisions
        seen: set = set()
        name_map: Dict[str, str] = {}   # colmap name → original path
        for i, p in enumerate(image_paths):
            fname = Path(p).name
            if fname in seen:
                fname = f"{i:04d}_{fname}"
            seen.add(fname)
            shutil.copy2(p, image_dir / fname)
            name_map[fname] = p

        # ── 1. Feature extraction ──────────────────────────────────────────
        reader_opts = pycolmap.ImageReaderOptions()
        reader_opts.camera_model = "OPENCV"   # fx, fy, cx, cy, k1, k2, p1, p2

        extraction_opts = pycolmap.FeatureExtractionOptions()
        extraction_opts.sift.max_num_features = 8192

        pycolmap.extract_features(
            database_path=db_path,
            image_path=str(image_dir),
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader_opts,
            extraction_options=extraction_opts,
        )

        # ── 2. Exhaustive matching ─────────────────────────────────────────
        pycolmap.match_exhaustive(database_path=db_path)

        # ── 3. Incremental reconstruction ─────────────────────────────────
        maps = pycolmap.incremental_mapping(
            database_path=db_path,
            image_path=str(image_dir),
            output_path=str(sparse_dir),
        )

        if not maps:
            raise ValueError(
                "COLMAP reconstruction failed — no models produced. "
                "Ensure images share sufficient scene overlap."
            )

        # Pick largest sub-model
        recon = max(maps.values(), key=lambda r: r.num_reg_images())
        num_reg = recon.num_reg_images()
        num_pts = recon.num_points3D()

        if num_reg < 2:
            raise ValueError(
                f"Only {num_reg} image(s) registered by COLMAP. "
                "Add more images with scene overlap."
            )

        # ── 4. Extract intrinsics ──────────────────────────────────────────
        # OPENCV model params: [fx, fy, cx, cy, k1, k2, p1, p2]
        cam = next(iter(recon.cameras.values()))
        p = list(cam.params)
        fx, fy, cx_c, cy_c = p[0], p[1], p[2], p[3]
        k1, k2, p1, p2    = p[4], p[5], p[6], p[7]

        mtx  = np.array([[fx, 0, cx_c], [0, fy, cy_c], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, k2, p1, p2, 0.0], dtype=np.float64)

        # ── 5. Per-image reprojection error ───────────────────────────────
        per_img: Dict[str, list] = {}
        for img in recon.images.values():
            pose = img.cam_from_world()
            R = pose.rotation.matrix()
            t = np.array(pose.translation)
            rvec, _ = cv2.Rodrigues(R)

            obj_pts, obs_pts = [], []
            for pt2d in img.points2D:
                if not pt2d.has_point3D():
                    continue
                obj_pts.append(recon.points3D[pt2d.point3D_id].xyz)
                obs_pts.append(pt2d.xy)

            if len(obj_pts) < 3:
                continue

            proj, _ = cv2.projectPoints(
                np.array(obj_pts, dtype=np.float64),
                rvec, t, mtx, dist,
            )
            proj = proj.reshape(-1, 2)
            obs  = np.array(obs_pts, dtype=np.float64)
            errs = np.linalg.norm(proj - obs, axis=1)
            per_img[img.name] = errs.tolist()

        # Overall RMS across all images
        all_sq = [e ** 2 for errs in per_img.values() for e in errs]
        rms_final = float(np.sqrt(np.mean(all_sq))) if all_sq else 0.0

        per_image_rms = [
            {"image": name, "rms": round(float(np.sqrt(np.mean([e**2 for e in errs]))), 4)}
            for name, errs in sorted(per_img.items())
        ]

        # Count image pairs that share triangulated 3-D points
        img_pairs: set = set()
        for pt3d in recon.points3D.values():
            ids = [e.image_id for e in pt3d.track.elements]
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    img_pairs.add((min(ids[a], ids[b]), max(ids[a], ids[b])))

        return {
            "rms": round(rms_final, 4),
            "mtx": mtx.tolist(),
            "dist": dist.tolist(),
            "image_size": [w, h],
            "num_images": num_reg,
            "num_points": num_pts,
            "num_pairs": len(img_pairs),
            "initial_focal": round((fx + fy) / 2, 2),
            "per_image_rms": per_image_rms,
        }


@app.post("/api/single-calib/projects/{project_id}/calibrate")
async def run_single_camera_calibration(project_id: str, body: dict):
    """SfM-based intrinsic calibration — no calibration target required."""
    pdir = _single_project_dir(project_id)
    meta_path = os.path.join(pdir, "meta.json")
    if not os.path.isfile(meta_path):
        raise HTTPException(404, "Project not found")
    with open(meta_path) as f:
        meta = json.load(f)

    images = meta.get("images", [])
    if len(images) < 2:
        raise HTTPException(400, f"Need at least 2 images, got {len(images)}")

    img_size = None
    for fname in images:
        img = cv2.imread(os.path.join(pdir, "images", fname))
        if img is not None:
            h_i, w_i = img.shape[:2]
            img_size = (w_i, h_i)
            break
    if img_size is None:
        raise HTTPException(400, "Cannot read any uploaded image")

    image_paths = [os.path.join(pdir, "images", f) for f in images]
    try:
        result = _sfm_calibrate(image_paths, img_size[0], img_size[1])
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"SfM calibration failed: {e}")

    with open(os.path.join(pdir, "calibration.json"), "w") as f:
        json.dump(
            {"name": meta.get("name", "Camera"), "mtx": result["mtx"], "dist": result["dist"]},
            f, indent=2,
        )

    meta["calibration"] = result
    _save_single_meta(project_id, meta)
    return result


@app.get("/api/single-calib/projects/{project_id}/download")
async def download_single_calibration(project_id: str):
    path = os.path.join(_single_project_dir(project_id), "calibration.json")
    if not os.path.isfile(path):
        raise HTTPException(404, "No calibration result yet")
    return FileResponse(path, filename="calibration.json", media_type="application/json")


@app.get("/api/single-calib/projects/{project_id}/undistort/{filename}")
async def get_undistorted_image(project_id: str, filename: str):
    pdir = _single_project_dir(project_id)
    calib_path = os.path.join(pdir, "calibration.json")
    img_path = os.path.join(pdir, "images", filename)
    if not os.path.isfile(calib_path):
        raise HTTPException(404, "No calibration result yet")
    if not os.path.isfile(img_path):
        raise HTTPException(404, "Image not found")
    with open(calib_path) as f:
        cal = json.load(f)
    mtx = np.array(cal["mtx"], dtype=np.float64)
    dist = np.array(cal["dist"], dtype=np.float64)
    img = cv2.imread(img_path)
    if img is None:
        raise HTTPException(400, "Could not read image")
    h, w = img.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha=0)
    undist = cv2.undistort(img, mtx, dist, None, new_mtx)
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        undist = undist[y:y+rh, x:x+rw]
    ok, buf = cv2.imencode(".jpg", undist, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(500, "Failed to encode image")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------
FRONTEND_DIR = os.environ.get("FRONTEND_DIR", "/app/frontend")

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# Mount static files last so API routes take priority
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
