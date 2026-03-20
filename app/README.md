# Camera Calibration Tool — Thermal / RGB Alignment

A browser-based application for calibrating and spatially aligning thermal and RGB cameras mounted on drones. Supports both **image pairs** and **video pairs** (with automatic central-frame extraction for video-mode calibrations where the cameras don't record open-gate).

## Features

- **Project management** — create multiple calibration sessions (e.g. one per recording mode)
- **Image pair upload** — drag-and-drop RGB + thermal images
- **Video pair upload** — upload short static-view video clips; the app extracts the central frame automatically
- **Interactive keypoint annotation** — side-by-side canvas with zoom/pan; click alternately on RGB then thermal to mark corresponding points
- **Camera calibration** — homography computation (RANSAC) + Nelder-Mead optimization of thermal intrinsics against the RGB reference
- **Overlay preview** — Canny-edge overlay of undistorted RGB onto thermal to visually verify alignment
- **Export** — download optimized `calibration.json` and `pairs.xml` (CVAT-compatible format)

## Quick Start

```bash
docker compose up --build
```

Then open **http://localhost:8000** in your browser.

## Workflow

### 1. Create a Project
Click **+ New Project** in the sidebar. Give it a descriptive name (e.g. "Video Mode 4K" or "Photo Mode").

### 2. Upload Data

**Image Pairs mode:** Upload RGB images on the left, thermal images on the right. Files are matched by sorted filename order — ensure matching names or numbering (e.g. `frame_000022.jpg` in both folders).

**Video Pairs mode:** Select one RGB video and one thermal video. Click "Extract & Upload Pair". The backend extracts the central frame from each video. Repeat for multiple calibration scenes.

### 3. Load Initial Calibration
In the **② Initial Calib** tab, upload your `initial_calibration.json` containing factory/nominal intrinsics for both cameras. Format:

```json
{
  "Thermal": {
    "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "dist": [[k1, k2, p1, p2, k3]]
  },
  "RGB": {
    "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "dist": [[k1, k2, p1, p2, k3]]
  }
}
```

You can also edit the values directly in the browser.

### 4. Annotate Keypoints
In the **③ Annotate** tab:
- The active panel is highlighted — click to place a point
- First click goes on **RGB** (blue), second on the **thermal** (orange) at the corresponding location
- Use **scroll wheel** to zoom, **shift+drag** or **middle-click+drag** to pan
- Navigate between pairs with the ◀ ▶ buttons
- Click **Save** to persist annotations
- Aim for **≥ 8 well-distributed point correspondences** across all pairs

### 5. Run Calibration
In the **④ Results** tab, click **Run Calibration**. The backend:
1. Computes a RANSAC homography between all point correspondences
2. Runs Nelder-Mead optimization on the thermal camera intrinsics (fx, fy, cx, cy, k1–k3, p1, p2) to minimize reprojection error against the RGB reference
3. Generates overlay images (Canny edges of undistorted RGB composited onto undistorted thermal)

Download the resulting `calibration.json` for use in your processing pipeline.

## Project Structure

```
camera-calib-app/
├── docker-compose.yml
├── Dockerfile
├── backend/
│   ├── main.py              # FastAPI application
│   ├── intrinsics.py         # Camera intrinsics classes
│   └── requirements.txt
├── frontend/
│   └── index.html            # Single-file React application
└── data/                     # Persisted via Docker volume
    └── <project-id>/
        ├── meta.json          # Project metadata + annotations
        ├── pairs.xml          # CVAT-compatible annotation export
        ├── rgb/               # RGB images
        ├── thermal/           # Thermal images
        └── results/
            ├── calibration.json
            └── overlay_*.jpg
```

## Development

For live-reload during development, uncomment the volume mounts in `docker-compose.yml`:

```yaml
volumes:
  - calib-data:/app/data
  - ./backend:/app/backend
  - ./frontend:/app/frontend
```

The uvicorn server runs with `--reload` by default.

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/projects` | Create project |
| `GET` | `/api/projects` | List projects |
| `GET` | `/api/projects/{id}` | Get project details |
| `DELETE` | `/api/projects/{id}` | Delete project |
| `POST` | `/api/projects/{id}/upload-images` | Upload image files |
| `POST` | `/api/projects/{id}/upload-video` | Upload video, extract central frame |
| `POST` | `/api/projects/{id}/upload-video-pair` | Upload RGB+thermal video pair |
| `GET` | `/api/projects/{id}/pairs` | Get matched image pairs |
| `GET` | `/api/projects/{id}/image/{channel}/{file}` | Serve image |
| `GET/POST` | `/api/projects/{id}/annotations` | Get/save annotations |
| `GET/POST` | `/api/projects/{id}/initial-calibration` | Get/set initial calibration |
| `POST` | `/api/projects/{id}/calibrate` | Run calibration optimization |
| `GET` | `/api/projects/{id}/download-calibration` | Download calibration.json |
| `GET` | `/api/projects/{id}/download-annotations` | Download pairs.xml |
