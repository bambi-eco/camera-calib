# Camera Calibration Project

This project performs camera calibration between RGB and thermal cameras by manually selecting corresponding points and computing the transformation matrix between the two camera views.

## Usage

The calibration process involves two main steps that must be run in sequence:

### Step 1: Manual Point Annotation (`do_annotations.ipynb`)

This script allows you to manually select corresponding points between thermal and RGB images.

1. Navigate to the `src/` directory and open `do_annotations.ipynb` in Jupyter
2. Run the notebook cells to:
   - Load RGB and thermal image pairs from `data/rgb/` and `data/thermal/`
   - Display images in an interactive window
   - Click to select corresponding points in thermal and RGB images
   - Save point pairs to `data/pairs.xml`

**How to use the annotation interface:**
- The script will display either a thermal or RGB image
- Click on a point in the current image to mark it
- The interface will automatically switch between thermal and RGB images
- Continue clicking corresponding points between the two images
- Press `Space` to move to the next image pair
- Press `Esc` to exit the annotation interface

The selected points are automatically saved to `data/pairs.xml` in XML format.

### Step 2: Camera Calibration (`do_calibration.ipynb`)

This script uses the annotated point pairs to compute the transformation between cameras.

1. Open `do_calibration.ipynb` in Jupyter
2. Run the notebook cells to:
   - Load the point pairs from `data/pairs.xml`
   - Extract corresponding points from thermal and RGB images
   - Compute the homography matrix using `cv2.findHomography()`
   - Calculate reprojection error to evaluate calibration quality
   - Optimize camera intrinsic parameters for better alignment

**Output:**
- Homography transformation matrix between RGB and thermal cameras
- Reprojection error metrics
- Visualizations showing point correspondences and alignment quality
- Optimized camera calibration parameters

## Project Structure

- `src/*` - Source code and scripts
- `data/*/` - data (i.e., frames from the RGB and thermal camera)
- `requirements.txt` - Python dependencies

