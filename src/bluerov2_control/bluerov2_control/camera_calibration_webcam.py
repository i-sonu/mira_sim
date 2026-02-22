import cv2
import numpy as np
import os
import time

# ==============================
# CAMERA DETECTION
# ==============================

def list_available_cameras(max_tests=5):
    available = []
    for i in range(max_tests):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
        cap.release()
    return available

available_cams = list_available_cameras()

if not available_cams:
    print("No cameras found.")
    exit()

print("Available Cameras:")
for cam in available_cams:
    print(f"Camera Index: {cam}")

camera_index = int(input("Enter camera index to use: "))

if camera_index not in available_cams:
    print("Invalid camera index.")
    exit()

# ==============================
# USER SETTINGS
# ==============================

pattern_size = (6, 8)   # INNER corners
square_size = 0.8       # cm
min_good_images = 40

save_folder = "calibration_images"
os.makedirs(save_folder, exist_ok=True)

# ==============================
# PREPARE OBJECT POINTS
# ==============================

objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []

# ==============================
# START SELECTED CAMERA
# ==============================

cap = cv2.VideoCapture(camera_index)

print(f"\nUsing Camera Index: {camera_index}")
print(f"Collecting minimum {min_good_images} good images...")
print("Move board around. Press 'q' to quit.\n")

good_images = 0
last_capture_time = 0

while good_images < min_good_images:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret_corners, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    display_frame = frame.copy()

    if ret_corners:
        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        cv2.drawChessboardCorners(display_frame, pattern_size, corners2, ret_corners)

        if time.time() - last_capture_time > 0.8:
            objpoints.append(objp)
            imgpoints.append(corners2)

            cv2.imwrite(
                os.path.join(save_folder, f"raw_{good_images}.jpg"),
                frame
            )

            cv2.imwrite(
                os.path.join(save_folder, f"annotated_{good_images}.jpg"),
                display_frame
            )

            good_images += 1
            last_capture_time = time.time()
            print(f"Saved good image {good_images}/{min_good_images}")

    cv2.putText(display_frame,
                f"Good Images: {good_images}/{min_good_images}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("Calibration", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ==============================
# CALIBRATION
# ==============================

if good_images >= min_good_images:
    print("\nCalibrating camera...")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        None,
        None
    )

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], mtx, dist
        )
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("\nCamera Matrix:\n", mtx)
    print("\nDistortion Coefficients:\n", dist)
    print("\nMean Reprojection Error: ", mean_error / len(objpoints))

    np.savez("camera_calibration_data.npz",
             camera_matrix=mtx,
             distortion_coeff=dist)

    print("\nCalibration saved as camera_calibration_data.npz")

else:
    print("Not enough good images collected.")
