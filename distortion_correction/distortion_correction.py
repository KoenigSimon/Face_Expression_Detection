import numpy as np
import cv2 as cv
import glob

# Define the dimensions of checkerboard
CHECKERBOARD = (5, 8)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.jpg')
print(f"found {len(images)} images")
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if ret == True:
        print(f"Picture {fname} success")
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img,CHECKERBOARD,corners2, ret)
        #cv.imshow('img', img)
        cv.imwrite("./edit_" + fname, img)
        #cv.waitKey(500)
#cv.destroyAllWindows()

#if everything good, continue to extract calibration parameters
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#print to console
print("ret: ", ret)
print("mtx: ", mtx)
print("dist: ", dist)
#print("rvecs: ", rvecs)
#print("tvecs: ", tvecs)

#ret:  2.5235711901583735
#mtx:  [[535.68751981   0.         325.61866707]
#       [  0.         532.16539606 234.24546875]
#       [  0.           0.           1.        ]]
# dist:  [[-0.88958658  1.02578331  0.01243967  0.00313227 -0.50098573]]