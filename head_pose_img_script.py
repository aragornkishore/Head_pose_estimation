import cv2
import numpy as np

def sort_image_points(image_points):
    sorted_image_points = image_points[image_points[:,1].argsort()]
    if sorted_image_points[0,0] > sorted_image_points[1,0]:
        x_l,y_l = sorted_image_points[1,:]
        x_r,y_r = sorted_image_points[0,:]
        sorted_image_points[0,:] = x_l,y_l
        sorted_image_points[1,:] = x_r,y_r
    
    if sorted_image_points[3,0] > sorted_image_points[4,0]:
        x_l,y_l = sorted_image_points[4,:]
        x_r,y_r = sorted_image_points[3,:]
        sorted_image_points[3,:] = x_l,y_l
        sorted_image_points[4,:] = x_r,y_r
    
    return sorted_image_points

def detect_blobs(frame):
    size = frame.shape

    lower = np.array([0, 0, 100])
    upper = np.array([0, 0, 255])

    mask = cv2.inRange(frame, lower, upper)
    frame_thresh = cv2.bitwise_and(frame, frame, mask = mask)
    frame_gs = cv2.cvtColor(frame_thresh, cv2.COLOR_BGR2GRAY)
    frame_gs_inv = 255 - frame_gs

    params = cv2.SimpleBlobDetector_Params()

    # Filter by Area.
    params.filterByArea = True
    params.minArea =1

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.05
 
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.1
 
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
 
    # Detect blobs.
    keypoints = detector.detect(frame_gs_inv)
    image_points = np.float32([keypoints[0].pt, keypoints[1].pt, keypoints[2].pt, keypoints[3].pt, keypoints[4].pt, keypoints[5].pt])
    return image_points
 
def head_pose(model_points, sorted_image_points, camera_matrix):
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, sorted_image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_mat, translation_vector))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
    return euler_angle

