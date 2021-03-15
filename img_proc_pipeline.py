import numpy as np
import cv2
import time
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import image_output_module as out


def calibrateCamera(img):
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    images = glob.glob('camera_cal/calibration*.jpg')
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    img_size = (img.shape[0], img.shape[1])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    
    return mtx, dist

def undistort(img, mtx, dist):
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

def hls_select(img, thresh=(0, 255)):
    
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # 2) Apply a threshold to the S channel
    S = hls[:,:,2]
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] =1
    # 3) Return a binary image of threshold result
    binary_output = S

    return binary_output

def warper(img, src, dst):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    return warped

# Sharpen image
def sharpen_img(img):
    gb = cv2.GaussianBlur(img, (5,5), 20.0)
    return cv2.addWeighted(img, 2, gb, -1, 0)

# Compute linear image transformation img*s+m
def lin_img(img,s=1.0,m=0.0):
    img2=cv2.multiply(img, np.array([s]))
    return cv2.add(img2, np.array([m]))

# Change image contrast; s>1 - increase
def contr_img(img, s=1.0):
    m=127.0*(1.0-s)
    return lin_img(img, s, m)

# Create perspective image transformation matrices
def create_M():
    src = np.float32([[0, 673], [1207, 673], [0, 450], [1280, 450]])
    dst = np.float32([[569, 330], [711, 330], [0, 0], [1280, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

# Main image transformation routine to get a warped image
def transform(img, M, mtx, dist):
    undist = undistort(img, mtx, dist)
    img_size = (1280, 330)
    warped = cv2.warpPerspective(undist, M, img_size)
    warped = sharpen_img(warped)
    warped = contr_img(warped, 1.1)
    return warped

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    # #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def lane_finding_process(frame, mtx, dist, M):

    vertices = np.array([[(0,frame.shape[0]),(560, 460), (720, 460), (frame.shape[1],frame.shape[0])]], dtype=np.int32)
    frame = region_of_interest(frame, vertices)
    hls_binary = hls_select(frame, thresh=(90, 255))
    warped = transform(hls_binary, M, mtx, dist)

    return hls_binary, warped


def main():
    cap = cv2.VideoCapture('test_videos/project_video.mp4')
    fps= int(cap.get(cv2.CAP_PROP_FPS))
    _, img = cap.read()
    mtx, dist = calibrateCamera(img)
    M, invM = create_M()
    while(cap.isOpened()):
        ret, frame = cap.read()

        ## image processing ##
        hls, warped = lane_finding_process(frame, mtx, dist, M)
        ## image visualize ##
        stackedimage = out.ManyImgs(0.5, ([frame], [hls], [warped]))
        cv2.imshow('output_images', stackedimage)

        if cv2.waitKey(fps-10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()