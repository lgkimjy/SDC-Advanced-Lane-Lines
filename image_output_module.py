import cv2
import numpy as np

def ManyImgs(scale, imgarray):
    rows = len(imgarray)
    cols = len(imgarray[0]) 
    # print("rows=", rows, "cols=", cols)
    
    rowsAvailable = isinstance(imgarray[0], list)
    height = imgarray[0][0].shape[0]
    width = imgarray[0][0].shape[1]
    # print("width=", width, "height=", height)

    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                # Traverse the tuple, if it is the first image, do not transform
                if imgarray[x][y].shape[:2] == imgarray[0][0].shape[:2]:
                    imgarray[x][y] = cv2.resize(imgarray[x][y], (0, 0), None, scale, scale)
                # Transform other matrices to the same size as the first image, and the zoom ratio is scale
                else:
                    imgarray[x][y] = cv2.resize(imgarray[x][y], (imgarray[0][0].shape[1], imgarray[0][0].shape[0]), None, scale, scale)
                # If the image is grayscale, convert it to color display
                if  len(imgarray[x][y].shape) == 2:
                    imgarray[x][y] = cv2.cvtColor(imgarray[x][y], cv2.COLOR_GRAY2BGR)

        # Create a blank canvas, the same size as the first picture
        imgBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imgBlank] * rows   # The same size as the first picture, and the same number of horizontal blank images as the tuple contains the list
        for x in range(0, rows):
            # Arrange the xth list in the tuple horizontally
            hor[x] = np.hstack(imgarray[x])
        ver = np.vstack(hor)   # Concatenate different lists vertically
    # If the incoming is a list
    else:
        # Transformation operation, same as before
        for x in range(0, rows):
            if imgarray[x].shape[:2] == imgarray[0].shape[:2]:
                imgarray[x] = cv2.resize(imgarray[x], (0, 0), None, scale, scale)
            else:
                imgarray[x] = cv2.resize(imgarray[x], (imgarray[0].shape[1], imgarray[0].shape[0]), None, scale, scale)
            if len(imgarray[x].shape) == 2:
                imgarray[x] = cv2.cvtColor(imgarray[x], cv2.COLOR_GRAY2BGR)
        # Arrange the list horizontally
        hor = np.hstack(imgarray)
        ver = hor
    return ver