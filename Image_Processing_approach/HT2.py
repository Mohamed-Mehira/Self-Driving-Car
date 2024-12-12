# The same as the "Hough_Transform" file, but just uses a Functional approach

import cv2
import numpy as np


def get_coordinates(line_parameters):
    slope, intercept = line_parameters
    # try:
    #     slope, intercept = line_parameters
    # except TypeError:
    #     slope, intercept = 0.0001, 0

    y1 = 480
    y2 = int(y1*(1/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept) / slope)

    if abs(y1 - y2) > 300 and abs(x1 - x2) < 150:
        return np.array([x1, y1, x2, y2])


def get_average_lines(lines):
    left_parameters = []
    right_parameters = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)     # Or line[0]
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]

            if x1 < 400 and x2 < 400:
                left_parameters.append((slope, intercept))
            elif x1 > 240 and x2 > 240:
                right_parameters.append((slope, intercept))

            # if x1 < 320 and x2 < 440 and slope < 0:
            #     left_parameters.append((slope, intercept))
            # elif x1 > 320 and x2 > 200 and slope > 0:
            #     right_parameters.append((slope, intercept))

        if len(left_parameters) == 0 or len(right_parameters) == 0:
            return None

        left_parameters_average = np.average(left_parameters, axis=0)
        right_parameters_average = np.average(right_parameters, axis=0)

        left_line = get_coordinates(left_parameters_average)
        right_line = get_coordinates(right_parameters_average)

        if left_line is None or right_line is None:
            return None

        return np.array([left_line, right_line])



def getCurve(img, display=False):

    h, w, c = img.shape

    ### Image Smoothening
    imgBlur = cv2.GaussianBlur(img, (5, 5), 0)


    ### Image Warping
    points = [[125, 111], [515, 111], [0, 325], [640, 325]]
    # points = [[87, 0], [553, 0], [27, 208], [613, 208]]
    # points = [[110, 208], [530, 208], [0, 480], [640, 480]]
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (640, 480))


    ### Color Masking
    imgHSV = cv2.cvtColor(imgWarp, cv2.COLOR_BGR2HSV)
    lowerWhite = np.array([85, 0, 0])
    upperWhite = np.array([179, 160, 255])
    maskedWhite = cv2.inRange(imgHSV, lowerWhite, upperWhite)


    ### Edge Detection
    imgCanny = cv2.Canny(maskedWhite, 70, 110)


    ### Finding Contours
    imgContour = imgWarp.copy()
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)


    ### Hough Lines
    lines = cv2.HoughLinesP(imgCanny, 3, np.pi/180, 60, maxLineGap=80, minLineLength=80)
    averaged_lines = get_average_lines(lines)

    if averaged_lines is not None:
        for [x1, y1, x2, y2] in averaged_lines:
            cv2.line(imgWarp, (x1, y1), (x2, y2), (0, 255, 0), 3)

        imgMask = np.zeros((480, 640))
        shape = np.array([[(averaged_lines[0][0], averaged_lines[0][1]), (averaged_lines[0][2], averaged_lines[0][3]), (averaged_lines[1][2], averaged_lines[1][3]), (averaged_lines[1][0], averaged_lines[1][1])]], np.int32)
        cv2.fillPoly(imgMask, shape, 255)

        cv2.imshow("mask", imgMask)


        midPoint = w/2
        curvePoint = (averaged_lines[0][0] + averaged_lines[1][0]) / 2
        curve = curvePoint - midPoint

        return curve




def main():

    path = 'Resources/vid1.mp4'
    vid = cv2.VideoCapture(path)

    while True:

        ### Capturing the Image
        success, img = vid.read()

        ### Video Looping
        if not success:
            vid = cv2.VideoCapture(path)
            continue

        curve = getCurve(img, True)
        if curve is not None:
            print(curve)

        cv2.imshow('Image', img)
        cv2.waitKey(1)



if __name__ == '__main__':
    main()