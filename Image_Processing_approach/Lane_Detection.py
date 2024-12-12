# This program uses simple image processing techniques to estimate a relative curve point indicating how much steering is needed, using histograms.

import cv2
import numpy as np
import utils

####################
frameWidth = 640
frameHeight = 480
####################


def getLaneCurve(img, display=2, warp=True):

    h, w, c = img.shape

    curveList = []
    avgVal = 10

    imgWarpPoints = img.copy()
    imgResult = img.copy()


    ####### 1- Getting the Masked Image #######
    mask_img = utils.thresholding(img)


    ####### 2- Getting the Warped Image #######
    '''
    points = utils.getPoints()
    print(points)
    '''
    points = [[110, 208], [530, 208], [0, 480], [640, 480]]
    utils.drawPoints(imgWarpPoints, points)
    warp_img = utils.warpImg(mask_img, points)


    ####### 3- Getting the Curve #######
    if warp:
        curvePoint, imgHistogram = utils.getHistogram(warp_img, 0.9, 1, True)
        centerPoint, imgHist2 = utils.getHistogram(warp_img, 0.5, 0.25, True)
    else:
        curvePoint, imgHistogram = utils.getHistogram(mask_img, 0.6, 1, True)
        centerPoint, imgHist2 = utils.getHistogram(mask_img, 0.5, 0.25, True)
    curveTotal = curvePoint - centerPoint

    curveList.append(curveTotal)
    if len(curveList) > avgVal:
        curveList.pop(0)
    curve = int(sum(curveList) / len(curveList))


    ####### 4- Displaying the Results (OPTIONAL) #######
    if display == 0:
        curve = curve / 100
        # if curve > 1:
        #     curve = 1
        # elif curve < -1:
        #     curve = -1
        return curve

    # imgCopy = img.copy()
    # imgCopy[0:h // 2, 0:w] = 0, 0, 0
    # imgMask = utils.thresholding(imgCopy)
    # or:
    # imgMask = utils.thresholding(img)
    # imgMask[0:h // 2, 0:w] = 0
    mask_img[0:h // 2, 0:w] = 0

    imgLaneColor = np.zeros_like(img)
    imgLaneColor[:] = 0, 235, 0
    imgLaneColor = cv2.bitwise_and(imgLaneColor, imgLaneColor, mask=mask_img)

    imgResult = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)

    midPoint = 450
    cv2.putText(imgResult, str(curve), (w // 2 - 80, 85), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
    cv2.line(imgResult, (w // 2, midPoint), (w // 2 + (curve * 2), midPoint), (255, 0, 255), 5)
    cv2.line(imgResult, ((w // 2 + (curve * 2)), midPoint - 25), (w // 2 + (curve * 2), midPoint + 25), (0, 255, 0), 5)

    w = 0
    for x in range(0, 20):
        w += frameWidth // 20
        cv2.line(imgResult, (w, midPoint - 10),
                 (w, midPoint + 10), (0, 0, 255), 2)

    if display == 1:
        cv2.imshow('Resutlt', imgResult)

    elif display == 2:
        imgBlack = np.zeros_like(img)
        imgStacked = utils.stackImages(0.5, ([img, imgWarpPoints, warp_img, mask_img],       # a function from utils.py very useful for displaying multiple images/videos
                                             [imgHistogram, imgLaneColor, imgResult, imgBlack]))
        cv2.imshow('ImageStack', imgStacked)


    ####### 5- Normalization #######
    curve = curve / 100
    # if curve > 1:
    #     curve = 1
    # elif curve < -1:
    #     curve = -1

    return curve





if __name__ == '__main__':

    #### Finding the Warping Points ####
    '''''
    initialTrackBarVals = [110, 208, 0, 480]
    #initialTrackBarVals = [135, 215, 30, 470]
    utils.createTrackBars(initialTrackBarVals)
    '''''

    path = 'Resources/vid1.mp4'
    vid = cv2.VideoCapture(path)

    frameCounter = 0

    while True:

        #### Video Looping ####
        frameCounter += 1
        if vid.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0

        success, img = vid.read()
        img = cv2.resize(img, (frameWidth, frameHeight))

        curve = getLaneCurve(img, warp=False)

        cv2.waitKey(1)