# This is intended to run on a Raspberry Pi.

import cv2.cv2 as cv2
from Image_Processing_approach.Lane_Detection import getLaneCurve
from motor_MD import Motors
import webCam as wc


motors = Motors(2, 3, 4, 17, 22, 27)

while True:

    img = wc.getImg()

    curve = getLaneCurve(img, 2)

    sen = 1.3  # SENSITIVITY
    maxVAl = 0.8

    if curve > maxVAl:
        curve = maxVAl
    if curve < -maxVAl:
        curve = -maxVAl

    if curve > 0:
        sen = 1.7
        if curve < 0.05:
            curve = 0
    else:
        if curve > -0.08:
            curve = 0

    # speed regulation when turning
    cx = 2.5
    if abs(curve) >= 0.5:
        cx = 1
    elif abs(curve) >= 0.3:
        cx = 1.5
    speed = cx * 0.123


    motors.move(speed, -curve * sen, 0.05)

    cv2.waitKey(1)

