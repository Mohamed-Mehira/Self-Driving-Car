import cv2.cv2 as cv2

vid = cv2.VideoCapture(0)

def getImg(display=False, size=[640, 480]):

    success, img = vid.read()
    img = cv2.resize(img, (size[0], size[1]))

    if display:
        cv2.imshow('Image', img)

    return img



if __name__ == '__main__':

    while True:
        img = getImg(True)

        cv2.waitKey(1)

