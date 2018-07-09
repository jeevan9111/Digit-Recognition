import sys
import cv2
import numpy as np

mser = cv2.MSER_create()
mser.setMinArea(600)


def draw_contours(img, contours):
    cv2.polylines(img, contours, 1, (0, 255, 0))
    for contour in contours:
        cv2.drawContours(img, [contour], -1, (255, 255, 255), -1)
    return img


def find_texts(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    thres = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 11, 2)
    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

    regions, bboxes = mser.detectRegions(thres, None)
    print (bboxes)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    mask = draw_contours(mask, hulls)
    cv2.imshow('hulls', mask)
    extract = cv2.bitwise_and(thres, thres, mask=mask)
    regions = np.asarray([cv2.boundingRect(part) for part in regions])
    _, _, maxWidth, maxHeight = np.amax(regions, 0)
    _, _, minWidth, minHeight = np.amin(regions, 0)
    testWidth = (maxWidth + minWidth) // 3
    string = ""
    xx, yy, ww, hh = regions[0]

    regions.view('uint8,uint8,uint8,uint8').sort(order=['f1'], axis=0)

    print (regions)
    for part in regions:
        x, y, w, h = part

        cv2.rectangle(extract, (x, y), (x + w, y + h), (250, 250, 250), 1)
        cv2.rectangle(extract, (x, y + 100), (x + w, y + h + 100), (250, 250, 250), 1)
        text = extract[y:y + h, x:x + w]

        # if abs(x - xx) < minWidth:
        #   continue

        if abs(x - xx - w) > testWidth:
            string = string + " "

        string = string + 'H'

        xx, yy, ww, hh = part

    print (string)
    result = extract
    cv2.imshow('result', result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread("c.png", cv2.IMREAD_GRAYSCALE)
find_texts(img)
