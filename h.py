import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

model = load_model('first_try.h5')


def get_label(index):
    if index < 10:
        index = index + 48
    elif index < 36:
        index = index - 10 + 65
    else:
        index = index - 36 + 97
    return chr(index)


def predict(im):
    img = image.load_img(im, target_size=(50, 50))
    img = np.asarray(img) / 255
    img = img.reshape((1, 50, 50, 3))
    out = model.predict(img)
    prediction = get_label((np.argmax(out, axis=1))[0])
    return prediction


mser = cv2.MSER_create()
mser.setMinArea(200)


def draw_contours(img, contours):
    cv2.polylines(img, contours, 1, (0, 255, 0))
    for contour in contours:
        cv2.drawContours(img, [contour], -1, (255, 255, 255), -1)
    return img


def find_texts(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    thres = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 11, 2)
    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    regions = mser.detectRegions(thres, None)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    mask = draw_contours(mask, hulls)
    extract = cv2.bitwise_and(thres, thres, mask=mask)

    regions = np.asarray([cv2.boundingRect(part) for part in regions])
    print(regions)
    _, _, maxWidth, maxHeight = np.amax(regions, 0)
    _, _, minWidth, minHeight = np.amin(regions, 0)
    testWidth = (maxWidth + minWidth) // 2

    string = " "

    # regions.view('uint8,uint8,uint8,uint8').sort(order=['f1'], axis=0)
    xx, yy, ww, hh = regions[0]
    i = 0
    for part in regions:
        x, y, w, h = part

       # if w < 20:
          #  continue
        if abs(x - xx) < minWidth:
            continue
        text = extract[y:y + h, x:x + w]
        text = text
        letter = np.zeros((h, h), dtype=np.uint8)

        if h > w:
            gap = (h - w) // 2
            letter[0: h, gap: gap + w] = text
        else:
            letter = np.zeros((w, w), dtype=np.uint8)
            gap = (w - h) // 2
            letter[gap: gap + h, 0: w] = text

        letter = cv2.resize(letter, (40, 40), interpolation=cv2.INTER_CUBIC)

        letter1 = np.zeros((50, 50), dtype=np.uint8)
        letter1[5:45, 5:45] = letter

        letter = cv2.bitwise_not(letter1)

        cv2.imwrite('letter.png', letter)
        if abs(x - xx - w) > testWidth:
            string = string + ' '
        if abs(y - yy) > 1.5 * maxHeight:
            string = string + "\n"

        hisila = predict('letter.png')
        string = string + hisila

        xx, yy, ww, hh = part

        cv2.rectangle(extract, (x, y), (x + w, y + h), (255, 255, 255), 1)
       #cv2.rectangle(extract, (x, y + 100), (x + w, y + h + 100), (255, 255, 255), 1)
        i += 1

    print(string)
    result = extract
    cv2.imshow('result', result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread("a.png", cv2.IMREAD_GRAYSCALE)
find_texts(img)
