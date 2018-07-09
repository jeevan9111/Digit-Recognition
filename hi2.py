import numpy as np
import keras
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


print(predict('0.png'))
print(predict('H.png'))
print(predict('9.png'))
print(predict('a.png'))
