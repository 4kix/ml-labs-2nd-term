import cv2
import numpy as np
# from keras.models import load_model
from tensorflow.keras.models import load_model
import io
from PIL import Image

def predict_num(img):
    model = load_model('recognizer_model_svhn_v2_adam.h5')

    X = np.ndarray(shape=(1, 64, 64, 3), dtype='float32')
    X[0, :, :, :] = img
    predictions = model.predict(X)

    number_length = ''
    number = ''
    for i, item in enumerate(predictions):
        if i != 0:
            num_item = np.argmax(item)
            if num_item != 10:
                number = number + str(num_item)
        else:
            number_length = str(np.argmax(item))

    print('num of digits', number_length)
    print('number', number)

    return number_length, number


def resize_img(img):
    width, height = 64, 64
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def recognize(image_file_bytes):
    image = Image.open(io.BytesIO(image_file_bytes))
    img_arr = np.array(image.resize((64, 64)), dtype='float32')
    return predict_num(img_arr)



