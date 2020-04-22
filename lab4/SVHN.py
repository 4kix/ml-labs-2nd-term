import h5py
import numpy as np
from PIL import Image
from keras.layers import  Input, Dense, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization
from keras.models import Model
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import np_utils


train_dataset = h5py.File("train/digitStruct.mat", "r")
test_dataset = h5py.File("test/digitStruct.mat", "r")


def get_name(dataset, index):
    names = dataset["digitStruct"]["name"]
    return ''.join([chr(c[0]) for c in dataset[names[index][0]].value])


def bbox_helper(dataset, attr):
    if (len(attr) > 1):
        attr = [dataset[attr.value[j].item()].value[0][0] for j in range(len(attr))]
    else:
        attr = [attr.value[0][0]]
    return attr


def get_bbox(dataset, index):
    item = dataset[dataset["digitStruct"]["bbox"][index].item()]
    return {
        "height": bbox_helper(dataset, item["height"]),
        "label": bbox_helper(dataset, item["label"]),
        "left": bbox_helper(dataset, item["left"]),
        "top": bbox_helper(dataset, item["top"]),
        "width": bbox_helper(dataset, item["width"]),
    }


print(get_bbox(train_dataset, 32300))


def get_whole_box(dataset, index, im):
    bbox = get_bbox(dataset, index)

    im_left = min(bbox["left"])
    im_top = min(bbox["top"])
    im_height = max(bbox["top"]) + max(bbox["height"]) - im_top
    im_width = max(bbox["left"]) + max(bbox["width"]) - im_left

    im_top = im_top - im_height * 0.15
    im_left = im_left - im_width * 0.15
    im_bottom = min(im.size[1], im_top + im_height * 1.3)
    im_right = min(im.size[0], im_left + im_width * 1.3)

    return {
        "label": bbox["label"],
        "left": im_left,
        "top": im_top,
        "right": im_right,
        "bottom": im_bottom
    }


im = Image.open("train/train/" + get_name(train_dataset, 32300))
im.show()
print(get_whole_box(train_dataset, 32300, im))


box = get_whole_box(train_dataset, 32300, im)

im = im.crop((box["left"], box["top"], box["right"], box["bottom"]))
im.resize((64, 64))
im.show()

train_count = train_dataset["digitStruct"]["name"].shape[0]

X_train = np.ndarray(shape=(train_count, 64, 64, 3), dtype='float32')
y = {
    0: np.zeros(train_count),
    1: np.ones(train_count) * 10,
    2: np.ones(train_count) * 10,
    3: np.ones(train_count) * 10,
    4: np.ones(train_count) * 10,
    5: np.ones(train_count) * 10
}

for i in range(train_count):
    im = Image.open("train/train/" + get_name(train_dataset, i))
    box = get_whole_box(train_dataset, i, im)
    if len(box["label"]) > 5:
        continue
    im = im.crop((box["left"], box["top"], box["right"], box["bottom"])).resize((64, 64))

    X_train[i, :, :, :] = np.array(im.resize((64, 64)), dtype='float32')

    labels = box["label"]

    y[0][i] = len(labels)

    for j in range(0, 5):
        if j < len(labels):
            if labels[j] == 10:
                y[j + 1][i] = 0
            else:
                y[j + 1][i] = int(labels[j])
        else:
            y[j + 1][i] = 10

    if i % 500 == 0:
        print(i, len(y[0]))

y_train = [
    np.array(y[0]).reshape(train_count, 1),
    np.array(y[1]).reshape(train_count, 1),
    np.array(y[2]).reshape(train_count, 1),
    np.array(y[3]).reshape(train_count, 1),
    np.array(y[4]).reshape(train_count, 1),
    np.array(y[5]).reshape(train_count, 1)
]

test_count = test_dataset["digitStruct"]["name"].shape[0]

X_test = np.ndarray(shape=(test_count, 64, 64, 3), dtype='float32')
y = {
    0: np.zeros(test_count),
    1: np.ones(test_count) * 10,
    2: np.ones(test_count) * 10,
    3: np.ones(test_count) * 10,
    4: np.ones(test_count) * 10,
    5: np.ones(test_count) * 10
}

for i in range(test_count):
    im = Image.open("train/test/" + get_name(test_dataset, i))
    box = get_whole_box(test_dataset, i, im)
    if len(box["label"]) > 5:
        continue
    im = im.crop((box["left"], box["top"], box["right"], box["bottom"])).resize((64, 64))

    X_test[i, :, :, :] = np.array(im.resize((64, 64)), dtype='float32')

    labels = box["label"]

    y[0][i] = len(labels)

    for j in range(0, 5):
        if j < len(labels):
            if labels[j] == 10:
                y[j + 1][i] = 10
            else:
                y[j + 1][i] = int(labels[j])
        else:
            y[j + 1][i] = 10

    if i % 500 == 0:
        print(i, len(y[0]))

y_test = [
    np.array(y[0]).reshape(test_count, 1),
    np.array(y[1]).reshape(test_count, 1),
    np.array(y[2]).reshape(test_count, 1),
    np.array(y[3]).reshape(test_count, 1),
    np.array(y[4]).reshape(test_count, 1),
    np.array(y[5]).reshape(test_count, 1)
]

input_ = Input(shape=(64, 64, 3))
model = BatchNormalization()(input_)
model = Conv2D(64, (7, 7), activation='relu', padding='same')(model)
model = MaxPool2D(pool_size=(2, 2))(model)
model = BatchNormalization()(model)
model = Conv2D(128, (5, 5), activation='relu', padding='valid')(model)
model = MaxPool2D(pool_size=(2, 2))(model)
model = BatchNormalization()(model)
model = Conv2D(256, (3, 3), activation='relu', padding='valid')(model)
model = MaxPool2D(pool_size=(2, 2))(model)
model = Dropout(0.5)(model)
model = Flatten()(model)
model = Dense(1024, activation='relu')(model)
model = Dense(512, activation='relu')(model)

x1 = Dense(6, activation='softmax')(model)
x2 = Dense(11, activation='softmax')(model)
x3 = Dense(11, activation='softmax')(model)
x4 = Dense(11, activation='softmax')(model)
x5 = Dense(11, activation='softmax')(model)
x6 = Dense(11, activation='softmax')(model)

x = [x1, x2, x3, x4, x5, x6]
model = Model(inputs=input_, outputs=x)

model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])
model.summary()
# Y_train = np_utils.to_categorical(y_train) # One-hot encode the labels
# Y_test = np_utils.to_categorical(y_test) # One-hot encode the labels

fit_data = model.fit(X_train, y_train, batch_size=64, epochs=1, verbose=2)


test_data = model.evaluate(X_test, y_test)

print('Test accuracy: {:.4f}'.format(test_data[1]))
print('Test loss: {:.4f}'.format(test_data[0]))

# model.save('recognizer_model_svhn_v2_adam.h5')
# del model
# model = load_model('recognizer_model_svhn_v2.h5')
# score = model.evaluate(X_test, y_test, verbose=2)
# print(score[7:])

print(fit_data.history.keys())

