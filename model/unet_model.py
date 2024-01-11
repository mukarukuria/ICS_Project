import numpy as np
import os
import pydicom
import glob
import re

def load_contour_file(path, image_shape):
    mask = np.zeros(image_shape)

    with open(path, 'r') as file:
        for line in file:
            x, y = map(int, map(float, line.split()))
            mask[y, x] = 1

    return mask

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image

def load_dicom_image(path):
    dicom = pydicom.read_file(path)
    image = dicom.pixel_array
    normalized_image = normalize_image(image)
    return image

dicom_dir = 'model\data\image'
contour_dir = 'model\data\mask'

image_filenames = glob.glob(os.path.join(dicom_dir, '*.dcm'))
mask_filenames = glob.glob(os.path.join(contour_dir, '*.txt'))

mapping = {}

for image_filename in image_filenames:
    image_id = re.findall(r'\d{4}', image_filename)[-1]

    for mask_filename in mask_filenames:
        mask_id = re.findall(r'\d{4}', mask_filename)[-1]

        if image_id == mask_id:
            mapping[image_filename] = mask_filename
            break

for image_filename, mask_filename in mapping.items():
    dicom_images = [load_dicom_image(file) for file in image_filenames]
    contour_masks = [load_contour_file(file, dicom_images[i].shape) for i, file in enumerate(mask_filenames)]

from sklearn.model_selection import train_test_split
X_train, X_test_val, y_train, y_test_val = train_test_split(dicom_images, contour_masks, test_size=0.25, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.4, random_state=42)

X_train = np.array(X_train)
y_train = np.array(y_train)

if len(X_train.shape) == 3:
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)

if len(y_train.shape) == 3:
    y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], y_train.shape[2], 1)

dicom_images = np.array(dicom_images)
dicom_images = dicom_images.reshape(dicom_images.shape[0], dicom_images.shape[1], dicom_images.shape[2], 1)


from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images horizontally
    vertical_flip=False)  # do not randomly flip images vertically

datagen.fit(dicom_images)


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

def create_unet_model(input_shape):
    inputs = Input(input_shape)

    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)

    up4 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=-1)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(up4)

    up5 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=-1)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up5)

    output = Conv2D(1, (1, 1), activation='sigmoid')(conv5)

    model = Model(inputs=[inputs], outputs=[output])

    return model

unet_model = create_unet_model((256, 256, 1))
unet_model.compile(optimizer='adam', loss='binary_crossentropy')

unet_model.fit(datagen.flow(X_train, y_train, batch_size=8), epochs=50)

loss = unet_model.evaluate(X_test, Y_test)

print(f'Test loss: {loss}')

from tensorflow.keras import backend as K

def iou(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')

    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection

    iou = intersection / union

    return iou

def dice_coef(y_true, y_pred, smooth=1):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')

    intersection = K.sum(y_true * y_pred)

    dice = (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

    return dice

y_pred = unet_model.predict(X_test)

iou_score = iou(y_test, y_pred)
dice_score = dice_coef(y_test, y_pred)

print(f'IoU: {iou_score}')
print(f'Dice Coefficient: {dice_score}')

unet_model.summary()

import matplotlib.pyplot as plt

idx = np.random.randint(len(X_test))
image = X_test[idx]
true_mask = Y_test[idx]

predicted_mask = unet_model.predict(image[np.newaxis, ...])[0]

unet_model.save('unet_model.h5')