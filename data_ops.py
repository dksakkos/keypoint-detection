import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.util.noise import random_noise
from scipy.signal import convolve2d

# Converts label into 2D binary images with positive values at the keypoints coordinates
def process_labels(labels):
    img_labels = []
    for i in range(labels.shape[0]):
        l = labels[i]
        x_points = l[:: 2]
        y_points = l[1::2]
        new_label = np.zeros((96, 96))
        for x, y in zip(x_points, y_points):
            new_label[int(x), int(y)] = 1
        new_label = new_label.T
        img_labels.append(new_label)
    return np.expand_dims(np.array(img_labels), -1)

# Read data from file and preprocess
def get_data():
    df_train = pd.read_csv('training.csv')

    feature_col = 'Image'
    target_cols = list(df_train.drop('Image', axis=1).columns)

    # Fill missing values
    df_train[target_cols] = df_train[target_cols].fillna(df_train[target_cols].mean())

    # Image characteristics
    IMG_WIDTH = 96
    IMG_HEIGHT = 96
    IMG_CHANNELS = 1

    raw_images = np.array(df_train[feature_col].str.split().tolist(), dtype='float')
    images = raw_images.reshape(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    labels = df_train[target_cols].values
    return images, process_labels(labels)


def show_examples(images, landmarks):
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 16))

    for img, marks, ax in zip(images, landmarks, axes.ravel()):
        # Keypoints
        x_points = marks[:: 2]
        y_points = marks[1::2]

        ax.imshow(img.squeeze(), cmap='gray')
        ax.scatter(x_points, y_points, s=10, color='red')

    plt.show()


def multi_convolver(image, kernel, iterations):
    assert image.shape == (1, 96, 96, 1)
    image = np.squeeze(image)
    for i in range(iterations):
        image = convolve2d(image, kernel, 'same', boundary='fill',
                           fillvalue=0)
    return np.expand_dims(np.expand_dims(image, 0), -1)


def blur(img):
    kernel = (1 / 16.0) * np.array([[1., 2., 1.],
                                    [2., 4., 2.],
                                    [1., 2., 1.]])
    return multi_convolver(img, kernel, np.random.randint(1, 4))


def sharpen(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return multi_convolver(img, kernel, np.random.randint(1, 4))


def augment_data(img, label):
    # Random flips with 50% probability
    if np.random.random() > 0.5:
        img = np.fliplr(img)
        label = np.fliplr(label)
    # Random noise addition with 80% probability
    if np.random.random() > 0.2:
        img = random_noise(img)
    # Random blurring OR sharpening with 50% probability
    if np.random.random() > 0.5:
        if np.random.random() > 0.5:
            img = blur(img)
        else:
            img = sharpen(img)
    return img, label

# Covnerts a greyscale into a fake RGB
def expand_to_3_channels(img):
  return np.concatenate((img, img, img),-1)

# Splits data into train, validation and testing sets
def split_data(images, labels, val_size=500, test_size=500):
    x_train, y_train = images[:-val_size-test_size], labels[:-val_size-test_size]
    x_val, y_val = images[-val_size-test_size:-test_size], labels[-val_size-test_size:-test_size]
    x_test, y_test = images[-test_size:], labels[-test_size]
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
