import numpy as np
from data_ops import augment_data, expand_to_3_channels


# Generator with option to augment data
def gen(data, labels, augment=False):
  c = 0
  while True:
    img = data[c:c+1]
    lbl = labels[c:c+1]
    if augment: img, lbl = augment_data(img, lbl)
    if c == data.shape[0] - 1: 
      c=0
      randomize = np.arange(len(data))
      np.random.shuffle(randomize)
      data = data[randomize]
      labels = labels[randomize]
    c+=1
    yield expand_to_3_channels(img), lbl

# Converts a generator into a batching generator
def batch_gen(gen, batch_size):
  def create_batch():
    images, labels = [], []
    for i in range(batch_size):
      img, lbl = next(gen)
      images.append(img[0])
      labels.append(lbl[0])
    images, labels = np.array(images), np.array(labels)
    return images, labels

  while True:
    yield create_batch()
