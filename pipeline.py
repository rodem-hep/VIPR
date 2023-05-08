import math
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tools import misc

from tensorflow import keras
from keras import layers

def preprocess_image(data, image_size=64):
    # center crop image
    height = tf.shape(data["image"])[0]
    width = tf.shape(data["image"])[1]
    crop_size = tf.minimum(height, width)
    image = tf.image.crop_to_bounding_box(
        data["image"],
        (height - crop_size) // 2,
        (width - crop_size) // 2,
        crop_size,
        crop_size,
    )

    # resize and clip
    # for image downsampling it is important to turn on antialiasing
    image = tf.image.resize(image, size=[image_size, image_size], antialias=True)
    return tf.clip_by_value(image / 255.0, 0.0, 1.0)


def prepare_dataset(dataset_name, dataset_repetitions, batch_size, split):
    # the validation dataset is shuffled as well, because data order matters
    # for the KID estimation
    return (
        tfds.load(dataset_name, split=split, shuffle_files=True)
        .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .repeat(dataset_repetitions)
        .shuffle(10 * batch_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

if __name__ == "__main__":
    config = misc.load_yaml("configs/configs.yaml")
    # load dataset
    train_dataset = prepare_dataset(config.dataset_name,
                                    config.dataset_repetitions,
                                    config.batch_size,
                                    "train[:80%]+validation[:80%]+test[:80%]")
    val_dataset = prepare_dataset(config.dataset_name,
                                    config.dataset_repetitions,
                                    config.batch_size,
                                    "train[80%:]+validation[80%:]+test[80%:]")