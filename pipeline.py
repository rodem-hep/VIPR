import math
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from torchvision import datasets,transforms
import torch as T
from tools import misc

from tensorflow import keras
from keras import layers

# def preprocess_image(data, image_size=64):
#     # center crop image
#     height = tf.shape(data["image"])[0]
#     width = tf.shape(data["image"])[1]
#     crop_size = tf.minimum(height, width)
#     image = tf.image.crop_to_bounding_box(
#         data["image"],
#         (height - crop_size) // 2,
#         (width - crop_size) // 2,
#         crop_size,
#         crop_size,
#     )

#     # resize and clip
#     # for image downsampling it is important to turn on antialiasing
#     image = tf.image.resize(image, size=[image_size, image_size], antialias=True)
#     return tf.clip_by_value(image / 255.0, 0.0, 1.0)


# def prepare_dataset(dataset_name, dataset_repetitions, batch_size, split):
#     # the validation dataset is shuffled as well, because data order matters
#     # for the KID estimation
#     return (
#         tfds.load(dataset_name, split=split, shuffle_files=True)
#         .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
#         .cache()
#         .repeat(dataset_repetitions)
#         .shuffle(10 * batch_size)
#         .batch(batch_size, drop_remainder=True)
#         .prefetch(buffer_size=tf.data.AUTOTUNE)
#     )
class ImagePipeline:
    def __init__(self, dataset_name):
        if dataset_name.lower()=="oxford_flowers102":
            self.data = datasets.Flowers102("test_data/")
        elif dataset_name.lower()=="cifar10":
            self.data = datasets.CIFAR10("test_data/")
            
        self.images = T.tensor(self.data.data)
        self.images_torch = self.images.permute(0, 3, 1, 2)
        self.images_torch = T.clip(self.images_torch / 255.0, 0.0, 1.0)
        self.dataloader_args = {"batch_size":64,"shuffle":True}

    def get_dataloader(self, dataloader_args={}):
        self.dataloader_args.update(dataloader_args)
        return T.utils.data.DataLoader(self.images_torch,
                                       **self.dataloader_args
                                       )

if __name__ == "__main__":
    config = misc.load_yaml("configs/configs.yaml")
    # load dataset
    # train_dataset = prepare_dataset(config.dataset_name,
    #                                 config.dataset_repetitions,
    #                                 config.batch_size,
    #                                 "train[:80%]+validation[:80%]+test[:80%]")
    # val_dataset = prepare_dataset(config.dataset_name,
    #                                 config.dataset_repetitions,
    #                                 config.batch_size,
    #                                 "train[80%:]+validation[80%:]+test[80%:]")
    images = prepare_dataset()
    import matplotlib.pyplot as plt
    for i in range(10):
        plt.figure()
        plt.imshow(images[i].detach().numpy())
        plt.tight_layout()
        plt.show()
        plt.close()