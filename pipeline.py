# import math
import matplotlib.pyplot as plt
# import tensorflow as tf
# import tensorflow_datasets as tfds
from torchvision import datasets,transforms
import torch as T
from tools import misc

# from tensorflow import keras
# from keras import layers

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
        data_path = "/home/users/a/algren/work/diffusion/test_data/"
        if dataset_name.lower()=="oxford_flowers102":
            self.data = datasets.Flowers102(data_path)
        elif dataset_name.lower()=="cifar10":
            self.data = datasets.CIFAR10(data_path)
        elif "fashion" in dataset_name.lower():
            self.data = datasets.FashionMNIST(data_path,
                                              transform=transforms.Compose([transforms.Resize(32),
                                                                            transforms.ToTensor()]
    ))

        self.dataloader_args = {"batch_size":64,"shuffle":True}

    def get_dataloader(self, dataloader_args={}):
        self.dataloader_args.update(dataloader_args)
        return T.utils.data.DataLoader(self.data,
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
    images = ImagePipeline()
    import matplotlib.pyplot as plt
    for i in range(10):
        plt.figure()
        plt.imshow(images[i].detach().numpy())
        plt.tight_layout()
        plt.show()
        plt.close()