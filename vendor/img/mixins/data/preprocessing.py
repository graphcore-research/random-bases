# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import tensorflow as tf

tf.compat.v1.disable_v2_behavior()


class BaseProcessor:
    def __init__(self, split=None, one_hot=False, as_tuple=False, **kwargs):
        self.split = split
        self.is_training = split == "train"
        self.one_hot = one_hot
        self.as_tuple = as_tuple
        self.options = kwargs

    def __call__(self, sample, label=None):
        if label is not None:
            sample = {"image": sample, "label": label}
        sample = self.run(sample)
        if int(self.one_hot) > 0:
            sample["label"] = tf.one_hot(sample["label"], depth=int(self.one_hot))

        if self.as_tuple:
            return sample["image"], sample["label"]

        return sample

    def run(self, sample):
        raise NotImplementedError


class MnistPreprocessor(BaseProcessor):
    def run(self, sample):
        sample["image"] = tf.cast(sample["image"], tf.float32) / 255.0
        sample["label"] = tf.cast(sample["label"], tf.int32)

        if self.split == "train" and self.options.get("augmentation", False):
            sample["image"] = tf.image.convert_image_dtype(
                sample["image"], tf.float32
            )  # Cast and normalize the image to [0,1]
            sample["image"] = tf.image.resize_with_crop_or_pad(
                sample["image"], 34, 34
            )  # Add 6 pixels of padding
            sample["image"] = tf.image.random_crop(
                sample["image"], size=[28, 28, 1]
            )  # Random crop back to 28x28
            sample["image"] = tf.image.random_brightness(
                sample["image"], max_delta=0.5
            )  # Random brightness

        return sample


class CifarPreprocessor(BaseProcessor):
    def run(self, sample):
        sample["image"] = tf.cast(sample["image"], tf.float32) / 255.0
        sample["label"] = tf.cast(sample["label"], tf.int32)

        if self.split == "train" and self.options.get("augmentation", False):
            # Pad 4 pixels on each dimension of feature map, done in mini-batch
            sample["image"] = tf.image.resize_with_crop_or_pad(sample["image"], 40, 40)
            sample["image"] = tf.image.random_crop(sample["image"], [32, 32, 3])
            sample["image"] = tf.image.random_flip_left_right(sample["image"])

        # standard mean and variance
        sample["image"] = tf.image.per_image_standardization(sample["image"])

        return sample
