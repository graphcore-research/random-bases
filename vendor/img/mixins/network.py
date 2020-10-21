# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

from machinable import Mixin

import tensorflow as tf

tf.compat.v1.disable_v2_behavior()


class NetworkMixin(Mixin):
    def load(self, name=None, classes=None, inputs=None, input_shape=None):
        if name is None:
            name = self.config.network

        if classes is None:
            try:
                classes = self.dataset_info.features["label"].num_classes
                if input_shape is None:
                    input_shape = self.dataset_info.features["image"].shape
            except (AttributeError, KeyError):
                classes = 10

        input_layer = []
        if inputs is not None:
            input_shape = (
                tuple(inputs.get_shape().as_list())[1:]
                if input_shape is None
                else input_shape
            )
            batch_size = tuple(inputs.get_shape().as_list())[0]
            input_layer = [
                tf.keras.layers.InputLayer(
                    input_shape=input_shape, batch_size=batch_size, input_tensor=inputs
                )
            ]

        if name == "conv":
            return tf.keras.Sequential(
                input_layer
                + [
                    tf.keras.layers.Conv2D(
                        filters=32,
                        kernel_size=[5, 5],
                        padding="same",
                        activation=tf.nn.relu,
                    ),
                    tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2),
                    tf.keras.layers.Conv2D(
                        filters=64,
                        kernel_size=[5, 5],
                        padding="same",
                        activation=tf.nn.relu,
                    ),
                    tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                    tf.keras.layers.Dense(classes, activation="softmax"),
                ]
            )

        if name == "base_conv":
            return tf.keras.Sequential(
                input_layer
                + [
                    tf.keras.layers.Conv2D(
                        filters=32, kernel_size=[3, 3], activation=tf.nn.relu
                    ),
                    tf.keras.layers.MaxPool2D((2, 2)),
                    tf.keras.layers.Conv2D(
                        filters=64, kernel_size=[3, 3], activation=tf.nn.relu
                    ),
                    tf.keras.layers.MaxPool2D((2, 2)),
                    tf.keras.layers.Conv2D(
                        filters=64, kernel_size=[3, 3], activation=tf.nn.relu
                    ),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(64, activation=tf.nn.relu),
                    tf.keras.layers.Dense(classes, activation="softmax"),
                ]
            )

        if name == "dense":
            return tf.keras.Sequential(
                input_layer
                + [
                    tf.keras.layers.Flatten(input_shape=input_shape),
                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                    tf.keras.layers.Dense(classes, activation="softmax"),
                ]
            )

        raise ValueError(f"Network {name} not found")
