# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import math

import numpy as np
from scipy.stats.stats import pearsonr

import tensorflow as tf
import tensorflow_datasets as tfds

from .rbd import RbdModel


class Grad(RbdModel):
    """
    Use gradient vectors as bases

    This is a highly inefficient and slow CPU implementation of
    basic ops that is only intended to support debugging and in-depth analysis.
    """

    def on_create(self):
        self.bases = {}
        self.weights = {}
        self.sess = sess = tf.compat.v1.Session()

        self._apply_layer_ops()

        # data

        self._image_data_.load()
        data_shapes = tf.compat.v1.data.get_output_shapes(self.data["train"])
        data_types = tf.compat.v1.data.get_output_types(self.data["train"])
        image_ = tf.compat.v1.placeholder(
            dtype=data_types["image"],
            shape=[None] + list(data_shapes["image"][1:]),
        )
        label_ = tf.compat.v1.placeholder(
            dtype=data_types["label"],
            shape=[None] + list(data_shapes["label"][1:]),
        )
        t_ = tf.compat.v1.placeholder(dtype=tf.int32, shape=[])

        steps_per_epoch_train = math.ceil(
            self.dataset_info.splits["train"].num_examples / self.config.data.batch_size
        )
        steps_per_epoch_val = math.ceil(
            self.dataset_info.splits["validation"].num_examples
            / self.config.data.batch_size
        )

        # network

        self.coordinates = tf.keras.backend.zeros(
            shape=[self.config.base_dimensions],
            dtype=tf.float32,
            name="coordinates",
        )
        self.network = self._image_network_.load(
            name=self.config.network,
            classes=self.dataset_info.features["label"].num_classes,
            input_shape=self.dataset_info.features["image"].shape,
        )
        predictions = self.network(image_)
        loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(label_, predictions)
        )
        acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(label_, predictions))

        grad = {k: tf.gradients(loss, weight)[0] for k, weight in self.weights.items()}
        norm = tf.math.add_n([tf.reduce_sum(tf.square(g)) for k, g in grad.items()])
        normed_grad = {k: v / norm for k, v in grad.items()}

        # ops

        reset_coordinates_op = tf.compat.v1.assign(
            self.coordinates, tf.zeros_like(self.coordinates)
        )

        apply_base_axis_ = tf.compat.v1.placeholder(dtype=tf.int32, shape=[])
        apply_base_ = {
            k: tf.compat.v1.placeholder(dtype=base.dtype, shape=base.shape[-1])
            for k, base in self.bases.items()
        }
        apply_base_op = tf.group(
            [
                tf.compat.v1.scatter_update(
                    base,
                    apply_base_axis_,
                    apply_base_[k],
                )
                for k, base in self.bases.items()
            ]
        )

        compute_coordinates_op = tf.gradients(loss, self.coordinates)[0]

        coordinates_ = tf.compat.v1.placeholder(
            dtype=self.coordinates.dtype, shape=self.coordinates.shape
        )

        assign_coordinates_op = tf.compat.v1.assign(self.coordinates, coordinates_)

        coordinate_grad_op = {
            k: self.coordinate_product(
                base,
                self.coordinates,
            )
            for k, base in self.bases.items()
        }

        update_op = [
            tf.compat.v1.assign(
                weight,
                weight
                - self.config.learning_rate
                * tf.reshape(coordinate_grad_op[k], weight.shape),
            )
            for k, weight in self.weights.items()
        ]

        update_gradient_op = {
            k: self.coordinate_product(
                base,
                self.coordinates,
            )
            for k, base in self.bases.items()
        }
        update_norm = tf.math.add_n(
            [tf.reduce_sum(tf.square(g)) for k, g in update_gradient_op.items()]
        )
        normed_update_gradient_op = {
            k: v / update_norm for k, v in update_gradient_op.items()
        }

        sess.run(tf.compat.v1.global_variables_initializer())

        dataset = tfds.as_numpy(self.data["train"])
        dataset_val = tfds.as_numpy(self.data["test"])

        def grad_w(data):
            return sess.run(
                normed_grad, {image_: data["image"], label_: data["label"], t_: t}
            )

        def reset_c():
            return sess.run(reset_coordinates_op)

        def apply_base(dict_tensor, axis=0):
            return sess.run(
                apply_base_op,
                feed_dict={
                    apply_base_axis_: axis,
                    **{
                        ph: np.reshape(dict_tensor[k], [-1])
                        for k, ph in apply_base_.items()
                    },
                },
            )

        def assign_c(c):
            return sess.run(assign_coordinates_op, feed_dict={coordinates_: c})

        def grad_c(data, and_assign=False):
            coor = sess.run(
                compute_coordinates_op,
                {image_: data["image"], label_: data["label"], t_: t},
            )
            if and_assign:
                assign_c(coor)

            return coor

        def weight_update():
            return sess.run(update_op)

        def update_gradient(normed=True):
            if normed:
                return sess.run(normed_update_gradient_op)
            return sess.run(update_gradient_op)

        def loss_and_acc(data):
            l, a = sess.run(
                [loss, acc], {image_: data["image"], label_: data["label"], t_: t}
            )
            return {"loss": l, "acc": a}

        def to_vector(dict_tensor):
            return np.concatenate(
                [np.reshape(v, [-1]) for k, v in dict_tensor.items()], axis=0
            )

        assigned_bases = [None] * self.config.base_dimensions
        base_corr = [None] * self.config.base_dimensions

        def compute_correlation(vector):
            if not self.config.compute_gradient_correlation:
                return
            for i, b in enumerate(assigned_bases):
                if b is None:
                    continue
                base_corr[i] = pearsonr(vector, b)[0]

        for epoch in range(self.config.epochs):
            self.record["epoch"] = epoch
            loss_sum = 0
            acc_sum = 0
            for t in range(steps_per_epoch_train):
                sample = next(dataset)
                result = loss_and_acc(sample)
                gradient = grad_w(sample)
                gradient_vector = to_vector(gradient)

                if self.config.offset:
                    # use past gradients only
                    compute_correlation(gradient_vector)
                    grad_c(sample, and_assign=True)
                    if self.config.use_update_as_base:
                        update_gradient(normed=True)
                    weight_update()
                    reset_c()
                    apply_base(gradient, axis=t % self.config.base_dimensions)
                    assigned_bases[t % self.config.base_dimensions] = gradient_vector
                else:
                    apply_base(gradient, axis=t % self.config.base_dimensions)
                    assigned_bases[t % self.config.base_dimensions] = gradient_vector

                    compute_correlation(gradient_vector)
                    grad_c(sample, and_assign=True)
                    if t < self.config.base_dimensions:
                        continue

                    weight_update()
                    reset_c()

                loss_sum += result["loss"]
                acc_sum += result["acc"]
            self.record["loss"] = loss_sum / steps_per_epoch_train
            self.record["acc"] = acc_sum / steps_per_epoch_train * 100

            # validation
            reset_c()
            loss_sum = 0
            acc_sum = 0
            for k in range(steps_per_epoch_val):
                result = loss_and_acc(next(dataset_val))
                loss_sum += result["loss"]
                acc_sum += result["acc"]
            self.record["val_loss"] = loss_sum / steps_per_epoch_val
            self.record["val_acc"] = acc_sum / steps_per_epoch_val * 100

            if self.config.stop_on_nan:
                if (
                    np.isnan(self.record["val_loss"])
                    or self.record["val_loss"] > 1000
                    or (epoch > 4 and self.record["val_acc"] <= 15)
                ):
                    self.record.save(echo=True)
                    self.log.info(
                        "Training finished early due to NaNs or non-convergence"
                    )
                    return

            self.record.save(echo=True)

    def on_execute(self):
        pass

    def coordinate_product(self, base, coordinates=None):
        if coordinates is None:
            coordinates = self.coordinates

        return tf.squeeze(
            tf.matmul(tf.reshape(coordinates, [1, self.config.base_dimensions]), base)
        )

    def _apply_layer_op(self, weight, layer_id, weight_id, trainable):
        if not trainable:
            return weight

        self.weights[(layer_id, weight_id)] = weight

        base = tf.keras.backend.zeros(
            shape=[self.config.base_dimensions] + self.get_flattened_shape(weight),
            dtype=tf.float32,
            name="bases/" + str(layer_id) + "/" + str(weight_id),
        )

        self.bases[(layer_id, weight_id)] = base

        return weight + tf.reshape(self.coordinate_product(base), weight.shape)
