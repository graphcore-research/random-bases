# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import time

import numpy as np

import tensorflow as tf
from mixins.random_base import get_noise
from tensorflow.keras import backend as K
from vendor.img.models.image import ImageModel

from .base import BaseModel

tf.compat.v1.disable_v2_behavior()


class NesModel(ImageModel, BaseModel):
    """
    Re-implementation of simple natural evolution strategies (http://arxiv.org/abs/1703.03864)
    """

    def on_create(self):
        self._ipu_.get_session()
        self._ipu_.configure()
        self._image_data_.load()

        self.offspring_seed = None
        self._weights = []

        infeed = self.prepare_data(
            self.data["train"], self._image_data_.steps_per_epoch("train")
        )

        with self._ipu_.device():

            coordinates = tf.Variable(tf.zeros(shape=[self.config.base_dimensions]))
            t = tf.Variable(tf.zeros(shape=[], dtype=tf.int32))

            def optimize():

                if self.config.reset_base_each_step:
                    base = t
                else:
                    base = 0

                # generate seeds
                seeds = self.base_seeds_generator(base)
                ta_seeds = tf.TensorArray(
                    dtype=tf.int32, size=self.config.base_dimensions, element_shape=[]
                ).unstack(seeds)

                def body(
                    index, total_loss, total_acc, coord, t, image, label, lr, worker
                ):
                    self.offspring_seed = ta_seeds.read(index)

                    self._apply_layer_ops()
                    self.network = self._image_network_.load(
                        name=self.config.network,
                        classes=self.dataset_info.features["label"].num_classes,
                        input_shape=self.dataset_info.features["image"].shape,
                    )
                    predictions = self.network(image)
                    self._rollback_layer_ops()
                    loss_object = tf.keras.losses.CategoricalCrossentropy()
                    loss = loss_object(label, predictions)
                    acc = tf.reduce_mean(
                        tf.keras.metrics.categorical_accuracy(label, predictions)
                    )

                    coord = tf.compat.v1.scatter_update(coordinates, index, loss)

                    return (
                        tf.add(index, 1),
                        tf.add(total_loss, loss),
                        tf.add(total_acc, acc),
                        coord,
                    )

                # population
                (
                    offspring,
                    offspring_loss,
                    offspring_acc,
                    values,
                ) = self._ipu_.loops_repeat(
                    n=self.config.base_dimensions,
                    body=body,
                    inputs=[
                        tf.constant(0, tf.int32),
                        tf.constant(0, tf.float32),
                        tf.constant(0, tf.float32),
                        tf.zeros(shape=[self.config.base_dimensions]),
                    ],
                    infeed_queue=infeed,
                    divide_by_n=False,
                )

                if self.config.transformation == "norm":
                    zero = values - (
                        tf.ones(self.config.base_dimensions) * tf.reduce_min(values)
                    )
                    norm = tf.divide(zero, tf.reduce_max(zero))
                    transformed = (norm - 0.5) * -1  # shift and invert
                elif self.config.transformation == "ranks":
                    argsort = tf.argsort(values, direction="DESCENDING")
                    ranks = tf.compat.v1.scatter_update(
                        coordinates,
                        argsort,
                        tf.cast(tf.range(tf.shape(values)[0]), dtype=tf.float32),
                    )
                    transformed = (
                        tf.divide(
                            ranks, tf.cast(tf.shape(ranks)[0] - 1, dtype=tf.float32)
                        )
                        - 0.5
                    )
                else:
                    transformed = tf.identity(values)

                normalized = transformed / float(
                    self.config.noise_std * self.config.base_dimensions
                )

                update_ops = []
                for (weight, state) in self._weights:
                    gradient = self._random_base_.product(
                        coordinates=normalized,
                        seeds=seeds,
                        state=state,
                        shape=weight.shape,
                    )
                    step = self.config.learning_rate * gradient
                    update_op = K.update_add(weight, step)
                    update_ops.append(update_op)
                update_op = tf.group(update_ops)

                # increase t
                timestep = t.assign_add(1)

                with tf.control_dependencies([update_op, timestep]):
                    return (
                        tf.math.divide(offspring_loss, self.config.base_dimensions),
                        tf.math.divide(offspring_acc, self.config.base_dimensions),
                    )

            self.train_op = self._ipu_.compile(optimize, [])

        self.network.summary()

        # evaluation (placed on CPU)
        self.eval_op = {
            split: self._ipu_.loops_repeat(
                n=self._image_data_.steps_per_epoch(split),
                body=lambda *args, **kwargs: self._evaluate(*args, **kwargs),
                inputs=[
                    tf.constant(0, tf.float32),
                    tf.constant(0, tf.float32),
                    tf.constant(0, tf.float32),
                ],
                infeed_queue=self.prepare_data(split, infeed=False),
                divide_by_n=True,
                mode="cpu",
            )
            for split in ["validation", "test"]
        }

        if self.config.ipu.enabled:
            self.sess.run(infeed.initializer)
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def _evaluate(
        self, total_loss, total_acc, total_k_acc, t, image, label, lr, worker
    ):
        if not getattr(self, "evaluation_network", None):
            self.evaluation_network = self._image_network_.load(
                name=self.config.network,
                classes=self.dataset_info.features["label"].num_classes,
                input_shape=self.dataset_info.features["image"].shape,
            )
        predictions = self.evaluation_network(image)
        loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(label, predictions)
        )
        acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(label, predictions))
        k_acc = tf.reduce_mean(
            tf.keras.metrics.top_k_categorical_accuracy(
                label, predictions, k=self.config.top_k_acc
            )
        )
        return (
            tf.add(total_loss, loss),
            tf.add(total_acc, acc),
            tf.add(total_k_acc, k_acc),
        )

    def _apply_layer_op(self, weight, layer_id, weight_id, trainable):
        if not trainable:
            return weight

        # each variable has different random state
        state = layer_id * 1000 + weight_id

        # register weight for update
        self._weights.append((weight, state))
        epsilon = get_noise(
            shape=weight.shape,
            seed=self.offspring_seed,
            state=state,
            dtype=weight.dtype,
            dist=self.config.base.distribution,
        )
        return weight + self.config.noise_std * epsilon

    def train(self):
        r = self.experiment.record
        epoch = 0
        iteration = 1
        while epoch < int(self.config.epochs):
            t = time.time()
            loss, acc = self.sess.run(self.train_op)
            r["seconds_per_step"] = time.time() - t
            r["steps_per_second"] = 1.0 / r["seconds_per_step"]
            if self.config.validation:
                self.evaluation_network.set_weights(self.network.get_weights())
                r["val_loss"], r["val_acc"] = self.sess.run(self.eval_op["validation"])
            r["val_acc"] *= 100
            r["loss"] = loss
            r["acc"] = acc * 100
            r["lr"] = self.config.learning_rate
            r["iteration"] = iteration
            r["images"] = self.config.data.batch_size * self.config.base_dimensions
            r["images_total"] = r["images"] * iteration
            r["images_per_second"] = r["images"] / r["seconds_per_step"]
            epoch = (
                r["images_total"]
                / self.experiment.dataset_info.splits["train"].num_examples
            )
            r["epoch"] = epoch

            if self.config.stop_on_nan:
                if np.isnan(r["val_loss"]):
                    r.save(echo=True)
                    self.log.info(
                        "Training finished early due to NaNs or non-convergence"
                    )
                    return

            r.save(echo=True)
            iteration += 1

        self.evaluation_network.set_weights(self.network.get_weights())
        test_loss, test_acc = self.sess.run(self.eval_op["test"])
        self.storage.save_data(
            "eval.json", {"test_acc": test_acc, "test_loss": test_loss}
        )
