# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import numpy as np

import tensorflow as tf

from .nes import NesModel

tf.compat.v1.disable_v2_behavior()


class NesRbModel(NesModel):
    """
    Use forwardpass to determine random bases coordinates.
    There is the subtle difference with standard NES in that all offspring are evaluated on the same mini-batch
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
            self.train_op = self._ipu_.compile(
                lambda: self._ipu_.loops_repeat(
                    n=self._image_data_.steps_per_epoch("train"),
                    body=lambda *args, **kwargs: self._build(*args, **kwargs),
                    inputs=[tf.constant(0, tf.float32), tf.constant(0, tf.float32)],
                    infeed_queue=infeed,
                    divide_by_n=True,
                ),
                [],
            )

        self.network.summary()

        # evaluation (placed on CPU)
        self.eval_op = {
            split: self._ipu_.loops_repeat(
                n=self._image_data_.steps_per_epoch(split),
                body=lambda *args, **kwargs: self._build(
                    *args, evaluate=True, **kwargs
                ),
                inputs=[tf.constant(0, tf.float32), tf.constant(0, tf.float32)],
                infeed_queue=self.prepare_data(
                    self.data[split],
                    self._image_data_.steps_per_epoch(split),
                    infeed=False,
                ),
                divide_by_n=True,
                mode="cpu",
            )
            for split in ["validation", "test"]
        }

        if self.config.ipu.enabled:
            self.sess.run(infeed.initializer)
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def _build(
        self, total_loss, total_acc, t, image, label, lr, worker=None, evaluate=False
    ):
        self.coordinates = tf.Variable(
            lambda: tf.zeros(shape=[self.config.base_dimensions])
        )

        if evaluate:
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
            acc = tf.reduce_mean(
                tf.keras.metrics.categorical_accuracy(label, predictions)
            )
            return tf.add(total_loss, loss), tf.add(total_acc, acc)

        if self.config.reset_base_each_step:
            base = t
        else:
            base = 0

        # generate seeds
        seeds = self.base_seeds_generator(base)
        ta_seeds = tf.TensorArray(
            dtype=tf.int32, size=self.config.base_dimensions, element_shape=[]
        ).unstack(seeds)

        def offspring_loop(index, population_loss, population_acc):
            self.offspring_seed = ta_seeds.read(index)
            self._apply_layer_ops()
            self.network = network = self._image_network_.load(
                name=self.config.network,
                classes=self.dataset_info.features["label"].num_classes,
                input_shape=self.dataset_info.features["image"].shape,
            )
            predictions = network(image)
            self._rollback_layer_ops()
            loss_object = tf.keras.losses.CategoricalCrossentropy()
            loss = loss_object(label, predictions)
            acc = tf.reduce_mean(
                tf.keras.metrics.categorical_accuracy(label, predictions)
            )

            write_op = tf.compat.v1.scatter_update(self.coordinates, index, loss)
            with tf.control_dependencies([write_op]):
                return (
                    tf.add(index, 1),
                    tf.add(population_loss, loss),
                    tf.add(population_acc, acc),
                )

        offspring, offspring_loss, offspring_acc = self._ipu_.loops_repeat(
            n=self.config.base_dimensions,
            body=offspring_loop,
            inputs=[
                tf.constant(0, tf.int32),
                tf.constant(0, tf.float32),
                tf.constant(0, tf.float32),
            ],
            divide_by_n=True,
            mode="tensorflow",
        )

        with tf.control_dependencies([offspring]):
            values = tf.identity(self.coordinates)

        if self.config.transformation == "norm":
            zero = values - (
                tf.ones(self.config.base_dimensions) * tf.reduce_min(values)
            )
            norm = tf.divide(zero, tf.reduce_max(zero))
            transformed = (norm - 0.5) * -1  # shift and invert
        elif self.config.transformation == "ranks":
            argsort = tf.argsort(values, direction="DESCENDING")
            ranks = tf.compat.v1.scatter_update(
                self.coordinates,
                argsort,
                tf.cast(tf.range(tf.shape(values)[0]), dtype=tf.float32),
            )
            transformed = (
                tf.divide(ranks, tf.cast(tf.shape(ranks)[0] - 1, dtype=tf.float32))
                - 0.5
            )
        else:
            transformed = tf.identity(values)

        coordinates = transformed

        update_ops = []
        for (weight, state) in self._weights:
            gradient = self._random_base_.product(
                coordinates=coordinates,
                seeds=seeds,
                state=state,
                shape=weight.shape,
            )
            step = lr * gradient
            update_op = tf.keras.backend.update_add(weight, step)
            update_ops.append(update_op)
        update_op = tf.group(update_ops)

        with tf.control_dependencies([update_op]):
            return tf.add(total_loss, offspring_loss), tf.add(total_acc, offspring_acc)

    def on_execute(self):
        r = self.record
        for epoch in range(1, int(self.config.epochs)):
            loss, acc = self.sess.run(self.train_op)
            self.evaluation_network.set_weights(self.network.get_weights())
            r["val_loss"], r["val_acc"] = self.sess.run(self.eval_op["validation"])
            r["val_acc"] *= 100
            r["epoch"] = epoch
            r["loss"] = loss
            r["acc"] = acc * 100
            r["steps"] = self._image_data_.steps_per_epoch("train")
            r["images"] = self._image_data_.images_per_epoch("train")
            r["images_total"] = r["images"] * epoch
            r["images_per_second"] = r["images"] / self.record.timing()
            coordinates = self.sess.run(self.coordinates)
            r["coordinates"] = {
                "mean": np.mean(coordinates),
                "std": np.std(coordinates),
                "min": np.min(coordinates),
                "max": np.max(coordinates),
            }

            if self.config.stop_on_nan:
                if np.isnan(r["val_loss"]) or (epoch > 10 and r["val_acc"] <= 15):
                    r.save(echo=True)
                    self.log.info(
                        "Training finished early due to NaNs or non-convergence"
                    )
                    return

            r.save(echo=True)

        self.evaluation_network.set_weights(self.network.get_weights())
        test_loss, test_acc = self.sess.run(self.eval_op["test"])
        self.storage.save_data(
            "eval.json", {"test_acc": test_acc, "test_loss": test_loss}
        )
