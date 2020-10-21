# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from mixins.surgery import cosine_similarity

from .rbd import RbdModel

tf.compat.v1.disable_v2_behavior()


class CorrModel(RbdModel):
    """Computes gradient correlation"""

    def on_create(self):
        self.weights = []

        self._ipu_.get_session()
        self._ipu_.configure()
        self._image_data_.load()

        infeed = self._ipu_.infeed_queue(self.data["train"])

        # infer shapes
        preflight = self._image_network_.load()
        preflight.build(
            input_shape=[self.config.data.batch_size]
            + list(self.dataset_info.features["image"].shape)
        )
        theta = [
            tf.zeros(shape=v.shape, dtype=v.dtype)
            for v in preflight.trainable_variables
        ]

        with self._ipu_.device():
            self.train_op = self._ipu_.compile(
                lambda: self._ipu_.loops_repeat(
                    n=self._image_data_.steps_per_epoch("train"),
                    body=lambda *args, **kwargs: self._build(*args, **kwargs),
                    inputs=[
                        tf.constant(0, tf.float32),
                        tf.constant(0, tf.float32),
                        tf.constant(0, tf.float32),
                    ]
                    + theta,
                    infeed_queue=infeed,
                    divide_by_n=True,
                ),
                [],
            )

        self.network.summary()

        if self.config.ipu.enabled:
            self.sess.run(infeed.initializer)
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def _build(
        self,
        total_corr,
        total_loss,
        total_acc,
        *previous_gradients,
        image=None,
        label=None
    ):
        self._apply_layer_ops()
        self.network = self._image_network_.load()
        predictions = self.network(image)
        self._rollback_layer_ops()

        loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(label, predictions)
        )
        acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(label, predictions))
        gradients = tf.gradients(loss, self.network.trainable_variables)

        # compute correlation
        a = tf.concat([tf.reshape(v, [-1]) for v in previous_gradients], axis=0)
        b = tf.concat([tf.reshape(v, [-1]) for v in gradients], axis=0)
        corr = cosine_similarity(a, b)

        with tf.control_dependencies(
            [
                tf.assign_add(var, 0 * -self.config.sgd_learning_rate * grad)
                for var, grad in zip(self.network.trainable_variables, gradients)
            ]
        ):
            return tuple(
                [
                    tf.add(total_corr, corr),
                    tf.add(total_loss, loss),
                    tf.add(total_acc, acc),
                ]
                + gradients
            )

    def _apply_layer_op(self, weight, layer_id, weight_id, trainable):
        if not trainable:
            return weight

        # register weight for update
        self.weights.append((weight, weight_id))
        return weight

    def on_execute(self):
        r = self.record
        for epoch in range(1, int(self.config.epochs)):
            out = self.sess.run(self.train_op)
            r["epoch"] = epoch
            r["mean_corr"] = out[0]
            r["loss"] = out[1]
            r["acc"] = out[2]
            r["acc"] *= 100
            r["steps"] = self._image_data_.steps_per_epoch("train")
            r["images"] = self._image_data_.images_per_epoch("train")
            r["images_total"] = r["images"] * epoch
            r["images_per_second"] = r["images"] / self.record.timing()
            r.save(echo=True)
