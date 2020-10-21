# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

from collections import deque

import numpy as np

import tensorflow as tf

from .nes import NesModel

tf.compat.v1.disable_v2_behavior()


def get_noise(shape, seed, state, dtype=None, dist="normal"):
    if dtype is None:
        dtype = tf.float32
    if isinstance(state, int):
        state = tf.constant(state, tf.int32)

    stacked_seed = tf.stack([state, tf.cast(tf.abs(seed), dtype=tf.int32)])

    sign = tf.cast(tf.math.sign(seed), dtype)

    if dist == "zeros":
        return tf.zeros(shape=shape, dtype=dtype)

    if dist == "normal":
        return tf.random.stateless_normal(
            shape, mean=0.0, stddev=1.0, seed=stacked_seed, dtype=dtype
        )

    if dist == "lognormal":
        return sign * tf.math.exp(
            tf.random.stateless_normal(shape, seed=stacked_seed, dtype=dtype)
        ) - tf.math.exp(0.5)

    if dist.startswith("bernoulli"):
        p = tf.constant(float(dist.split("-")[-1]), dtype=tf.float32)
        u = tf.random.stateless_uniform(
            shape, minval=0.0, maxval=1.0, seed=stacked_seed, dtype=tf.float32
        )
        return sign * tf.cast(
            tf.cast(tf.math.greater(u, 1 - p), tf.float32), dtype=dtype
        )

    if dist == "uniform":
        # zero mean uniform
        u = tf.random.stateless_uniform(
            shape, minval=0.0, maxval=1.0, seed=stacked_seed, dtype=dtype
        )
        return sign * tf.cast(u, dtype=dtype)

    raise ValueError(f"Invalid distribution {dist}")


class RbdAsymModel(NesModel):
    """
    Plots loss landscape in random directions
    """

    def on_create(self):
        # bins should be uneven for symmetry
        assert self.config.bins % 2 == 1

        self.coordinate_history = deque()
        self.outfeed_data = {}
        self._ipu_.get_session()
        self._ipu_.configure()
        self._image_data_.load()

        self.offspring_seed = None
        self.index = None
        self._weights = []

        infeed = self.prepare_data(
            self.data["train"], self._image_data_.steps_per_epoch("train")
        )

        outfeed_queue = self._ipu_.outfeed_queue(
            "outfeed" + str(self.flags.SEED)  # , outfeed_mode="LAST"
        )

        with self._ipu_.device():
            self.train_op = self._ipu_.compile(
                lambda: self._ipu_.loops_repeat(
                    n=self._image_data_.steps_per_epoch("train"),
                    body=lambda *args, **kwargs: self._build(*args, **kwargs),
                    inputs=[
                        tf.constant(0, tf.float32),
                        tf.constant(0, tf.float32),
                    ],
                    infeed_queue=infeed,
                    outfeed_queue=outfeed_queue,
                    divide_by_n=True,
                ),
                [],
            )
            self.outfeed = outfeed_queue.dequeue()

        self.network.summary()

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
        self,
        total_loss,
        total_acc,
        t,
        image,
        label,
        lr,
        worker=None,
        outfeed_queue=None,
        evaluate=False,
    ):
        self.coordinates = tf.Variable(
            lambda: tf.zeros(shape=[self.config.base_dimensions * self.config.bins])
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

        # generate random direction seeds
        dim_seeds = tf.random.stateless_uniform(
            shape=[self.config.base_dimensions],
            seed=tf.stack(
                [tf.constant(self.flags.SEED, tf.int32), tf.cast(base, tf.int32)]
            ),
            maxval=2 ** 30,
            dtype=tf.int32,
        )
        seeds = tf.tile(dim_seeds, [self.config.bins])

        ta_seeds = tf.TensorArray(
            dtype=tf.int32,
            size=self.config.base_dimensions * self.config.bins,
            element_shape=[],
        ).unstack(seeds)

        def offspring_loop(index, population_loss, population_acc):
            self.index = index
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

            write_op = tf.compat.v1.scatter_update(self.coordinates, index, loss)
            with tf.control_dependencies([write_op]):
                return (
                    tf.add(index, 1),
                    tf.add(population_loss, loss),
                    tf.add(population_acc, acc),
                )

        offspring, offspring_loss, offspring_acc = self._ipu_.loops_repeat(
            n=self.config.base_dimensions * self.config.bins,
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
            coordinates = tf.identity(self.coordinates)

        # reshape into
        #         base_dimensions
        #         -1   -1     -1
        # bins    0     0      0      * binsize
        #         1     1      1
        binned_coordinates = tf.reshape(
            coordinates, [self.config.bins, self.config.base_dimensions]
        )

        self.outfeed_data["coord"] = binned_coordinates

        # SGD ------

        # unfortunately, TensorFlow does not allow re-using the weights
        #  that have been defined in the while_loop above, so we have
        #  to create a second copy of the network to compute SGD steps
        self.sgd_network = self._image_network_.load(
            name=self.config.network,
            classes=self.dataset_info.features["label"].num_classes,
            input_shape=self.dataset_info.features["image"].shape,
        )
        self.sgd_network.build(input_shape=image.get_shape().as_list())
        with tf.control_dependencies(
            [  # copy weights
                tf.compat.v1.assign(dest, src)
                for src, dest in zip(
                    self.network.trainable_variables,
                    self.sgd_network.trainable_variables,
                )
            ]
        ):
            predictions = self.sgd_network(image)
            loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(label, predictions)
            )
            sgd_grads = tf.gradients(loss, self.sgd_network.trainable_variables)
            sgd = {
                v.name: g for g, v in zip(sgd_grads, self.network.trainable_variables)
            }

        # RBD ------

        values = (
            binned_coordinates[self.config.bins // 2 + 1, :]
            - binned_coordinates[self.config.bins // 2 - 1, :]
        )

        assert values.get_shape().as_list() == [self.config.base_dimensions]

        if self.config.transformation == "norm":
            zero = values - (
                tf.ones(self.config.base_dimensions) * tf.reduce_min(values)
            )
            norm = tf.divide(zero, tf.reduce_max(zero))
            transformed = (norm - 0.5) * -1  # shift and invert
        elif self.config.transformation == "ranks":
            var = tf.Variable(lambda: tf.zeros(shape=[self.config.base_dimensions]))
            argsort = tf.argsort(values, direction="DESCENDING")
            ranks = tf.compat.v1.scatter_update(
                var,
                argsort,
                tf.cast(tf.range(tf.shape(values)[0]), dtype=tf.float32),
            )
            transformed = (
                tf.divide(ranks, tf.cast(tf.shape(ranks)[0] - 1, dtype=tf.float32))
                - 0.5
            )
        else:
            transformed = tf.identity(values)

        # hessian
        if self.config.compute_hessian:
            hessians = [
                self._random_base_.product(
                    coordinates=transformed,
                    seeds=dim_seeds,
                    state=state,
                    shape=weight.shape,
                    mode="second_order",
                    elementwise=True,
                    elementwise_reduce=True,
                )
                for (weight, state) in self._weights
            ]
            hessian_norm = tf.add_n(hessians)

            zero_ = hessian_norm - (
                tf.ones(self.config.base_dimensions) * tf.reduce_min(hessian_norm)
            )
            hessian_normalized = tf.divide(zero_, tf.reduce_max(zero_) + 1e-15) - 0.5

            self.outfeed_data["hessian_norm"] = hessian_normalized

            transformed = transformed / (hessian_normalized + 1e-15)

        nes = {
            weight.name: self._random_base_.product(
                coordinates=transformed,
                seeds=dim_seeds,
                state=state,
                shape=weight.shape,
            )
            for (weight, state) in self._weights
        }

        if not self.config.update_schedule.enabled:
            # apply update
            update_ops = []
            for var1, var2 in zip(
                self.network.trainable_variables,
                self.sgd_network.trainable_variables,
            ):
                name = var1.name
                for v in [var1, var2]:
                    # decide what step to use
                    if self.config.use_sgd:
                        step = -self.config.sgd_learning_rate * sgd[name]
                    else:
                        step = lr * nes[name]

                    update_ops.append(
                        tf.compat.v1.assign_add(v, tf.cast(step, v.dtype))
                    )
            update_op = tf.group(update_ops)
        else:
            update_op = tf.cond(
                self.update_schedule(t),
                lambda: tf.group(
                    [
                        tf.compat.v1.assign_add(
                            v, -self.config.sgd_learning_rate * sgd[var1.name]
                        )
                        for var1, var2 in zip(
                            self.network.trainable_variables,
                            self.sgd_network.trainable_variables,
                        )
                        for v in [var1, var2]
                    ]
                ),
                lambda: tf.group(
                    [
                        tf.compat.v1.assign_add(v, lr * nes[var1.name])
                        for var1, var2 in zip(
                            self.network.trainable_variables,
                            self.sgd_network.trainable_variables,
                        )
                        for v in [var1, var2]
                    ]
                ),
            )

        with tf.control_dependencies([update_op]):
            with tf.control_dependencies([outfeed_queue.enqueue(self.outfeed_data)]):
                return (
                    tf.add(total_loss, offspring_loss),
                    tf.add(total_acc, offspring_acc),
                )

    def update_schedule(self, t):
        # Return True to update with SGD, False to use RBD gradient instead
        #  e.g. to interleave SGD and RBD return tf.equal(tf.math.floormod(t, 2), 0)
        if self.config.update_schedule.mode == "sgd_first":
            return tf.less(
                t,
                self._image_data_.steps_per_epoch()
                * self.config.update_schedule.epochs,
            )

        if self.config.update_schedule.mode == "sgd_last":
            return tf.greater_equal(
                t,
                self._image_data_.steps_per_epoch()
                * self.config.update_schedule.epochs,
            )

        assert False

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

        if self.config.base.normalized is False:
            norm_factor = 1.0
        elif self.config.base.normalized is True:
            norm_factor = tf.norm(epsilon)
        else:
            norm_factor = self.config.base.normalized

        return weight + self.get_multiplier(self.index) * epsilon / norm_factor

    def get_multiplier(self, index):
        """Maps from linear index to std multiplier

            0 1 2 | 3 4 5 | 6 7 8
        ->   -1   |   0   |  +1
        """
        offset = tf.constant(self.config.bins // 2, dtype=tf.int32)
        bucket = tf.math.floordiv(index, self.config.base_dimensions)
        return self.config.binsize * tf.cast(bucket - offset, tf.float32)

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
            outfeed_data = self.sess.run(self.outfeed)
            coordinates = outfeed_data["coord"]
            r["coordinates"] = {
                "mean": np.mean(coordinates),
                "std": np.std(coordinates),
                "min": np.min(coordinates),
                "max": np.max(coordinates),
            }

            if self.config.store_coordinates:
                self.coordinate_history.append(coordinates)
                self.storage.save_data(
                    "coordinates.npy",
                    np.asarray(self.coordinate_history),
                    overwrite=True,
                )

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
