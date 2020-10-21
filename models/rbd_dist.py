# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import copy
import time

import numpy as np
import ray

import tensorflow as tf
from tensorflow.keras import backend as K

from .rbd import RbdModel

tf.compat.v1.disable_v2_behavior()


class RbdDistModel(RbdModel):
    """
    Distributed version using Ray https://github.com/ray-project/ray
    """

    def on_after_init(self):
        if self.is_worker:
            return

        # Initialise workers
        self.workers = [
            ray.remote(self.__class__).remote(
                config=self.config.toDict(evaluate=True),
                flags={"WORKER_ID": worker_id, **self.flags.toDict(evaluate=True)},
            )
            for worker_id in range(self.config.workers)
        ]

        ray.get([worker.create.remote() for i, worker in enumerate(self.workers)])

    @property
    def is_worker(self):
        return self.flags.get("WORKER_ID", -1) >= 0

    def on_create(self):
        self.metrics = {"loss": [], "acc": []}
        self._ipu_.configure(num_ipus=1)
        self._ipu_.get_session()

        # evaluation
        if not self.is_worker:
            splits = ["train", "validation", "test"]
            self._rollback_layer_ops()
            feed = {
                k: self._ipu_.infeed_queue(self.data[k], feed_name=str(k) + "_infeed")
                for k in splits
            }
            with self._ipu_.device():
                self.eval_op = {
                    k: self._ipu_.compile(
                        lambda: self._ipu_.loops_repeat(
                            n=self._image_data_.steps_per_epoch(k),
                            body=lambda *args, **kwargs: self._build_evaluate(
                                *args, **kwargs
                            ),
                            inputs=[
                                tf.constant(0, tf.float32),
                                tf.constant(0, tf.float32),
                                tf.constant(0, tf.float32),
                            ],
                            infeed_queue=feed[k],
                            divide_by_n=True,
                        )
                    )
                    for k in splits
                }
                self.evaluation_network.get_weights()  # workaround to initialize weights
                for k in splits:
                    try:
                        self.sess.run(feed[k].initializer)
                    except ValueError:
                        pass

        self._apply_layer_ops()

        outfeed_queue = self._ipu_.outfeed_queue("outfeed" + str(self.flags.SEED))

        if self.config.same_images_for_each_worker:
            raise ValueError("same_images_for_each_worker option is not supported")

        train_feed = self.prepare_data("train", replication_factor=1)

        with self._ipu_.device():
            self.train_op = self._ipu_.compile(
                lambda: self._ipu_.loops_repeat(
                    n=1,
                    body=lambda *args, **kwargs: self._build_compute_values(
                        *args, **kwargs
                    ),
                    inputs=[
                        tf.constant(0, tf.float32),
                        tf.constant(0, tf.float32),
                    ],
                    infeed_queue=train_feed,
                    outfeed_queue=outfeed_queue,
                    divide_by_n=False,
                )
            )

        self._ipu_.move_variable_initialization_to_cpu()
        try:
            self.sess.run(train_feed.initializer)
        except ValueError:
            pass
        self.outfeed = outfeed_queue.dequeue()

        # update
        self.t = tf.compat.v1.placeholder(tf.int32, shape=[], name="t_placeholder")
        self._t = None
        self.lr = tf.compat.v1.placeholder(tf.float32, shape=[])
        self._lr = None
        self.worker_id = tf.compat.v1.placeholder(tf.int32, shape=[])
        self._worker_id = self.flags.get("WORKER_ID", -1)
        self.values = [
            K.placeholder(shape=c.shape, dtype=c.dtype) for c in self.coordinates
        ]
        with self._ipu_.device():
            self.update_op = self._ipu_.compile(
                lambda *args, **kwargs: self._build_update(*args, **kwargs),
                inputs=[self.t, self.worker_id, self.lr] + self.values,
            )

        # init
        self.timestep(0)
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def add_base(self, index=0, dimensions=None):
        self.bases.append(tf.Variable(lambda: self.generate_base_seeds(index)))

    def timestep(self, t=None):
        if t is None:
            self._t += 1
        else:
            self._t = t

        # update learning rate based on schedule

        if self.config.learning_rate_warmup.steps > 0 and self._t < int(
            self.config.learning_rate_warmup.steps
        ):
            if self.config.learning_rate_warmup.mode == "linear":
                self._lr = 1 / (
                    float(self.config.learning_rate_warmup.steps) * self._t + 1e-7
                )
            else:
                self._lr = (
                    np.exp(
                        np.log(2)
                        / (self.config.learning_rate_warmup.steps * self._t + 1e-7)
                    )
                    - 1
                )
        else:
            self._lr = 1

        self._lr *= self.config.learning_rate

    def _build_compute_values(
        self, total_loss, total_acc, t, image, label, lr, worker, outfeed_queue
    ):
        loss, acc, gradients = self._build_compute_gradients(t, image, label)

        return (
            tf.add(total_loss, loss),
            tf.add(total_acc, acc),
            outfeed_queue.enqueue(gradients),
        )

    def _build_update(self, t, worker_id, lr, *coordinates):
        reset_bases = [
            tf.compat.v1.assign(b, self.generate_base_seeds(index, t, worker_id))
            for index, b in enumerate(self.bases)
        ]

        with tf.control_dependencies(reset_bases):
            return self._build_apply_gradients(lr, coordinates, t)

    def _build_compute_gradients(self, t, image, label):
        self.network = self._image_network_.load(
            name=self.config.network,
            classes=self.dataset_info.features["label"].num_classes,
            input_shape=self.dataset_info.features["image"].shape,
            inputs=image,
        )
        predictions = self.network(image)
        loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(label, predictions)
        )
        acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(label, predictions))

        if self.config.reset_coordinates_each_step:
            reset_coordinates_op = [
                tf.compat.v1.assign(
                    c, tf.constant(value=0.0, shape=c.shape, dtype=tf.float32)
                )
                for c in self.coordinates
            ]
        else:
            reset_coordinates_op = [tf.no_op()]

        if self.config.reset_base_each_step:
            reset_base_op = [
                tf.compat.v1.assign(b, self.generate_base_seeds(index, t))
                for index, b in enumerate(self.bases)
            ]
        else:
            reset_base_op = [tf.no_op()]

        with tf.control_dependencies(reset_coordinates_op + reset_base_op):
            gradients = tf.gradients(loss, self.coordinates)

        return loss, acc, gradients

    def on_execute(self):
        self.summary(self.config.workers)
        self.weights_to_workers()
        self.update = None

    def on_execute_iteration(self, iteration: int):
        self.update = [
            worker.compute_values.remote(update=self.update) for worker in self.workers
        ]

        # wait for update (blocking)
        self.apply_updates(self.update)

        if self.config.compile_test and iteration > 2:
            self.log.info(
                f"Compiling was successful for batching={self.config.base.batching}"
            )
            return StopIteration

        # evaluate
        steps_total = iteration * self.config.workers
        last_epoch = (
            self.record.latest["epoch"] if self.record.latest is not None else 0
        )
        epoch = steps_total / float(self._image_data_.steps_per_epoch())

        coordinates = np.array(ray.get(self.update))
        self.coordinate_history.append(coordinates)

        r = self.record
        if steps_total > 0 and epoch >= last_epoch + 1:
            if self.config.store_coordinates:
                self.storage.save_data(
                    "coordinates.npy",
                    np.asarray(self.coordinate_history),
                    overwrite=True,
                )
            self.coordinate_statistics(coordinates)
            weights = self.network.get_weights()
            self.weight_statistics(weights)
            self.evaluation_network.set_weights(weights)
            r["epoch"] = int(epoch)
            r["loss"], r["acc"], r["k_acc"] = self.sess.run(self.eval_op["train"])
            r["acc"] *= 100
            r["k_acc"] *= 100
            r["val_loss"], r["val_acc"], r["val_k_acc"] = self.sess.run(
                self.eval_op["validation"]
            )
            r["val_acc"] *= 100
            r["val_k_acc"] *= 100
            r["num_steps"] = self._image_data_.steps_per_epoch()
            r["num_steps_total"] = steps_total
            r["lr"] = self._lr
            r["workers"] = ray.get(
                [worker.get_metrics.remote() for worker in self.workers]
            )
            r["avg_timesteps_per_second"] = r["num_steps_total"] / r.timing("total")
            r["_iteration"] = iteration

            r.save(echo=True)

        if epoch >= int(self.config.epochs):
            return StopIteration

    def train(self):
        loss, acc = self.sess.run(self.train_op)
        return loss, acc, self.get_params()

    def compute_value(self):
        loss, acc = self.sess.run(self.train_op)
        coordinates = self.sess.run(self.outfeed)

        self.metrics["loss"].append(loss)
        self.metrics["acc"].append(acc * 100)

        return coordinates[0]  # replication_factor = 1

    def compute_values(self, update=None):
        t = time.time()

        if update is not None:
            self.apply_updates(update)

        values = self.compute_value()

        self.metrics["compute_values_time"] = time.time() - t

        return values

    def apply_updates(self, values):
        t = time.time()

        try:
            if isinstance(values[0], ray.ObjectID):
                # retrieve values
                values = np.array(ray.get(values))
        except IndexError:
            pass

        # values.shape = (WORKERS, NUM_COORDINATES, COORDINATES)
        for worker_id, worker_values in enumerate(values):
            self.sess.run(
                self.update_op,
                feed_dict={
                    self.lr: self._lr,
                    self.t: self._t,
                    self.worker_id: worker_id,
                    **{p: v for p, v in zip(self.values, worker_values)},
                },
            )

        self.metrics["apply_update_time"] = time.time() - t

        self.timestep()

        return self.metrics

    def on_destroy(self):
        if not self.is_worker:
            for worker in self.workers:
                ray.get(worker.on_destroy.remote())
        super().on_destroy()

    def get_metrics(self):
        metrics = copy.deepcopy(self.metrics)

        # aggregate
        metrics["loss"] = np.mean(metrics["loss"])
        metrics["acc"] = np.mean(metrics["acc"])
        metrics["acc"] *= 100

        # reset
        self.metrics["loss"] = []
        self.metrics["acc"] = []

        return metrics

    def weights_to_workers(self):
        self.log.info("Syncing weights to workers")
        t = time.time()
        theta = self.get_params()
        ray.get([worker.set_params.remote(theta) for worker in self.workers])
        self.log.info(
            f"Syncing weights to workers completed in {time.time() - t} seconds"
        )

    def weights_from_workers(self):
        self.log.info("Syncing weights from workers")
        t = time.time()
        theta = ray.get(self.workers[0].get_params.remote())
        self.set_params(theta)
        self.log.info(
            f"Syncing weights from workers completed in {time.time() - t} seconds"
        )

    def config_base_learning_rate(self, base_learning_rate):
        if isinstance(base_learning_rate, str):
            eval("base_learning_rate = " + base_learning_rate)
        lr = base_learning_rate * self.config.data.batch_size
        return lr
