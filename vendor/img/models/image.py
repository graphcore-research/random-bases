# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import numpy as np
from machinable import Component

import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer as KerasLayerBase

tf.compat.v1.disable_v2_behavior()


class ImageModel(Component):
    def on_create(self):
        self._ipu_.get_session()
        self._image_data_.load()
        self._ipu_.configure(num_ipus=self.config.workers)

        # evaluation (placed on CPU)
        self.eval_op = {
            split: self._ipu_.loops_repeat(
                n=self._image_data_.steps_per_epoch(split),
                body=lambda *args, **kwargs: self._build_evaluate(*args, **kwargs),
                inputs=[
                    tf.constant(0, tf.float32),
                    tf.constant(0, tf.float32),
                    tf.constant(0, tf.float32),
                ],
                infeed_queue=tf.compat.v1.data.make_one_shot_iterator(self.data[split]),
                divide_by_n=True,
                mode="cpu",
            )
            for split in ["validation", "test"]
        }

        self._apply_layer_ops()

        # training
        infeed_data = self.prepare_data("train")
        with self._ipu_.device():
            n = self._image_data_.steps_per_epoch("train")
            if self.config.get("compile_test", False):
                n = 1
            self.train_op = self._ipu_.compile(
                lambda: self._ipu_.loops_repeat(
                    n=n,
                    body=lambda *args, **kwargs: self._build(*args, **kwargs),
                    inputs=[tf.constant(0, tf.float32), tf.constant(0, tf.float32)],
                    infeed_queue=infeed_data,
                    divide_by_n=True,
                )
            )
            self.initialisation = self.network.get_weights()
            self._ipu_.move_variable_initialization_to_cpu()
            try:
                self.sess.run(infeed_data.initializer)
            except ValueError:
                pass

        self.network.summary()

    def _build_evaluate(
        self,
        total_loss,
        total_acc,
        total_k_acc,
        image,
        label,
    ):
        if not getattr(self, "evaluation_network", None):
            self.evaluation_network = self._image_network_.load()
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

    def _build(self, total_loss, total_acc, t, image, label, lr, worker):
        self.network = self._image_network_.load()
        predictions = self.network(image)
        loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(label, predictions)
        )
        acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(label, predictions))

        gradients = tf.gradients(
            loss, self.network.trainable_variables, unconnected_gradients="zero"
        )

        if self.config.workers > 1:
            gradients = [
                self._ipu_.cross_replica_sum(g) / self.config.workers for g in gradients
            ]

        grads_and_vars = zip(gradients, self.network.trainable_variables)

        optimize_op = [
            tf.compat.v1.assign(v, v - self.config.learning_rate * g)
            for g, v in grads_and_vars
        ]

        with tf.control_dependencies(optimize_op):
            return (
                tf.add(total_loss, tf.cast(loss, total_loss.dtype)),
                tf.add(total_acc, tf.cast(acc, total_acc.dtype)),
            )

    def get_optimizer(self, lr=None):
        if lr is None:
            lr = self.config.learning_rate
        if isinstance(self.config.optimizer, str):
            options = {}
            opt = self.config.optimizer
        else:
            options = self.config.optimizer.toDict(evaluate=True).copy()
            opt = options.pop("type")
        return getattr(tf.keras.optimizers, opt)(learning_rate=lr, **options)

    def on_execute(self):
        r = self.record
        for epoch in range(1, int(self.config.epochs) + 1):
            r["epoch"] = epoch
            r["loss"], r["acc"] = self.sess.run(self.train_op)
            if self.config.get("compile_test", False):
                print("Compile test successful.")
                return
            r["acc"] *= 100
            r["images"] = self._image_data_.images_per_epoch("train")
            r["images_total"] = r["images"] * epoch
            r["images_per_second"] = r["images"] / self.record.timing()
            self.evaluation_network.set_weights(self.network.get_weights())
            r["val_loss"], r["val_acc"], r["val_k_acc"] = self.sess.run(
                self.eval_op["validation"]
            )
            r["val_acc"] *= 100
            r["val_k_acc"] *= 100
            r.save(echo=True)

        self.test()

    def test(self):
        if not hasattr(self, "eval_op"):
            return float("nan"), float("nan"), float("nan")

        l, a, ka = self.sess.run(self.eval_op["test"])
        results = {"test_loss": l, "test_acc": a * 100, "test_k_acc": ka * 100}
        self.storage.save_data("test_evaluation.json", results)
        self.log.info("Test evaluation: " + str(results))
        return l, a, ka

    def _lr_schedule(self, t):
        _t = tf.cast(t, dtype=tf.float32)

        def warmup():
            if self.config.learning_rate_warmup.mode == "linear":
                factor = 1.0 / (
                    float(self.config.learning_rate_warmup.steps) * _t + 1e-7
                )
            else:
                factor = (
                    tf.math.exp(
                        tf.math.log(2.0)
                        / (self.config.learning_rate_warmup.steps * _t + 1e-7)
                    )
                    - 1
                )
            return factor * float(self.config.learning_rate)

        def schedule():
            if not self.config.learning_rate_schedule.type:
                # constant, no decay
                return float(self.config.learning_rate)

            global_step = tf.math.floor(_t / self._image_data_.steps_per_epoch())
            lr = tf.constant(self.config.learning_rate, shape=(), dtype=tf.float32)

            if self.config.learning_rate_schedule.type == "piecewise":
                return tf.compat.v1.train.piecewise_constant_decay(
                    global_step,
                    [
                        x * self.config.epochs
                        for x in self.config.learning_rate_schedule.boundaries
                    ],
                    [x * lr for x in self.config.learning_rate_schedule.multipliers],
                )

            if self.config.learning_rate_schedule.type == "exponential":
                return tf.compat.v1.train.exponential_decay(
                    lr,
                    global_step,
                    self.config.learning_rate_schedule.decay_steps * self.config.epochs,
                    self.config.learning_rate_schedule.decay_rate,
                    staircase=False,
                )

            if self.config.learning_rate_schedule.type == "cosine_decay":
                return tf.compat.v1.train.cosine_decay(
                    lr,
                    global_step,
                    self.config.learning_rate_schedule.cosine_decay_steps
                    * self.config.epochs,
                    alpha=self.config.learning_rate_schedule.decay_alpha,
                )

            raise ValueError(
                f"Invalid learning rate schedule: '{self.config.learning_rate_schedule.type}'"
            )

        if self.config.learning_rate_warmup.steps > 0:
            return tf.cond(
                tf.math.less_equal(t, self.config.learning_rate_warmup.steps),
                warmup,
                schedule,
            )
        else:
            return schedule()

    def prepare_data(
        self,
        dataset,
        length=None,
        feed_name="infeed",
        infeed=True,
        replication_factor=None,
    ):
        if replication_factor is None:
            replication_factor = self.config.workers

        if isinstance(dataset, str):
            length = self._image_data_.steps_per_epoch(dataset)
            dataset = self.data[dataset]

        steps = tf.data.Dataset.range(0, self.config.epochs * length + 1)
        # repeat so that each replica ends up with the same t
        steps = steps.flat_map(
            lambda x: tf.data.Dataset.from_tensors(x).repeat(replication_factor)
        ).repeat()
        workers = tf.data.Dataset.range(0, self.config.workers).repeat()
        data = tf.data.Dataset.zip((dataset, steps, workers)).map(
            lambda sample, t, worker: {
                "t": tf.cast(t, dtype=tf.int32),
                "image": tf.cast(sample["image"], tf.float32),
                "label": sample["label"],
                "lr": tf.cast(
                    self._lr_schedule(tf.cast(t, dtype=tf.int32)), tf.float32
                ),
                "worker": tf.cast(worker, dtype=tf.int32),
            }
        )

        if self.config.ipu.enabled:
            if infeed:
                try:
                    from tensorflow.python.ipu import ipu_infeed_queue

                    return ipu_infeed_queue.IPUInfeedQueue(
                        data,
                        feed_name=feed_name + str(self.flags.get("SEED")),
                        replication_factor=replication_factor,
                    )
                except ImportError:
                    raise ImportError(
                        "Could not establish IPU-support. Set ipu.enabled=False to run on CPU."
                    )
            else:
                return tf.compat.v1.data.make_one_shot_iterator(data)
        else:
            return tf.compat.v1.data.make_one_shot_iterator(data)

    def get_weights(self, flat=False, network=None):
        if network is None:
            network = self.network

        weights = network.get_weights()

        if flat is True:
            return np.concatenate([v.flatten() for v in weights], axis=0)

        return weights

    def set_weights(self, params, flat=False, network=None):
        if network is None:
            network = self.network

        if flat is True:
            shapes = [(w.shape, np.prod(w.shape)) for w in network.get_weights()]

            offset = 0
            a = []
            for (shape, w) in zip(shapes, params):
                a.append(np.reshape(params[offset : offset + shape[1]], shape[0]))
                offset += shape[1]

            params = a

        network.set_weights(params)

        return True

    def set_params(self, params, flat=False, network=None):
        return self.set_weights(params, flat, network)

    def get_params(self, flat=False, network=None):
        return self.get_weights(flat, network)

    def config_base_learning_rate(self, base_learning_rate):
        if isinstance(base_learning_rate, str):
            eval("base_learning_rate = " + base_learning_rate)
        return base_learning_rate * self.config.data.batch_size * self.config.workers

    def _apply_layer_op(self, variable, layer_id, weight_id, trainable):
        return variable

    def _apply_layer_ops(this, layers=None):
        this.layer_register = []
        this.layer_weight_counter = 0
        this.layer_trainable_weight_counter = 0
        this._current_weight_id = 0
        this._current_trainable_weight_id = 0

        # Modified Keras.Layer.add_weight that will be monkey patched
        def add_weight(
            self,
            name,
            shape,
            dtype=None,
            initializer=None,
            regularizer=None,
            trainable=None,
            **kwargs,
        ):

            this._current_weight_id += 1
            if trainable:
                this._current_trainable_weight_id += 1

            try:
                dtype = this.dtype_policy.variable_dtype
            except AttributeError:
                dtype = tf.float32

            variable = KerasLayerBase.parent_add_weight(
                self, name, shape, dtype, initializer, regularizer, trainable, **kwargs
            )

            # register layer
            if self not in this.layer_register:
                this.layer_register.append(self)
                this.layer_weight_counter = 0
                this.layer_trainable_weight_counter = 0
            else:
                this.layer_weight_counter += 1
                if trainable:
                    this.layer_trainable_weight_counter += 1

            var = this._apply_layer_op(
                variable,
                len(this.layer_register) - 1,
                this.layer_weight_counter,
                trainable,
            )
            # return var

            try:
                cdtype = this.dtype_policy.compute_dtype
            except AttributeError:
                cdtype = tf.float32

            return tf.cast(var, dtype=cdtype)

        if layers is None:
            # global monkey patch
            tf.keras.layers.Layer.parent_add_weight = tf.keras.layers.Layer.add_weight
            tf.keras.layers.Layer.add_weight = add_weight
        else:
            for layer in layers:
                # monkey-patch layer instance
                layer.parent_add_weight = layer.add_weight
                layer.add_weight = add_weight.__get__(layer, layer.__class__)

    def _rollback_layer_ops(self):
        if not hasattr(tf.keras.layers.Layer, "parent_add_weight"):
            return

        tf.keras.layers.Layer.add_weight = tf.keras.layers.Layer.parent_add_weight
        delattr(tf.keras.layers.Layer, "parent_add_weight")

    def on_destroy(self):
        self._rollback_layer_ops()
        try:
            self.sess.close()
        except AttributeError:
            pass
        tf.compat.v1.reset_default_graph()
