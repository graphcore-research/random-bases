# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import json
import math
import random
from collections import deque

import numpy as np

import tensorflow as tf
from mixins import surgery
from mixins.random_base import noise_matrix
from tensorflow.keras import backend as K
from vendor.img.models.image import ImageModel

from .base import BaseModel

tf.compat.v1.disable_v2_behavior()


class RbdModel(ImageModel, BaseModel):
    def on_before_create(self):
        # validate config
        if self.config.keep_projection_in_memory:
            assert self.config.base_learning.enabled is False
        if self.config.base_learning.enabled:
            assert (
                self.config.reset_coordinates_each_step > 1
            ), "base_learning must be enabled"

        self.coordinate_history = deque()
        self.gradient_correlation_history = deque()
        self.hessian_history = deque()
        self.found_correlations = deque()
        self._weights_per_compartment_counter = 0
        self._weight_index_map = {}
        self._weights = []
        self.coordinates = []
        self.h_norms = []
        self.bases = []
        self.previous_weights = None
        self.outfeed_data = {}
        self.t = None
        self.worker = None

        self.total_number_of_weights = {
            "base_conv-cifar10": 122570,
            "base_conv-mnist": 93322,
            "base_conv-fashion_mnist": 93322,
            "dense-cifar10": 394634,
            "dense-mnist": 101770,
            "dense-fashion_mnist": 101770,
        }[self.config.network + "-" + self.config.data.dataset.name]

        self._image_data_.load()

    def on_create(self):
        self._ipu_.get_session()
        self._ipu_.configure(num_ipus=self.config.workers * (self.shards or 1))

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

        outfeed_queue = self._ipu_.outfeed_queue(
            "outfeed" + str(self.flags.SEED), self.config.workers
        )

        if self.config.workers > 1 and self.config.same_images_for_each_worker:
            dataset = self.data["train"].flat_map(
                lambda x: tf.data.Dataset.from_tensors(x).repeat(self.config.workers)
            )
        else:
            dataset = self.data["train"]

        train_feed = self.prepare_data(
            dataset, self._image_data_.steps_per_epoch("train")
        )

        with self._ipu_.device():
            self.train_op = self._ipu_.compile(
                lambda: self._ipu_.loops_repeat(
                    n=self.n,
                    body=lambda *args, **kwargs: self._build_optimize(*args, **kwargs),
                    inputs=[
                        tf.constant(0, tf.float32),
                        tf.constant(0, tf.float32),
                    ],
                    infeed_queue=train_feed,
                    outfeed_queue=outfeed_queue,
                    divide_by_n=True,
                )
            )
            self._ipu_.move_variable_initialization_to_cpu()
            try:
                self.sess.run(train_feed.initializer)
            except (ValueError, AttributeError):
                pass
            self.outfeed = outfeed_queue.dequeue()

        self._rollback_layer_ops()

        self.summary()

        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.evaluation_network.set_weights(self.network.get_weights())

        self.sess.graph.finalize()

    def _build_evaluate(
        self, total_loss, total_acc, total_k_acc, image, label, id=None
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

    @property
    def shards(self):
        if self.config.use_sharding is False:
            return None

        if int(self.config.use_sharding) > 1:
            return int(self.config.use_sharding)

        raise ValueError("Please provide the number of shards.")

    def _build_optimize(
        self,
        total_loss,
        total_acc,
        t,
        image,
        label,
        lr,
        worker,
        outfeed_queue=None,
        id=None,
    ):
        self.t = t
        self.worker = worker

        self.network = self._image_network_.load(
            name=self.config.network,
            classes=self.dataset_info.features["label"].num_classes,
            input_shape=self.dataset_info.features["image"].shape,
            inputs=image,
        )

        predictions = self.network(image)

        # compute gradients
        loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(label, predictions)
        )
        acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(label, predictions))
        gradients = tf.gradients(loss, self.coordinates)

        if self.config.workers > 1 and self.config.average_in_coordinate_space:
            gradients = [self._ipu_.cross_replica_sum(g) for g in gradients]

        if self.shards is not None:
            # apply auto-sharding
            self._ipu_.auto_shard(self.shards, image, loss)

        if self.config.retrieve_coordinates:
            for i, g in enumerate(gradients):
                self.outfeed_data["coordinates/" + str(i)] = g
        else:
            self.outfeed_data["coordinates/0"] = tf.constant(0)

        update_op = self._build_apply_gradients(lr, gradients, t, loss, image, label)

        with tf.control_dependencies([update_op]):
            with tf.control_dependencies([outfeed_queue.enqueue(self.outfeed_data)]):
                return tf.add(total_loss, loss), tf.add(total_acc, acc)

    def _build_apply_gradients(
        self,
        lr,
        coordinate_gradient,
        t,
        loss=None,
        image=None,
        label=None,
    ):
        # coordinate transformation
        if self.config.coordinate_transformation == "norm":
            for c in range(len(coordinate_gradient)):
                zero = coordinate_gradient[c] - (
                    tf.ones(self.coordinates[c].get_shape().as_list()[0])
                    * tf.reduce_min(coordinate_gradient[c])
                )
                norm = tf.divide(zero, tf.reduce_max(zero))
                coordinate_gradient[c] = norm - 0.5
        elif self.config.coordinate_transformation == "ranks":
            for c in range(len(coordinate_gradient)):
                argsort = tf.argsort(coordinate_gradient[c], direction="ASCENDING")
                ranks = tf.compat.v1.scatter_update(
                    self.coordinates[c],
                    argsort,
                    tf.cast(
                        tf.range(tf.shape(self.coordinates[c])[0]), dtype=tf.float32
                    ),
                )
                coordinate_gradient[c] = (
                    tf.divide(ranks, tf.cast(tf.shape(ranks)[0] - 1, dtype=tf.float32))
                    - 0.5
                )

        if int(self.config.continuous_coordinate_update) > 1:
            apply_coordinates = [
                tf.compat.v1.assign(var, var - lr * value)
                for var, value in zip(self.coordinates, coordinate_gradient)
            ]

            with tf.control_dependencies(apply_coordinates):
                return tf.cond(
                    tf.equal(
                        tf.math.mod(t, int(self.config.continuous_coordinate_update)),
                        int(self.config.continuous_coordinate_update) - 1,
                    ),
                    # project back with -1 as coordinates encode an update step not a gradient
                    true_fn=lambda: self.project_update(lr=-1, t=t, loss=loss),
                    false_fn=lambda: tf.no_op(),
                )

        apply_coordinates = [
            tf.compat.v1.assign(var, value)
            for var, value in zip(self.coordinates, coordinate_gradient)
        ]

        with tf.control_dependencies(apply_coordinates):
            return self.project_update(lr, t, loss)

    def project_update(self, lr, t, loss=None):
        if self.config.use_top_directions is not False:
            if isinstance(self.config.use_top_directions, int):
                self.coordinates = [
                    surgery.masked_update(
                        c,
                        update=tf.zeros_like(c),
                        mask=surgery.top_k_mask(
                            c,
                            k=abs(self.config.use_top_directions),
                            reverse=self.config.use_top_directions > 0,
                        ),
                    )
                    for c in self.coordinates
                ]
            else:
                if self.config.use_top_directions > 0:
                    self.coordinates = [
                        surgery.elite(c, self.config.use_top_directions)
                        for c in self.coordinates
                    ]
                else:
                    self.coordinates = [
                        surgery.lowest(c, abs(self.config.use_top_directions))
                        for c in self.coordinates
                    ]

        update_steps = []
        correlations = []
        hessians = []
        for (
            layer_id,
            weight_id,
            weight,
            index,
            groups,
            state,
            factor,
            v,
            variable_id,
        ) in self._weights:
            weight_identifier = str(layer_id) + "/" + str(weight_id) + "/" + weight.name

            # hessian
            if self.config.compute_hessian:
                if self.config.compute_full_hessian is False:
                    hessian = self.coordinate_noise_product(
                        index,
                        weight.get_shape().as_list(),
                        groups,
                        state,
                        mode="second_order_square",
                        elementwise=True,
                        elementwise_reduce=True,
                    )
                else:
                    hessian = self.coordinate_noise_product(
                        index,
                        weight.get_shape().as_list(),
                        groups,
                        state,
                        mode="second_order",
                        elementwise=True,
                        elementwise_reduce=False,
                    )
                    self.outfeed_data["hessian/" + weight_identifier] = hessian
                hessians.append(hessian)

            perturbation = self.coordinate_noise_product(
                index,
                weight.get_shape().as_list(),
                groups,
                state,
            )
            gradient = factor * perturbation
            if v is None:
                step = gradient
            else:
                step = tf.compat.v1.assign(
                    v,
                    self.config.momentum * v + (1.0 - self.config.momentum) * gradient,
                )

            sgd = tf.gradients(loss, weight)[0] if loss is not None else None

            if loss is not None and self.config.compute_gradient_correlation:
                corr = surgery.cosine_similarity(sgd, gradient)
                self.outfeed_data["correlation/" + weight_identifier] = corr
                correlations.append(corr)

            if float(self.config.surgeon_update) > 0:
                step = tf.cond(
                    tf.math.greater_equal(corr, self.config.surgeon_update),
                    true_fn=lambda: gradient,
                    false_fn=lambda: tf.zeros_like(gradient),
                )

            if self.config.workers > 1 and not self.config.average_in_coordinate_space:
                step = self._ipu_.cross_replica_sum(step)

            if self.config.weight_streaming:
                self.outfeed_data["updated_weights/" + weight.name] = weight + step

            if self.config.correlation_multiplier:
                step = step * tf.math.maximum(corr, 0)

            update_steps.append(
                (
                    weight,
                    step,
                    sgd,
                    variable_id == abs(self.config.use_sgd_for_layer)
                    if np.sign(int(self.config.use_sgd_for_layer)) > 0
                    else variable_id != abs(self.config.use_sgd_for_layer),
                )
            )

        # norm of hessian contribution for each direction
        if self.config.compute_hessian:
            self.hessian_norm = tf.add_n(hessians)
            self.outfeed_data["h_norm"] = self.hessian_norm

        slr = self.config.sgd_learning_rate

        if self.config.use_sgd:
            return tf.group(
                [
                    tf.compat.v1.assign_add(var, tf.cast(-slr * sgd_step, var.dtype))
                    for var, g, sgd_step, gn in update_steps
                ]
            )

        if self.config.update_schedule.enabled:

            return tf.cond(
                self.update_schedule(t),
                lambda: tf.group(
                    [
                        tf.compat.v1.assign_add(
                            var, tf.cast(-slr * sgd_step, var.dtype)
                        )
                        for var, g, sgd_step, gn in update_steps
                    ]
                ),
                lambda: tf.group(
                    [
                        tf.compat.v1.assign_add(var, tf.cast(-lr * g, var.dtype))
                        for var, g, sgd_step, gn in update_steps
                    ]
                ),
            )

        if self.config.use_sgd_for_layer is not False:
            print(
                "Using SGD for ",
            )
            _var_sgd = 0
            _var_all = 0
            for var, g, sgd_step, gn in update_steps:
                _var_all += np.prod(var.get_shape().as_list())
                if gn:
                    s = np.prod(var.get_shape().as_list())
                    _var_sgd += s
                    print(var, s)
            print(_var_sgd, _var_all, _var_all - _var_sgd)
            return tf.group(
                [
                    tf.compat.v1.assign_add(var, tf.cast(-slr * sgd_step, var.dtype))
                    if gn
                    else tf.compat.v1.assign_add(var, tf.cast(-lr * g, var.dtype))
                    for var, g, sgd_step, gn in update_steps
                ]
            )

        if self.config.skip_update_if_correlation_lower_than is not False:
            corr_boundary = tf.constant(
                float(self.config.skip_update_if_correlation_lower_than),
                correlations[0].dtype,
            )

            use_sgd_step = tf.math.less(tf.math.reduce_min(correlations), corr_boundary)
            self.outfeed_data["used_sgd_step"] = use_sgd_step
            return tf.cond(
                use_sgd_step,
                lambda: tf.group(
                    [
                        tf.compat.v1.assign_add(
                            var, tf.cast(-slr * sgd_step, var.dtype)
                        )
                        for var, g, sgd_step, gn in update_steps
                    ]
                ),
                lambda: tf.group(
                    [
                        tf.compat.v1.assign_add(var, tf.cast(-lr * g, var.dtype))
                        for var, g, sgd_step, gn in update_steps
                    ]
                ),
            )

        return self.get_optimizer(lr).apply_gradients(
            grads_and_vars=[(g, var) for var, g, sgd_g, gn in update_steps]
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

        # each variable has a different random state
        state = layer_id * 1000 + weight_id

        if self.config.weights_per_compartment == 0:
            # no separation
            self.config.weights_per_compartment = 1e12

        weights_count = int(np.prod(weight.get_shape().as_list()))
        self._weights_per_compartment_counter += weights_count

        if self.config.group_weights_by_type:
            # add base for generic type
            if len(self.coordinates) < 1:
                self.add_coordinates()
                self.add_base(len(self.coordinates))

            if weight.name.find("kernel") != -1:
                # add base for kernel
                self.add_coordinates()
                self.add_base(len(self.coordinates))

                index = len(self.coordinates) - 1
            else:
                index = 0
            groups = 1
        elif self.config.dynamic_compartments is not False:

            dimensions = max(
                math.ceil(weights_count / float(self.config.dynamic_compartments)), 10
            )

            if self.config.dynamic_allocation_mode == "exp":
                dimensions = max(
                    math.ceil(
                        weights_count ** 1.2 / float(self.config.dynamic_compartments)
                    ),
                    10,
                )

            if self.config.split_dimensions_across_workers:
                dimensions = max(math.ceil(dimensions / float(self.config.workers)), 2)

            # split layer if approximation dimension gets too large
            if weights_count > self.config.weights_per_compartment:
                groups = math.ceil(weights_count / self.config.weights_per_compartment)
            else:
                groups = 1

            dimensions_per_group = math.ceil(dimensions / float(groups))

            for i in range(groups):
                if dimensions > dimensions_per_group:
                    d = dimensions_per_group
                else:
                    d = dimensions
                d += d % 2

                self.add_coordinates(
                    str(len(self.coordinates)) + "/group" + str(i), dimensions=d
                )
                self.add_base(layer_id + i, dimensions=d)

            index = len(self.coordinates) - groups
        else:
            if self.config.weights_per_compartment < 0:
                # group all weights in the layer
                if (
                    layer_id > len(self.coordinates) - 1
                    and layer_id % abs(self.config.weights_per_compartment) == 0
                    and weight_id == 0
                ):
                    self.add_coordinates(str(layer_id))
                    self.add_base(layer_id)

                groups = 1
                # end group by layer
            else:
                # group weights into constant sized compartments, crossing layer boundaries
                groups = math.ceil(
                    weights_count / float(self.config.weights_per_compartment)
                )

                num_compartments = math.ceil(
                    self.total_number_of_weights / self.config.weights_per_compartment
                )

                if self.config.split_dimensions_across_compartments is not False:
                    d = math.ceil(self.config.base_dimensions / num_compartments)
                else:
                    d = self.config.base_dimensions
                d += d % 2

                if (
                    groups == 1
                    and len(self.coordinates) > 0
                    and self._weights_per_compartment_counter
                    <= self.config.weights_per_compartment
                ):
                    # re-use coordinate of previous layer
                    pass
                else:
                    # add coordinates
                    for i in range(groups):
                        self.add_coordinates(
                            str(len(self.coordinates)) + "/group" + str(i), dimensions=d
                        )
                        self.add_base(layer_id + i, dimensions=d)

                if (
                    self._weights_per_compartment_counter
                    > self.config.weights_per_compartment
                ):
                    self._weights_per_compartment_counter = weights_count
                # end group by modulo

            index = len(self.coordinates) - groups

        factor = 1.0

        # add momentum
        if self.config.momentum > 0:
            v = K.zeros(shape=weight.shape, dtype=weight.dtype)
        else:
            v = None

        # register weight for update
        self._weights.append(
            (
                layer_id,
                weight_id,
                weight,
                index,
                groups,
                state,
                factor,
                v,
                self._current_trainable_weight_id,
            )
        )

        # register index mapping
        if index not in self._weight_index_map:
            self._weight_index_map[index] = []
        self._weight_index_map[index].append(self._weights[-1])

        # on-the-fly noise perturbation
        perturbation = self.coordinate_noise_product(
            index, weight.get_shape().as_list(), groups, state
        )
        return weight + tf.cast(factor * perturbation, dtype=weight.dtype)

    def coordinate_noise_product(
        self,
        index,
        shape,
        groups,
        state=0,
        mode="default",
        elementwise=False,
        elementwise_reduce=True,
        weights=None,
    ):

        if weights is not None:
            coordinates = tf.compat.v1.assign(
                self.coordinates[index], self.coordinates[index] * weights
            )
        else:
            coordinates = self.coordinates[index]

        if groups <= 1 and int(self.config.weights_per_compartment) <= 0:
            if not self.config.keep_projection_in_memory:
                return self._random_base_.product(
                    coordinates=coordinates,
                    seeds=self.bases[index],
                    state=state,
                    shape=shape,
                    mode=mode,
                    elementwise=elementwise,
                    elementwise_reduce=elementwise_reduce,
                )
            else:
                return self.stored_random_base_product(coordinates, index, state, shape)

        size = np.prod(shape)
        weights_per_group = math.ceil(int(size) / float(groups))
        sizes = []
        while size > weights_per_group:
            sizes.append(weights_per_group)
            size -= weights_per_group
        sizes.append(size)

        assert weights is None, "Weighting factors are not supported in this mode"

        if not self.config.keep_projection_in_memory:
            return tf.reshape(
                tf.concat(
                    [
                        self._random_base_.product(
                            coordinates=self.coordinates[index + group],
                            seeds=self.bases[index + group],
                            state=state,
                            shape=[sizes[group]],
                            mode=mode,
                            elementwise=elementwise,
                            elementwise_reduce=elementwise_reduce,
                        )
                        for group in range(groups)
                    ],
                    axis=0,
                ),
                shape,
            )
        else:
            return tf.reshape(
                tf.concat(
                    [
                        self.stored_random_base_product(
                            self.coordinates[index + group],
                            index + group,
                            state,
                            [sizes[group]],
                        )
                        for group in range(groups)
                    ],
                    axis=0,
                ),
                shape,
            )

    def stored_random_base_product(self, coordinates, index, state, shape):
        if state not in self.bases[index]:
            # generate fixed projection matrix
            self.bases[index][state] = tf.transpose(
                noise_matrix(
                    seeds=self.generate_base_seeds(index),
                    shape=[np.prod(shape)],
                    state=state,
                    dist=self.config.base.distribution,
                    norm=self.config.base.normalized,
                )
            )

        p = tf.reshape(
            tf.matmul(self.bases[index][state], tf.expand_dims(coordinates, -1)), shape
        )
        if self.config.base.normalized is False:
            p = p / coordinates.get_shape().as_list()[0]
        return p

    def on_execute(self):
        # restore from checkpoint if existing
        epoch_start = 1
        checkpoint = self.storage.data("checkpoint.json", default=None)
        if checkpoint is not None:
            self.log.info(f"Resuming from checkpoint: {checkpoint}")
            epoch_start = checkpoint["epoch"] + 1
            weights = self.storage.data("checkpoint.npy")
            self.network.set_weights(weights)

        r = self.record
        for epoch in range(epoch_start, int(self.config.epochs) + 1):
            r["epoch"] = epoch
            r["images"] = self.n * self.config.data.batch_size * self.config.workers
            r["loss"], r["acc"] = self.sess.run(self.train_op)
            r["acc"] *= 100
            if self.config.compile_test:
                self.log.info(
                    f"Compiling was successful for batching={self.config.base.batching}, "
                    f"d={self.config.base_dimensions}, wpc={self.config.weights_per_compartment}, "
                    f"dync={self.config.dynamic_compartments}, batch_size={self.config.data.batch_size}"
                )
                return
            outfeed_data = self.sess.run(self.outfeed)

            r["images_total"] = r["images"] * r["epoch"]
            r["images_per_second"] = r["images"] / self.record.timing()
            r["images_per_second_avg"] = r["images_total"] / self.record.timing("total")

            if self.config.compute_full_hessian:
                H = {k: v for k, v in outfeed_data.items() if k.startswith("hessian")}
                if len(H) > 0:
                    self.hessian_history.append(H)
                    self.storage.save_data(
                        "hessians.p", self.hessian_history, overwrite=True
                    )

            if self.config.compute_hessian:
                self.h_norms.append(outfeed_data["h_norm"])
                if self.config.store_hessian:
                    self.storage.save_data(
                        "hessian_norm.p", self.h_norms, overwrite=True
                    )

            if self.config.compute_gradient_correlation:
                corr = {
                    k: v for k, v in outfeed_data.items() if k.startswith("correlation")
                }
                if len(corr) > 0:
                    self.gradient_correlation_history.append(corr)

                    if self.config.store_correlations:
                        self.storage.save_data(
                            "correlations.p",
                            self.gradient_correlation_history,
                            overwrite=True,
                        )

                    # overall
                    cr = []
                    for k, v in corr.items():
                        cr.extend(v)

                    r["corr_mean"] = np.mean(cr)
                    r["corr_std"] = np.std(cr)
                    r["corr_min"] = np.min(cr)
                    r["corr_max"] = np.max(cr)

                if (
                    self.config.skip_update_if_correlation_lower_than is not False
                    and not self.config.use_sgd
                ):
                    r["used_sgd_step"] = np.sum(outfeed_data["used_sgd_step"]) / len(
                        outfeed_data["used_sgd_step"]
                    )

            # coordinates
            coordinates = {
                k: v for k, v in outfeed_data.items() if k.startswith("coordinates")
            }
            all_coordinates = [0] * len(coordinates)
            for k, v in coordinates.items():
                all_coordinates[int(k.split("/")[-1])] = v
            self.coordinate_statistics(all_coordinates)
            if self.config.store_coordinates:
                self.coordinate_history.append(all_coordinates)
                self.storage.save_data(
                    "coordinates.npy",
                    np.asarray(self.coordinate_history),
                    overwrite=True,
                )

            if self.config.weight_streaming:
                if self.config.workers > 1:
                    weights = [
                        outfeed_data["updated_weights/" + w.name][-1][0]
                        for w in self.network.weights
                    ]
                else:
                    weights = [
                        outfeed_data["updated_weights/" + w.name][-1]
                        for w in self.network.weights
                    ]
            else:
                weights = self.network.get_weights()

            if self.config.update_schedule.enabled:
                # write diagnostic checkpoints at switch points
                if (
                    self.config.update_schedule.epochs - 1
                    <= epoch
                    <= self.config.update_schedule.epochs + 1
                ):
                    self.storage.save_data("checkpoint-" + str(epoch) + ".npy", weights)

            if len(weights) == 0:
                r.save(echo=True)
                break

            # keep latest checkpoint
            self.storage.save_data("checkpoint.npy", weights)
            self.storage.save_data("checkpoint.json", {"epoch": epoch})

            self.weight_statistics(weights)
            # validation
            self.evaluation_network.set_weights(weights)
            r["val_loss"], r["val_acc"], r["val_k_acc"] = self.sess.run(
                self.eval_op["validation"]
            )
            r["val_acc"] *= 100
            r["val_k_acc"] *= 100

            if self.config.stop_on_nan:
                if (
                    np.isnan(r["val_loss"])
                    or r["val_loss"] > 1000
                    or (epoch > 4 and r["val_acc"] <= 15)
                ):
                    r.save(echo=True)
                    self.log.info(
                        "Training finished early due to NaNs or non-convergence"
                    )
                    return

            r.save(echo=True)

    def coordinate_statistics(self, coordinates):
        coordinates = np.column_stack(coordinates)
        if np.prod(coordinates.shape) < 10:
            self.record["coordinates"] = coordinates
        self.record["coordinates_mean"] = np.mean(coordinates)
        self.record["coordinates_std"] = np.std(coordinates)
        self.record["coordinates_min"] = np.min(coordinates)
        self.record["coordinates_max"] = np.max(coordinates)

    def weight_statistics(self, weights):
        if len(weights) == 0:
            return
        flattend_weights = np.concatenate([v.flatten() for v in weights], axis=0)
        self.record["weights_mean"] = np.mean(flattend_weights)
        self.record["weights_abs_mean"] = np.mean(np.abs(flattend_weights))
        self.record["weights_std"] = np.std(flattend_weights)
        self.record["update_magnitude"] = 0.0
        if self.previous_weights is not None:
            self.record["update_magnitude"] = np.mean(
                1 - np.abs(flattend_weights / self.previous_weights)
            )
        self.previous_weights = flattend_weights

    def summary(self, network=True):
        if network:
            self.network.summary()

        # display layer-coordinate mapping
        for (
            layer_id,
            weight_id,
            weight,
            index,
            groups,
            state,
            factor,
            _,
            variable_id,
        ) in self._weights:
            print(
                f"{index}: {weight.name} {weight.shape}={np.prod(weight.shape)}: "
                f"d={self.coordinates[index].get_shape().as_list()[0]}, "
                f"groups={groups}, "
                f"variable_id={variable_id}"
            )

        info = {}
        weight_count = np.sum(
            [np.prod(w[2].get_shape().as_list()) for w in self._weights]
        )
        communication = sum([c.get_shape().as_list()[0] for c in self.coordinates])
        info["coordinate_communication"] = communication
        info["weight_count"] = weight_count
        info["communication_per_worker"] = communication
        info["communication_total"] = communication * self.config.workers
        info["layers"] = len(self.layer_register)
        info["registered_weights"] = weight_count
        info["variables"] = len(self._weights)
        info["steps_per_epoch"] = self._image_data_.steps_per_epoch()
        info["num_compartments"] = len(self.coordinates)
        info["compression_ratio"] = weight_count / float(communication)
        info["compression_ratio_percent"] = communication / weight_count * 100

        self.storage.save_data("info.json", info, overwrite=True)
        self.storage.save_data("info.p", info, overwrite=True)
        if len(info) > 0:
            self.log.info(json.dumps(info, indent=4, sort_keys=True, default=str))

        return info

    def add_coordinates(self, name=None, dimensions=None):
        if name is None:
            name = str(len(self.coordinates))
        if dimensions is None:
            dimensions = self.config.base_dimensions
        self.coordinates.append(
            K.zeros(
                shape=[dimensions],
                dtype=tf.float32,
                name="coordinates/" + name,
            )
        )

        if self.config.reset_coordinates_each_step:

            def reset_value(var):
                if self.config.antithetic_sampling:
                    sigma = tf.random.uniform(
                        shape=[var.get_shape().as_list()[0] // 2],
                        minval=0.0,
                        maxval=1.0,
                        dtype=var.dtype,
                    )
                    return tf.concat([-sigma, sigma], axis=0)

                return tf.zeros(
                    shape=var.shape,
                    dtype=var.dtype,
                )

            if int(self.config.reset_coordinates_each_step) > 1:
                mask = tf.math.ceil(
                    tf.cast(
                        tf.math.mod(
                            self.t, int(self.config.reset_coordinates_each_step)
                        ),
                        dtype=self.coordinates[-1].dtype,
                    )
                    / tf.constant(
                        self.config.reset_coordinates_each_step,
                        dtype=self.coordinates[-1].dtype,
                    )
                )
                self.coordinates[-1] = tf.compat.v1.assign(
                    self.coordinates[-1],
                    self.coordinates[-1] * mask,
                )
            else:
                self.coordinates[-1] = tf.compat.v1.assign(
                    self.coordinates[-1], reset_value(self.coordinates[-1])
                )

    def add_base(self, index=0, dimensions=None):
        if dimensions is None:
            dimensions = self.config.base_dimensions
        if not self.config.keep_projection_in_memory:
            self.bases.append(
                tf.Variable(
                    lambda: self.generate_base_seeds(index, dimensions=dimensions)
                )
            )
        else:
            # mark base as yet-to-be-generated
            self.bases.append({})

        if (
            not self.config.reset_base_each_step
            or self.config.keep_projection_in_memory
        ):
            return

        # re-draw every n-th step
        timestep = tf.math.floordiv(self.t, int(self.config.reset_base_each_step))

        if self.config.base_learning.enabled:

            def get_mask():
                if self.config.base_learning.mode.endswith("_magnitude"):
                    threshold, masking = surgery.low_magnitude(
                        self.coordinates[-1],
                        fraction=self.config.base_learning.fraction,
                        reverse=self.config.base_learning.mode == "high_magnitude",
                    )
                    return masking

                raise ValueError("Invalid base_learning mode")

            if int(self.config.reset_coordinates_each_step) > 1:
                mask = tf.cond(
                    tf.equal(
                        tf.math.mod(
                            tf.cast(
                                self.t,
                                dtype=tf.int32,
                            ),
                            int(self.config.reset_coordinates_each_step),
                        ),
                        0,
                    ),
                    get_mask,
                    lambda: tf.zeros([self.config.base_dimensions], dtype=tf.int32),
                )
            else:
                mask = get_mask()

            generated_base = self.generate_base_seeds(index, timestep, self.worker)
            new_base = surgery.masked_update(self.bases[-1], generated_base, mask)
            self.bases[-1] = tf.compat.v1.assign(self.bases[-1], new_base)
            return

        self.bases[-1] = tf.compat.v1.assign(
            self.bases[-1],
            self.generate_base_seeds(
                index,
                timestep,
                self.worker,
                dimensions=self.bases[-1].get_shape().as_list()[0],
            ),
        )

    def randomize(self, state):
        return random.Random(state).randint(0, 2 ** 15)

    def randomize_tensor(self, state):
        return tf.random.stateless_uniform(
            shape=[],
            seed=tf.stack(
                [tf.cast(state, tf.int32), tf.cast(self.flags.get("SEED"), tf.int32)]
            ),
            maxval=2 ** 15,
            dtype=tf.int32,
        )

    def generate_base_seeds(
        self, index: int = 0, t=0, worker=None, split=None, dimensions=None
    ):
        if worker is None:
            worker = self.randomize(self.flags.get("WORKER_ID", 0))
        if dimensions is None:
            dimensions = self.config.base_dimensions

        if not self.config.base_for_each_worker:
            worker = self.randomize(123)

        if split is None:
            split = 0

        index = self.randomize(index)
        t = tf.cast(t, tf.int32)
        worker = tf.cast(worker, tf.int32)
        split = tf.cast(split, tf.int32)

        # make base timestep dependent to reset after each iteration
        if self.config.reset_base_each_step:
            return self.base_seeds_generator(
                index + t + worker + split,
                dimensions=dimensions,
            )
        else:
            return self.base_seeds_generator(
                index + worker + split, dimensions=dimensions
            )

    def get_flattened_shape(self, var):
        if isinstance(var, (list, tuple)):
            return [sum([self.get_flattened_shape(v)[0] for v in var])]

        return [np.prod(var.get_shape().as_list())]

    def get_flattened_tensor(self, dict_like):
        return tf.concat([tf.reshape(v, [-1]) for k, v in dict_like.items()], axis=0)

    @property
    def n(self):
        if self.config.n is not None:
            return self.config.n

        if self.config.compile_test:
            return 1

        return int(
            round(self._image_data_.steps_per_epoch("train") / self.config.workers)
        )
