# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import numpy as np
from machinable import Mixin

import tensorflow as tf
from tensorflow.python.framework import function
from tensorflow.python.util import nest

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
        # we use bernoulli - p for zero mean
        u = tf.random.stateless_uniform(
            shape, minval=0.0, maxval=1.0, seed=stacked_seed, dtype=tf.float32
        )
        return sign * tf.cast(
            tf.cast(tf.math.greater(u, 1 - p), tf.float32) * 2 - 1 - p, dtype=dtype
        )

    if dist == "uniform":
        # zero mean uniform
        u = tf.random.stateless_uniform(
            shape, minval=-1.0, maxval=1.0, seed=stacked_seed, dtype=dtype
        )
        return sign * tf.cast(u, dtype=dtype)

    raise ValueError(f"Invalid distribution {dist}")


def noise_matrix(seeds, shape, state, m=None, dist="normal", norm=False, dtype=None):
    if m is None:
        m = seeds.get_shape().as_list()[0]
    if dtype is None:
        dtype = tf.float32
    ta_seeds = tf.TensorArray(
        seeds.dtype, m, element_shape=[], name="ta_seeds"
    ).unstack(seeds)
    ta = tf.TensorArray(dtype=dtype, size=m, element_shape=shape, name="ta")

    def row(i, matrix):
        seed = ta_seeds.read(i)
        epsilon = get_noise(shape, seed=seed, state=state, dist=dist, dtype=dtype)
        if norm is False:
            norm_factor = 1.0
        elif norm is True:
            norm_factor = tf.norm(epsilon)
        else:
            norm_factor = norm
        return (
            tf.add(i, 1),
            matrix.write(i, epsilon / norm_factor, name="ta_noise_matrix"),
        )

    _, ta_matrix = tf.while_loop(
        cond=lambda i, _: i < m,
        body=row,
        loop_vars=(tf.constant(0, shape=[], dtype=tf.int32), ta),
        parallel_iterations=1,
        maximum_iterations=m,
        back_prop=False,
    )

    return tf.reshape(
        ta_matrix.stack(),
        [
            m,
        ]
        + shape,
    )


def noise_product(
    coordinates,
    seeds,
    state,
    shape,
    batch,
    dist="normal",
    norm=False,
    mode="default",
    elementwise=False,
    elementwise_reduce=True,
):
    """
    TensorFlow custom op that re-generates noise in the backwardpass to save memory
    """
    if mode == "zero_order":
        coordinates = tf.one_hot(
            0, coordinates.get_shape().as_list()[0], dtype=coordinates.dtype
        )

    sig = [coordinates, seeds]
    num = seeds.get_shape().as_list()[0]

    if batch is None:
        num_batches = None
    else:
        if num <= batch:
            batch = None
            num_batches = None
        else:
            if not num % batch == 0:
                raise ValueError(
                    f"Batch has to be multiple of basis but {num} % {batch} != 0"
                )
            num_batches = num // batch
            lengths = tf.constant([batch] * num_batches, dtype=tf.int32)

    def backward(op, grad):
        if num_batches is None:
            E = tf.transpose(
                noise_matrix(
                    seeds=op.inputs[1],
                    shape=[np.prod(shape)],
                    state=state,
                    m=num,
                    dist=dist,
                    norm=norm,
                    dtype=coordinates.dtype,
                )
            )
            flat_grad = tf.reshape(grad, [1, np.prod(grad.shape)])
            grad_coordinates = tf.squeeze(tf.matmul(flat_grad, E))
            return [grad_coordinates, None]
        else:
            # batched iteration
            ta_batch_seeds = tf.TensorArray(
                op.inputs[1].dtype,
                num_batches,
                element_shape=[batch],
                name="ta_batch_seeds",
            ).split(op.inputs[1], lengths=lengths)
            ta_batch_grad = tf.TensorArray(
                grad.dtype, num_batches, element_shape=[batch], name="ta_batch_grad"
            )

            def iteration(i, result):
                E_batch = tf.transpose(
                    noise_matrix(
                        seeds=ta_batch_seeds.read(i),
                        shape=[np.prod(shape)],
                        state=state,
                        m=batch,
                        dist=dist,
                        norm=norm,
                        dtype=coordinates.dtype,
                    )
                )
                flat_grad = tf.reshape(grad, [1, np.prod(grad.shape)])
                p_batch = tf.squeeze(tf.matmul(flat_grad, E_batch))
                return (
                    tf.add(i, 1),
                    result.write(i, p_batch, name="ta_noise_product_grad"),
                )

            _, p_array = tf.while_loop(
                cond=lambda i, _: i < num_batches,
                body=iteration,
                loop_vars=(tf.constant(0, shape=[], dtype=tf.int32), ta_batch_grad),
                parallel_iterations=1,
                back_prop=False,
            )

            return [p_array.concat(), None, None]

    def GetOpDtypes(struct):
        """Returns all tensors' data types in a list."""
        return [x.dtype for x in nest.flatten(struct)]

    @function.Defun(*GetOpDtypes(sig), python_grad_func=backward)
    def op(coordinates_, seeds_):
        if num_batches is None:
            # p = Noise[MxN] . [coordinates (N)]
            E = tf.transpose(
                noise_matrix(
                    seeds=seeds_,
                    shape=[np.prod(shape)],
                    state=state,
                    m=num,
                    dist=dist,
                    norm=norm,
                )
            )
            if mode.startswith("second_order"):
                E = tf.math.square(E) - 1

            if elementwise is True:
                A = tf.linalg.tensor_diag(coordinates_)
                p = tf.transpose(tf.matmul(E, A))
            else:
                A = tf.expand_dims(coordinates_, -1)
                p = tf.reshape(tf.matmul(E, A), shape)

            if norm is False:
                p = p / num

            if mode.endswith("_abs"):
                p = tf.abs(p)
            elif mode.endswith("_square"):
                p = tf.square(p)

            if elementwise and elementwise_reduce:
                p = tf.reduce_sum(p, axis=-1)

            return p

        # batched iteration
        ta_batch_coord = tf.TensorArray(
            coordinates_.dtype,
            num_batches,
            element_shape=[batch],
            name="ta_batch_coord",
        ).split(coordinates_, lengths=lengths)
        ta_batch_seeds = tf.TensorArray(
            seeds_.dtype, num_batches, element_shape=[batch], name="ta_batch_seeds"
        ).split(seeds_, lengths=lengths)

        if elementwise:
            if elementwise_reduce:
                element_shape = [batch]
            else:
                element_shape = [batch, np.prod(shape)]
            ta_batch_el = tf.TensorArray(
                ta_batch_coord.dtype,
                num_batches,
                element_shape=element_shape,
                name="ta_batch_grad",
            )

        def iteration(i, result):
            E_batch = tf.transpose(
                noise_matrix(
                    seeds=ta_batch_seeds.read(i),
                    shape=[np.prod(shape)],
                    state=state,
                    m=batch,
                    dist=dist,
                    norm=norm,
                    dtype=ta_batch_coord.dtype,
                )
            )

            if mode.startswith("second_order"):
                E_batch = tf.math.square(E_batch) - 1

            if elementwise:
                A = tf.linalg.tensor_diag(ta_batch_coord.read(i))
                p_batch = tf.transpose(tf.matmul(E_batch, A))
            else:
                A = tf.expand_dims(ta_batch_coord.read(i), -1)
                p_batch = tf.reshape(tf.matmul(E_batch, A), shape)

            if mode.endswith("_abs"):
                p_batch = tf.abs(p_batch)
            elif mode.endswith("_square"):
                p_batch = tf.square(p_batch)

            if elementwise and elementwise_reduce:
                p_batch = tf.reduce_sum(p_batch, axis=-1)

            if elementwise:
                return (
                    tf.add(i, 1),
                    result.write(i, p_batch, name="ta_noise_product_elementwise"),
                )
            else:
                return tf.add(i, 1), tf.add(result, p_batch)

        if elementwise:
            loop_vars = [tf.constant(0, shape=[], dtype=tf.int32), ta_batch_el]
        else:
            loop_vars = [
                tf.constant(0, shape=[], dtype=tf.int32),
                tf.constant(0, shape=shape, dtype=ta_batch_coord.dtype),
            ]

        _, p_sum = tf.while_loop(
            cond=lambda i, _: i < num_batches,
            body=iteration,
            loop_vars=loop_vars,
            parallel_iterations=1,
            back_prop=False,
        )

        if elementwise:
            if norm is False:
                return p_sum.concat() / num

            return p_sum.concat()
        else:
            if norm is False:
                return p_sum / num

            return p_sum

    y = op(coordinates, seeds)

    if elementwise:
        if elementwise_reduce:
            y.set_shape([num])
        else:
            y.set_shape([num] + list(shape))
    else:
        y.set_shape(shape)

    return y


class RandomBase(Mixin):
    def product(
        self,
        coordinates,
        seeds,
        state,
        shape,
        norm=None,
        mode="default",
        elementwise=False,
        elementwise_reduce=True,
    ):
        """
        Multiplies given coordinate vector with random projection generated on-the-fly from given seeds.

        :param coordinates: Coordinate tensor
        :param seeds: Seeds tensor
        :param state: Integer random state
        :param shape: Shape of the projection, i.e. length of the basis vectors
        :param norm: Normalization of the generated vectors
        :param mode: Supports three modes:
        - loop: basic implementation using tf.while_loop
        - matrix: more performant matrix projection
        - optimized: Recommended choice as it re-generates noise in backwards-pass to be more efficient.
        :param elementwise: If True, multiplication will be Hadamard
        :param elementwise_reduce: If False, multiplication result will not be reduced
        """
        if norm is None:
            norm = self.config.base.normalized

        if self.config.base.mode == "matrix":
            flat_shape = [np.prod(shape)]
            E = tf.transpose(
                noise_matrix(
                    seeds=seeds,
                    shape=flat_shape,
                    state=state,
                    dist=self.config.base.distribution,
                    norm=norm,
                )
            )
            n = seeds.get_shape().as_list()[0]
            p = tf.reshape(tf.matmul(E, tf.expand_dims(coordinates, -1)), shape)
            if norm is False:
                p = p / n
            return p
        elif self.config.base.mode == "optimized":
            return noise_product(
                coordinates,
                seeds,
                state,
                shape,
                batch=self.config.base.batching,
                dist=self.config.base.distribution,
                norm=norm,
                mode=mode,
                elementwise=elementwise,
                elementwise_reduce=elementwise_reduce,
            )
        elif self.config.base.mode == "loop":
            ta_seeds = tf.TensorArray(
                seeds.dtype, seeds.get_shape()[0], element_shape=[]
            ).unstack(seeds)
            ta_coord = tf.TensorArray(
                coordinates.dtype, coordinates.get_shape()[0], element_shape=[]
            ).unstack(coordinates)
            n = seeds.get_shape().as_list()[0]

            def loop_body(i, summation):
                epsilon = get_noise(
                    shape,
                    seed=ta_seeds.read(i),
                    state=state,
                    dist=self.config.base.distribution,
                )
                if norm is False:
                    norm_factor = 1.0
                elif norm is True:
                    norm_factor = tf.norm(epsilon)
                else:
                    norm_factor = norm
                summand = tf.scalar_mul(ta_coord.read(i), epsilon / norm_factor)
                return tf.add(i, 1), tf.add(summation, summand)

            _, result = tf.while_loop(
                cond=lambda i, _: i < n,
                body=loop_body,
                loop_vars=(
                    tf.constant(0, dtype=tf.int32),
                    tf.constant(0.0, shape=shape, dtype=tf.float32),
                ),
                parallel_iterations=1,
                maximum_iterations=n,
                back_prop=True,
            )

            if norm is False:
                result = result / n

            return result
        else:
            raise ValueError("Invalid noise product mode")
