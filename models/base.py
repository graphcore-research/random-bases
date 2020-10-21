# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import numpy as np
from machinable import Component

import tensorflow as tf

tf.compat.v1.disable_v2_behavior()


class BaseModel(Component):
    def config_auto_norm(self, k=0):
        # mean of chi distribution for network size, approximately sqrt(k)
        return np.sqrt(k)

    def config_2e(self, e):
        return 2 ** float(e)

    def base_seeds_generator(self, seed, state=None, dimensions=None):
        if state is None:
            state = self.flags.SEED
        if dimensions is None:
            dimensions = self.config.base_dimensions

        if isinstance(state, int):
            state = tf.constant(state, tf.int32)

        if self.config.antithetic_sampling:
            assert (
                dimensions % 2 == 0
            ), "Antithetic sampling requires even base dimension"

            seeds = tf.random.stateless_uniform(
                shape=[dimensions // 2],
                seed=tf.stack([tf.cast(state, tf.int32), tf.cast(seed, tf.int32)]),
                maxval=2 ** 30,
                dtype=tf.int32,
            )

            return tf.concat([seeds, -seeds], axis=0)

        # stateless so that workers end up with the same seeds
        return tf.random.stateless_uniform(
            shape=[dimensions],
            seed=tf.stack([tf.cast(state, tf.int32), tf.cast(seed, tf.int32)]),
            maxval=2 ** 30,
            dtype=tf.int32,
        )
