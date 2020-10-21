# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import numpy as np
from machinable import Component

import tensorflow as tf
from mixins.random_base import get_noise
from mixins.surgery import cosine_similarity

tf.compat.v1.disable_v2_behavior()


class OrthoModel(Component):
    """
    Measuring the orthogonality of high-dimensional random bases
    """

    def prepare(self):
        if getattr(self, "sess", False):
            self.sess.close()
            tf.compat.v1.reset_default_graph()

        self._ipu_.get_session()
        self._ipu_.configure()

        outfeed_queue = self._ipu_.outfeed_queue(
            "outfeed" + str(np.random.randint(0, 99999))
        )

        with self._ipu_.device():
            self.op = self._ipu_.compile(
                lambda: self._ipu_.loops_repeat(
                    n=self.config.iterations,
                    body=lambda *args, **kwargs: self._build(*args, **kwargs),
                    inputs=[
                        tf.constant(0, tf.int32),
                        tf.constant(0, tf.float32),
                        tf.constant(0, shape=[self.config.dimension], dtype=tf.float32),
                    ],
                    outfeed_queue=outfeed_queue,
                    divide_by_n=True,
                ),
                [],
            )
            self.outfeed = outfeed_queue.dequeue()

        self.sess.run(tf.global_variables_initializer())

    def _build(self, i, cos_total, v, outfeed_queue=None):
        phi = get_noise(shape=[self.config.dimension], seed=self.flags.SEED, state=i)

        cos = cosine_similarity(phi, v)

        with tf.control_dependencies([outfeed_queue.enqueue({"cos": cos})]):
            return tf.add(i, 1), tf.add(cos_total, cos), phi

    def on_execute(self):
        for d in range(1, 9):
            self.prepare()
            self.config.dimension = 10 ** d
            self.sess.run(self.op)
            outfeed_data = self.sess.run(self.outfeed)
            cos = outfeed_data["cos"][1:]
            self.record["d"] = d
            self.record["dimension"] = self.config.dimension
            self.record["cos_mean"] = np.mean(np.abs(cos))
            self.record["cos_std"] = np.std(np.abs(cos))
            self.record.save(echo=True)
