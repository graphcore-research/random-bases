# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import numpy as np

import tensorflow as tf
from mixins.surgery import low_magnitude


def test_mixins_surgery(ipu):
    if not ipu:
        return

    compile = ipu.ipu_compiler.compile

    ipu_options = ipu.utils.create_ipu_config()
    ipu_options = ipu.utils.auto_select_ipus(ipu_options, 1)
    ipu.utils.configure_ipu_system(ipu_options)
    ipu.utils.move_variable_initialization_to_cpu()

    with ipu.scopes.ipu_scope("/device:IPU:0"):

        def _e(f):
            out = sess.run(compile(f))
            return out

        def _eq(f, value):
            return np.testing.assert_equal(_e(f), value)

        def _m(a):
            return np.array(a, np.int32)

        def test_low_magnitude():
            t1 = [3.0, 1.0, 2.0, 4.0]
            t2 = [-3.0, -1.0, 2.0, 4.0]
            for t in [t1, t2]:
                _eq(lambda: low_magnitude(t, 0.25)[1], _m([[0, 1, 0, 0]]))
                _eq(lambda: low_magnitude(t, 0.5)[1], _m([[0, 1, 1, 0]]))
                _eq(lambda: low_magnitude(t, 0.0)[1], _m([[0, 0, 0, 0]]))
                _eq(lambda: low_magnitude(t, 1.0)[1], _m([[1, 1, 1, 1]]))
                # reverse
                _eq(lambda: low_magnitude(t, 0.0, reverse=True)[1], _m([[0, 0, 0, 0]]))
                _eq(lambda: low_magnitude(t, 1.0, reverse=True)[1], _m([[1, 1, 1, 1]]))
                _eq(lambda: low_magnitude(t, 0.25, reverse=True)[1], _m([[0, 0, 0, 1]]))
                _eq(lambda: low_magnitude(t, 0.5, reverse=True)[1], _m([[1, 0, 0, 1]]))

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            test_low_magnitude()


if __name__ == "__main__":
    test_mixins_surgery(True)
