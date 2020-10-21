# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

from functools import partial

from machinable import Mixin

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.framework.ops import device

tf.compat.v1.disable_v2_behavior()


try:
    from tensorflow.python import ipu
    from tensorflow.python.ipu import ipu_outfeed_queue, ipu_infeed_queue
    from tensorflow.python.ipu.ipu_outfeed_queue import IPUOutfeedMode
    from tensorflow.python.ipu.autoshard import automatic_sharding

    POPLAR_TENSORFLOW = True
except ImportError:
    print("Warning: The installed TensorFlow version does not support IPUs.")
    POPLAR_TENSORFLOW = False


class ArrayOutfeedQueue:
    """
    Helper class that provides an Infeed interface for CPUs
    """

    def __init__(self, size=None):
        self.size = size
        self.queues = None
        self.index = None
        self._dict_map = {}

    def set_size(self, size):
        self.size = size

    def enqueue(self, tensors):
        assert self.size is not None

        tensor_keys = None
        if isinstance(tensors, dict):
            # transform to list but keep keys
            tensor_keys = {v: k for k, v in tensors.items()}
            tensors = [v for v in tensors.values()]

        if not isinstance(tensors, (list, tuple)):
            tensors = [tensors]

        if self.queues is None:
            self.queues = []
            for tensor in tensors:
                var = tf.Variable(
                    lambda: tf.zeros(
                        [self.size] + list(tensor.shape), dtype=tensor.dtype
                    ),
                    dtype=tensor.dtype,
                )
                tf.compat.v1.add_to_collection("outfeed_queue_variables", var)
                self.queues.append(var)
                if tensor_keys:
                    self._dict_map[var.name] = tensor_keys[tensor]

        for i in range(len(self.queues)):
            self.queues[i] = self.queues[i].scatter_nd_update(
                [
                    [
                        self.index,
                    ]
                ],
                [tensors[i]],
            )

        with tf.control_dependencies(self.queues):
            return tf.no_op()

    def dequeue(self):
        if len(self._dict_map) > 0:
            d = {}
            for var in tf.compat.v1.get_collection("outfeed_queue_variables"):
                d[self._dict_map[var.name]] = var
            return d

        return tf.compat.v1.get_collection("outfeed_queue_variables")


class IpuMixin(Mixin):
    """
    Provides a small hardware abstraction layer.
    Every method provides a fall-back to CPU equivalents.
    """

    def device(self, name=None):
        if self.config.ipu.enabled and POPLAR_TENSORFLOW:
            if name is None:
                name = "/device:IPU:0"
            return ipu.scopes.ipu_scope(name)
        else:
            if name is None:
                if (
                    len(
                        [
                            x
                            for x in device_lib.list_local_devices()
                            if x.device_type == "GPU"
                        ]
                    )
                    > 0
                ):
                    name = "/device:GPU:0"
                else:
                    name = "cpu"

            return device(name)

    def outfeed_queue(
        self, feed_name="outfeed", replication_factor=1, outfeed_mode="ALL"
    ):
        if self.config.ipu.enabled and POPLAR_TENSORFLOW:
            if isinstance(outfeed_mode, str):
                outfeed_mode = getattr(IPUOutfeedMode, outfeed_mode)
            return ipu_outfeed_queue.IPUOutfeedQueue(
                feed_name=feed_name,
                outfeed_mode=outfeed_mode,
                replication_factor=replication_factor,
            )
        else:
            return ArrayOutfeedQueue()

    def infeed_queue(self, dataset, feed_name="infeed", replication_factor=1):
        if self.config.ipu.enabled and POPLAR_TENSORFLOW:
            return ipu_infeed_queue.IPUInfeedQueue(
                dataset, feed_name=feed_name, replication_factor=replication_factor
            )
        else:
            return dataset.make_one_shot_iterator()

    def loops_repeat(
        self,
        n,
        body,
        inputs,
        infeed_queue=None,
        outfeed_queue=None,
        divide_by_n=False,
        mode=None,
    ):
        if outfeed_queue is not None:
            body = partial(body, outfeed_queue=outfeed_queue)
            if isinstance(outfeed_queue, ArrayOutfeedQueue):
                outfeed_queue.set_size(n)

        if mode is None and self.config.ipu.enabled and POPLAR_TENSORFLOW:
            repeat_output = ipu.loops.repeat(n, body, inputs, infeed_queue)
        else:
            # device agnostic equivalent
            def iterate(i, *args):
                if outfeed_queue is not None:
                    outfeed_queue.index = i

                if infeed_queue is None:
                    sample = {}
                else:
                    try:
                        sample = infeed_queue.get_next()
                    except AttributeError:
                        sample = infeed_queue

                if not isinstance(sample, (list, tuple, dict)):
                    sample = [sample]

                if isinstance(sample, (list, tuple)):
                    arqs = list(args) + list(sample)
                    out = body(*arqs)
                else:
                    out = body(*args, **sample)

                if isinstance(out, (list, tuple)):
                    ops = [
                        o for o in out if not isinstance(o, (tf.Variable, tf.Tensor))
                    ]
                    if len(ops) > 0:
                        with tf.control_dependencies(ops):
                            return (tf.add(i, 1),) + tuple(
                                o
                                for o in out
                                if isinstance(o, (tf.Variable, tf.Tensor))
                            )

                    return (tf.add(i, 1),) + tuple(out)
                else:
                    if not isinstance(out, tf.Tensor):
                        with tf.control_dependencies([out]):
                            return tf.add(i, 1)

                    return tf.add(i, 1), out

            repeat_output = tf.while_loop(
                cond=lambda i, *args, **kwargs: tf.less(i, n),
                body=iterate,
                loop_vars=[tf.constant(0, dtype=tf.int32)] + list(inputs),
                parallel_iterations=1,
            )

            if isinstance(repeat_output, tuple):
                # remove loop index
                repeat_output = repeat_output[1:]

        if divide_by_n:
            if isinstance(repeat_output, (list, tuple)):
                return tuple(
                    map(
                        lambda x: tf.math.divide(tf.cast(x, dtype=tf.float32), n),
                        repeat_output,
                    )
                )
            else:
                return repeat_output / n
        else:
            return repeat_output

    def compile(self, computation, inputs=None):
        if self.config.ipu.enabled and POPLAR_TENSORFLOW:
            return ipu.ipu_compiler.compile(computation, inputs)
        if inputs is None:
            inputs = []

        if self.config.ipu.xla_on_non_ipu_device:
            return tf.xla.experimental.compile(computation, inputs)

        return computation(*inputs)

    def configure(self, num_ipus=None):
        if self.config.ipu.enabled and POPLAR_TENSORFLOW:
            if num_ipus is None:
                num_ipus = self.config.ipu.num_ipus
            ipu_options = ipu.utils.create_ipu_config()

            if self.config.ipu.device == -1:
                ipu_options = ipu.utils.auto_select_ipus(ipu_options, num_ipus)
            else:
                ipu_options = ipu.utils.select_ipus(ipu_options, self.config.ipu.device)

            if self.config.ipu.compilation_options:
                ipu_options = ipu.utils.set_compilation_options(
                    ipu_options, self.config.ipu.compilation_options.toDict()
                )

            if self.config.ipu.connectionType is not None:
                ipu_options = ipu.utils.set_ipu_connection_type(
                    ipu_options,
                    getattr(
                        ipu.utils.DeviceConnectionType, self.config.ipu.connectionType
                    ),
                    self.config.ipu.connectionVersion,
                )

            ipu.utils.configure_ipu_system(ipu_options)

    def move_variable_initialization_to_cpu(self):
        if self.config.ipu.enabled and POPLAR_TENSORFLOW:
            return ipu.utils.move_variable_initialization_to_cpu()

    def cross_replica_sum(self, x, name=None):
        if self.config.ipu.enabled:
            return ipu.ops.cross_replica_ops.cross_replica_sum(x, name=name)

        return tf.identity(x)

    def auto_shard(
        self, num_shards, input_ts, loss_ts, edge_filter=None, frozen_inference=False
    ):
        if self.config.ipu.enabled and POPLAR_TENSORFLOW and num_shards > 1:
            automatic_sharding(
                num_shards, input_ts, loss_ts, edge_filter, frozen_inference
            )

    def get_session(self, allow_soft_placement=True, **kwargs):
        config = tf.compat.v1.ConfigProto(
            allow_soft_placement=allow_soft_placement, **kwargs
        )
        self.sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(self.sess)
