# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import tensorflow as tf
from tensorflow.python.ops import array_ops

tf.compat.v1.disable_v2_behavior()


def invert_mask(mask):
    return tf.cast(1 - mask, dtype=mask.dtype)


def low_magnitude(ref, fraction, fixed=False, reverse=False):
    """
    This functions first estimates the threshold value such that the given fraction of the tensor has magnitude less
    than the threshold. It returns a mask where 1 indicates that the magnitude of this element is part of the fraction.

    If fraction == 0.0 then the threshold will be the first element of the tensor and the mask will be zeros everywhere
    If fraction == 1.0 then the threshold will be the last element of the tensor and the mask will be ones everywhere

    Args:
    tensor: The tensor that needs to be masked.
    fraction: Percent of elements to be considered low
    fixed: If true, the threshold won't be estimated but fraction will be used as a fixed threshold
    reverse: Reverse order

    Returns:
    threshold: The value of the threshold based on the given tensor
    mask: A tensor of the same size and shape containing 0 or 1 to indicate which of the tensor value falls below
          the threshold, i.e. 1 for values that fall below the calculated threshold
    """
    abs_tensor = tf.math.abs(ref)

    criterion = tf.math.less if reverse is False else tf.math.greater_equal

    if fixed:
        return (
            tf.constant(fraction, tf.float32),
            tf.cast(criterion(abs_tensor, fraction), tf.int32),
        )

    multiplier = (1 - tf.cast(fraction, tf.float32)) if not reverse else fraction
    k = tf.cast(
        tf.math.round(tf.cast(array_ops.size(abs_tensor), tf.float32) * multiplier),
        tf.int32,
    )
    # Sort the entire array
    values, _ = tf.nn.top_k(
        array_ops.reshape(abs_tensor, [-1]), k=array_ops.size(abs_tensor)
    )
    # Select the (k-1)th value
    threshold = array_ops.gather(values, k - 1)

    if float(fraction) == 1.0:
        return (
            values[-1 if not reverse else 0],
            tf.ones(abs_tensor.shape, dtype=tf.int32),
        )
    if float(fraction) == 0.0:
        return (
            values[0 if not reverse else -1],
            tf.zeros(abs_tensor.shape, dtype=tf.int32),
        )

    mask = tf.cast(criterion(abs_tensor, threshold), tf.int32)

    return threshold, mask


def tensor_update(ref, update):
    indices = tf.expand_dims(
        tf.range(update.get_shape().as_list()[0], dtype=tf.int32), -1
    )
    return tf.tensor_scatter_nd_update(ref, indices, update)


def top_k_mask(tensor, k=1, reverse=False):
    tensor = tf.convert_to_tensor(tensor)
    shape = tensor.get_shape().as_list()
    if len(shape) > 1:
        raise ValueError("Tensor must be rank 1")
    mask = tf.scatter_nd(
        indices=tf.expand_dims(tf.math.top_k(tensor, k=k).indices, axis=1),
        updates=tf.ones([k]),
        shape=[shape[0]],
    )
    if reverse:
        return invert_mask(mask)

    return mask


def masked_update(ref, update, mask=None, assign=False):
    """
    Updates a given tensor with respect to the mask, i.e. only where mask is 1

    For example:
    masked_update(tf.Variable(np.arange(6), dtype=tf.int32), tf.range(6) + 10, tf.constant([0, 1, 0, 1, 1, 0])))
    >>> [0 11 2 13 14 5]

    :param ref:
    :param update:
    :param mask:
    :param assign: If true, result will be assigned to ref
    :return:
    """
    if mask is None:
        if assign:
            return tf.assign(ref, update)
        else:
            return update

    # filter the update using the mask
    filtered_update = tf.math.multiply(update, tf.cast(mask, update.dtype))
    # zero out the values that should be updated
    target = tf.math.multiply(ref, tf.cast(invert_mask(mask), dtype=ref.dtype))
    # add the update
    updated = target + filtered_update

    if not assign:
        return updated

    # re-assign
    return tf.assign(ref, updated)


def elite(ref, elite_fraction):
    _, mask = low_magnitude(ref, elite_fraction, reverse=True)
    return tf.math.multiply(ref, tf.cast(mask, dtype=ref.dtype))


def lowest(ref, lowest_fraction):
    _, mask = low_magnitude(ref, lowest_fraction)
    return tf.math.multiply(ref, tf.cast(mask, dtype=ref.dtype))


def cosine_similarity(a, b):
    normalize_a = tf.nn.l2_normalize(tf.reshape(a, [-1]), 0)
    normalize_b = tf.nn.l2_normalize(tf.reshape(b, [-1]), 0)
    return tf.reduce_sum(tf.multiply(normalize_a, normalize_b))


def gradient_magnitude_similarity(a, b):
    norm_a = tf.norm(a)
    norm_b = tf.norm(b)
    return 2 * norm_a * norm_b / (tf.square(norm_a) + tf.square(norm_b))
