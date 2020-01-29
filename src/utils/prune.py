import tensorflow as tf

def prune_layer(procent: float, 
                mask: tf.Tensor,
                weights: tf.Tensor) -> tf.Tensor:
    sorted_weights = tf.sort(tf.abs(weights[mask == 1]))

    cutoff_idx = tf.cast(procent * sorted_weights.shape[0], tf.int32)
    cutoff = sorted_weights[cutoff_idx]

    return tf.where(tf.abs(weights) > cutoff, mask, tf.zeros(mask.shape))
