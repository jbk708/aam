import tensorflow as tf


class AttentionPooling(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionPooling, self).__init__()
        self.query = tf.keras.layers.Dense(1, use_bias=False)

    def call(self, inputs, mask=None):
        attention_scores = self.query(inputs)

        dk = tf.cast(tf.shape(inputs)[-1], dtype=tf.float32)
        scale = 1/ tf.sqrt(dk)
        attention_scores = attention_scores * scale

        if mask is not None:
            mask = tf.cast(mask, dtype=tf.float32)
            attention_scores = attention_scores + (1.0 - mask * -1e9)

        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        pooled_output = tf.reduce_sum(inputs * attention_weights, axis=1)
        return pooled_output
