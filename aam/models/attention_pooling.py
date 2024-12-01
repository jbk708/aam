import tensorflow as tf


class AttentionPooling(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionPooling, self).__init__()
        self.query = tf.keras.layers.Dense(8, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, dtype=tf.float32)

    def call(self, inputs, mask=None, training=False):
        attention_scores = self.query(inputs)
        attention_scores = tf.expand_dims(attention_scores, axis=-1)
        attention_scores = tf.transpose(attention_scores, perm=[0, 2, 1, 3])
        dk = tf.cast(tf.shape(inputs)[-1], dtype=tf.float32)
        scale = 1 / tf.sqrt(dk)
        attention_scores = attention_scores * scale

        if mask is not None:
            mask = tf.expand_dims(mask, axis=1)
            mask = tf.cast(mask, dtype=tf.float32)
            attention_scores = attention_scores + (1.0 - mask * -1e9)
        attention_weights = self.dropout(attention_scores, training=training)
        attention_weights = tf.nn.softmax(attention_scores, axis=2)

        inputs = tf.expand_dims(inputs, axis=1)
        pooled_output = tf.reduce_sum(inputs * attention_weights, axis=2)
        pooled_output = tf.reduce_mean(pooled_output, axis=1)
        return self.norm(pooled_output)
