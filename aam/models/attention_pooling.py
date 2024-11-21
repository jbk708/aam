import tensorflow as tf


class AttentionPooling(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionPooling, self).__init__()
        self.query = tf.keras.layers.Dense(1)

    # def build(self, input_shape):
    #     self.emb_dim = input_shape[-1]
    #     # self.query = self.add_weight(
    #     #     "query",
    #     #     shape=(1, 1, self.emb_dim),
    #     #     initializer="glorot_uniform",
    #     #     dtype=tf.float32,
    #     # )
    #     self.pooler = tfm.nlp.layers.ReZeroTransformer(
    #         num_attention_heads=4,
    #         inner_dim=self.emb_dim,
    #         inner_activation="gelu",
    #     )

    def call(self, inputs, mask=None):
        # # batch_dim = tf.shape(inputs)[0]
        # # query = tf.broadcast_to(self.query, shape=[batch_dim, 1, self.emb_dim])
        # # tf.print(tf.reduce_sum(self.query))
        # query = tf.transpose(inputs, perm=[0, 2, 1])
        # query = self.query(inputs)
        # query = tf.transpose(inputs, perm=[0, 2, 1])
        # if mask is not None:
        #     mask = tf.transpose(mask, perm=[0, 2, 1])
        #     # query = query * mask
        #     # tf.print(query, mask)
        # pooled_output = self.pooler([query, inputs, mask])
        # pooled_output = tf.squeeze(pooled_output, axis=1)
        attention_scores = self.query(inputs)

        if mask is not None:
            mask = tf.cast(mask, dtype=tf.float32)
            attention_scores = attention_scores + (1.0 - mask * -1e9)

        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        pooled_output = tf.reduce_sum(inputs * attention_weights, axis=1)
        return pooled_output
