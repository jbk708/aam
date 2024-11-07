import tensorflow as tf


@tf.keras.saving.register_keras_serializable(package="ConvBlock")
class ConvBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        activation="gelu",
        use_bias=True,
        norm_epsilon=1e-6,
        dilation=1,
        **kwargs,
    ):
        super(ConvBlock, self).__init__(**kwargs)
        self._activation = activation
        self._use_bias = use_bias
        self._norm_epsilon = norm_epsilon
        self._dilation = dilation

        self.conv_norm = tf.keras.layers.LayerNormalization(epsilon=self._norm_epsilon)

        self.ff_norm = tf.keras.layers.LayerNormalization(epsilon=self._norm_epsilon)

    def build(self, input_shape):
        seq_dim = input_shape[1]
        emb_dim = input_shape[2]
        self.conv1d = tf.keras.layers.Conv1D(
            seq_dim,
            kernel_size=3,
            strides=1,
            padding="same",
            activation=self._activation,
            dilation_rate=self._dilation,
        )
        self.ff = tf.keras.layers.Dense(
            emb_dim, use_bias=self._use_bias, activation=self._activation
        )

    def get_config(self):
        config = {
            "activation": self._activation,
            "use_bias": self._use_bias,
            "norm_epsilon": self._norm_epsilon,
            "dilation": self._dilation,
        }
        base_config = super(ConvBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, encoder_inputs, training=False):
        """Return the output of the encoder.

        Args:
          encoder_inputs: A tensor with shape `(batch_size, input_length,
            hidden_size)`.
          attention_mask: A mask for the encoder self-attention layer with shape
            `(batch_size, input_length, input_length)`.

        Returns:
          Output of encoder which is a `float32` tensor with shape
            `(batch_size, input_length, hidden_size)`.
        """
        conv_inputs = tf.transpose(encoder_inputs, perm=[0, 2, 1])
        conv_outputs = tf.transpose(self.conv1d(conv_inputs), perm=[0, 2, 1])
        conv_outputs = encoder_inputs + self.conv_norm(conv_outputs)

        ff_outputs = self.ff(conv_outputs)
        output = conv_outputs + self.ff_norm(ff_outputs)
        return output


@tf.keras.saving.register_keras_serializable(package="ConvModule")
class ConvModule(tf.keras.layers.Layer):
    def __init__(
        self,
        layers,
        activation="gelu",
        use_bias=True,
        norm_epsilon=1e-6,
        **kwargs,
    ):
        super(ConvModule, self).__init__(**kwargs)
        self.layers = layers
        self._activation = activation
        self._use_bias = use_bias
        self._norm_epsilon = norm_epsilon

        self.conv_blocks = []
        dilation = 1
        for i in range(self.layers):
            self.conv_blocks.append(
                ConvBlock(
                    self._activation,
                    self._use_bias,
                    self._norm_epsilon,
                    dilation=(2**i) * dilation,
                )
            )

    def get_config(self):
        config = {
            "layers": self.layers,
            "activation": self._activation,
            "use_bias": self._use_bias,
            "norm_epsilon": self._norm_epsilon,
        }
        base_config = super(ConvModule, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, encoder_inputs, training=False):
        """Return the output of the encoder.

        Args:
          encoder_inputs: A tensor with shape `(batch_size, input_length,
            hidden_size)`.
          attention_mask: A mask for the encoder self-attention layer with shape
            `(batch_size, input_length, input_length)`.

        Returns:
          Output of encoder which is a `float32` tensor with shape
            `(batch_size, input_length, hidden_size)`.
        """
        for layer_idx in range(self.layers):
            encoder_inputs = self.conv_blocks[layer_idx](encoder_inputs)
        return encoder_inputs
