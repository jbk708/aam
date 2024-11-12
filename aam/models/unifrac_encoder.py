from __future__ import annotations

from typing import Union

import tensorflow as tf

from aam.losses import PairwiseLoss
from aam.models.base_sequence_encoder import BaseSequenceEncoder
from aam.models.transformers import TransformerEncoder
from aam.optimizers.gradient_accumulator import GradientAccumulator
from aam.optimizers.loss_scaler import LossScaler
from aam.utils import apply_random_mask, float_mask


@tf.keras.saving.register_keras_serializable(package="UniFracEncoder")
class UniFracEncoder(tf.keras.Model):
    def __init__(
        self,
        token_limit: int,
        dropout_rate: float = 0.0,
        embedding_dim: int = 128,
        attention_heads: int = 4,
        attention_layers: int = 4,
        intermediate_size: int = 1024,
        intermediate_activation: str = "gelu",
        max_bp: int = 150,
        include_alpha: bool = True,
        is_16S: bool = True,
        vocab_size: int = 6,
        add_token: bool = True,
        asv_dropout_rate: float = 0.0,
        accumulation_steps: int = 1,
        unifrac_metric: str = "unifrac",
        **kwargs,
    ):
        super(UniFracEncoder, self).__init__(**kwargs)

        self.token_limit = token_limit
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.intermediate_size = intermediate_size
        self.intermediate_activation = intermediate_activation
        self.max_bp = max_bp
        self.include_alpha = include_alpha
        self.is_16S = is_16S
        self.vocab_size = vocab_size
        self.add_token = add_token
        self.asv_dropout_rate = asv_dropout_rate
        self.accumulation_steps = accumulation_steps
        self.unifrac_metric = unifrac_metric

        self.loss_tracker = tf.keras.metrics.Mean()
        if self.unifrac_metric == "unifrac":
            self.unifrac_loss = PairwiseLoss()
            self.unifrac_out_dim = self.embedding_dim
        else:
            self.unifrac_loss = tf.keras.losses.MeanSquaredError(reduction="none")
            self.unifrac_out_dim = 1
        self.unifrac_tracker = tf.keras.metrics.Mean()

        # layers used in model
        self.base_encoder = BaseSequenceEncoder(
            self.embedding_dim,
            self.max_bp,
            self.token_limit,
            sample_attention_heads=self.attention_heads,
            sample_attention_layers=self.attention_layers,
            sample_intermediate_size=self.intermediate_size,
            dropout_rate=self.dropout_rate,
            nuc_attention_heads=1,
            nuc_attention_layers=3,
            nuc_intermediate_size=128,
            intermediate_activation=self.intermediate_activation,
            is_16S=self.is_16S,
            vocab_size=self.vocab_size,
            add_token=self.add_token,
            name="base_encoder",
        )

        self.unifrac_encoder = TransformerEncoder(
            num_layers=self.attention_layers,
            num_attention_heads=self.attention_heads,
            intermediate_size=intermediate_size,
            dropout_rate=self.dropout_rate,
            activation=self.intermediate_activation,
            name="unifrac_encoder",
        )
        self.unifrac_ff = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.embedding_dim,
                    use_bias=True,
                    dtype=tf.float32,
                    activation="gelu",
                ),
                tf.keras.layers.Dense(
                    self.unifrac_out_dim,
                    use_bias=True,
                    dtype=tf.float32,
                ),
            ]
        )

        self.loss_metrics = sorted(["loss", "target_loss", "count_mse"])
        self.gradient_accumulator = GradientAccumulator(self.accumulation_steps)
        self.loss_scaler = LossScaler()

    def evaluate_metric(self, dataset, metric, **kwargs):
        metric_index = self.loss_metrics.index(metric)
        evaluated_metrics = super(UniFracEncoder, self).evaluate(dataset, **kwargs)
        return evaluated_metrics[metric_index]

    def _compute_nuc_loss(self, nuc_tokens, nuc_pred):
        return self.base_encoder._compute_nuc_loss(nuc_tokens, nuc_pred)

    def _compute_unifrac_loss(
        self,
        y_true: tf.Tensor,
        unifrac_embeddings: tf.Tensor,
    ) -> tf.Tensor:
        loss = self.unifrac_loss(y_true, unifrac_embeddings)
        return tf.reduce_mean(loss)

    def _compute_loss(
        self,
        model_inputs: tuple[tf.Tensor, tf.Tensor],
        y_true: Union[tf.Tensor, tuple[tf.Tensor, tf.Tensor]],
        outputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        nuc_tokens, counts = model_inputs
        embeddings, unifrac_embeddings, nuc_pred = outputs
        unifrac_loss = self._compute_unifrac_loss(y_true, unifrac_embeddings)
        nuc_loss = self._compute_nuc_loss(nuc_tokens, nuc_pred) * 0.0
        loss = unifrac_loss + nuc_loss
        return [loss, unifrac_loss, nuc_loss]

    def predict_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, sample_ids = data
        embeddings, _, _ = self(inputs, training=False)

        return embeddings, sample_ids

    def train_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        if not self.gradient_accumulator.built:
            self.gradient_accumulator.build(self.optimizer, self)

        inputs, y = data
        y_target, unifrac_target = y
        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)
            loss, unifrac_loss, nuc_loss = self._compute_loss(
                inputs, unifrac_target, outputs
            )
            scaled_losses = self.loss_scaler([unifrac_loss])
            loss = tf.reduce_sum(tf.stack(scaled_losses, axis=0))

        gradients = tape.gradient(
            loss,
            self.trainable_variables,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )
        self.gradient_accumulator.apply_gradients(gradients)

        self.loss_tracker.update_state(loss)
        self.unifrac_tracker.update_state(unifrac_loss)
        self.base_encoder.nuc_entropy.update_state(nuc_loss)
        return {
            "loss": self.loss_tracker.result(),
            "unifrac_mse": self.unifrac_tracker.result(),
            "nuc_entropy": self.base_encoder.nuc_entropy.result(),
        }

    def test_step(
        self,
        data: Union[
            tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
            tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        ],
    ):
        inputs, y = data
        y_target, unifrac_target = y
        outputs = self(inputs, training=False)
        loss, unifrac_loss, nuc_loss = self._compute_loss(
            inputs, unifrac_target, outputs
        )

        self.loss_tracker.update_state(loss)
        self.unifrac_tracker.update_state(unifrac_loss)
        self.base_encoder.nuc_entropy.update_state(nuc_loss)
        return {
            "loss": self.loss_tracker.result(),
            "unifrac_mse": self.unifrac_tracker.result(),
            "nuc_entropy": self.base_encoder.nuc_entropy.result(),
        }

    def call(
        self, inputs: tuple[tf.Tensor, tf.Tensor], training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # keras cast all input to float so we need to manually cast to expected type
        tokens, counts = inputs
        tokens = tf.cast(tokens, dtype=tf.int32)
        counts = tf.cast(counts, dtype=tf.int32)

        # account for <SAMPLE> token
        count_mask = float_mask(counts, dtype=tf.int32)
        random_mask = None
        if training and self.asv_dropout_rate > 0:
            random_mask = apply_random_mask(count_mask, self.asv_dropout_rate)

        sample_embeddings, nuc_embeddings = self.base_encoder(
            tokens, random_mask=random_mask, training=training
        )

        if self.add_token:
            count_mask = tf.pad(count_mask, [[0, 0], [1, 0], [0, 0]], constant_values=1)
        count_attention_mask = count_mask

        unifrac_gated_embeddings = self.unifrac_encoder(
            sample_embeddings, mask=count_attention_mask, training=training
        )

        if self.add_token:
            unifrac_pred = unifrac_gated_embeddings[:, 0, :]
        else:
            mask = tf.cast(count_mask, dtype=tf.float32)
            unifrac_pred = tf.reduce_sum(unifrac_gated_embeddings * mask, axis=1)
            unifrac_pred /= tf.reduce_sum(mask, axis=1)

        unifrac_pred = self.unifrac_ff(unifrac_pred)
        unifrac_embeddings = sample_embeddings + unifrac_gated_embeddings
        return [unifrac_embeddings, unifrac_pred, nuc_embeddings]

    def base_embeddings(
        self, inputs: tuple[tf.Tensor, tf.Tensor]
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # keras cast all input to float so we need to manually cast to expected type
        tokens, counts = inputs
        tokens = tf.cast(tokens, dtype=tf.int32)
        counts = tf.cast(counts, dtype=tf.int32)

        sample_embeddings = self.base_encoder.base_embeddings(tokens)

        # account for <SAMPLE> token
        count_mask = float_mask(counts, dtype=tf.int32)
        count_mask = tf.pad(count_mask, [[0, 0], [1, 0], [0, 0]], constant_values=1)
        count_attention_mask = count_mask

        unifrac_gated_embeddings = self.unifrac_encoder(
            sample_embeddings, mask=count_attention_mask
        )
        unifrac_pred = unifrac_gated_embeddings[:, 0, :]
        unifrac_pred = self.unifrac_ff(unifrac_pred)

        unifrac_embeddings = (
            sample_embeddings + unifrac_gated_embeddings * self._unifrac_alpha
        )

        return unifrac_embeddings

    def asv_embeddings(
        self, inputs: tuple[tf.Tensor, tf.Tensor], training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # keras cast all input to float so we need to manually cast to expected type
        tokens, counts = inputs
        tokens = tf.cast(tokens, dtype=tf.int32)
        counts = tf.cast(counts, dtype=tf.int32)

        asv_embeddings = self.base_encoder.asv_embeddings(tokens, training=training)

        return asv_embeddings

    def asv_gradient(
        self, inputs: tuple[tf.Tensor, tf.Tensor], asv_embeddings
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # keras cast all input to float so we need to manually cast to expected type
        tokens, counts = inputs
        tokens = tf.cast(tokens, dtype=tf.int32)
        counts = tf.cast(counts, dtype=tf.int32)

        sample_embeddings = self.base_encoder.asv_gradient(tokens, asv_embeddings)

        # account for <SAMPLE> token
        count_mask = float_mask(counts, dtype=tf.int32)
        count_mask = tf.pad(count_mask, [[0, 0], [1, 0], [0, 0]], constant_values=1)
        count_attention_mask = count_mask

        unifrac_gated_embeddings = self.unifrac_encoder(
            sample_embeddings, mask=count_attention_mask
        )
        unifrac_pred = unifrac_gated_embeddings[:, 0, :]
        unifrac_pred = self.unifrac_ff(unifrac_pred)

        unifrac_embeddings = (
            sample_embeddings + unifrac_gated_embeddings * self._unifrac_alpha
        )

        return unifrac_embeddings

    def get_config(self):
        config = super(UniFracEncoder, self).get_config()
        config.update(
            {
                "token_limit": self.token_limit,
                "dropout_rate": self.dropout_rate,
                "embedding_dim": self.embedding_dim,
                "attention_heads": self.attention_heads,
                "attention_layers": self.attention_layers,
                "intermediate_size": self.intermediate_size,
                "intermediate_activation": self.intermediate_activation,
                "max_bp": self.max_bp,
                "include_alpha": self.include_alpha,
                "is_16S": self.is_16S,
                "vocab_size": self.vocab_size,
                "add_token": self.add_token,
                "asv_dropout_rate": self.asv_dropout_rate,
                "accumulation_steps": self.accumulation_steps,
                "unifrac_metric": self.unifrac_metric,
            }
        )
        return config
