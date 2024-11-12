import tensorflow as tf


class LossScaler:
    def __init__(self):
        self.moving_avg = None
        self.decay = 0.99

    def __call__(self, losses):
        if self.moving_avg is None:
            self.moving_avg = [
                tf.Variable(
                    initial_value=loss,
                    trainable=False,
                    dtype=tf.float32,
                    name=f"avg_loss_{i}",
                )
                for i, loss in enumerate(losses)
            ]

        # Accumulate batch gradients
        for i in range(len(self.moving_avg)):
            self.moving_avg[i].assign(
                self.moving_avg[i] * self.decay + (1 - self.decay) * losses[i],
                read_value=False,
            )
        return [losses[i] / self.moving_avg[i] for i in range(len(self.moving_avg))]
