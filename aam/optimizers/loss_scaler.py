import tensorflow as tf


class LossScaler:
    def __init__(self):
        self.moving_avg = None
        self.decay = 0.99
        self.accum_loss = tf.Variable(
            initial_value=1,
            trainable=False,
            dtype=tf.int32,
            name="accum_loss",
        )

    def __call__(self, losses):
        if self.moving_avg is None:
            self.scaled_loss = [
                tf.Variable(
                    initial_value=loss,
                    trainable=False,
                    dtype=tf.float32,
                    name=f"scaled_loss_{i}",
                )
                for i, loss in enumerate(losses)
            ]
            self.moving_avg = [
                tf.Variable(
                    initial_value=loss,
                    trainable=False,
                    dtype=tf.float32,
                    name=f"avg_loss_{i}",
                )
                for i, loss in enumerate(losses)
            ]

        for i in range(len(self.moving_avg)):
            self.moving_avg[i].assign(
                self.moving_avg[i] * self.decay + (1 - self.decay) * losses[i],
                read_value=False,
            )

        return [
            tf.math.divide_no_nan(losses[i], self.scaled_loss[i])
            for i in range(len(self.scaled_loss))
        ]

    def accumulate_loss(self):
        tf.cond(
            tf.equal(self.accum_loss, 1),
            true_fn=self._accumlate_loss,
            false_fn=lambda: None,
        )
        self.accum_loss.assign(0)

    def _accumlate_loss(self):
        tf.print("???")
        for i in range(len(self.scaled_loss)):
            self.scaled_loss[i].assign(self.moving_avg[i], read_value=False)
