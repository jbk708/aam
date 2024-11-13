import tensorflow as tf


class GradientAccumulator:
    def __init__(self, accumulation_steps):
        self.accum_steps = tf.constant(
            accumulation_steps, dtype=tf.int32, name="accum_steps"
        )
        self.accum_step_counter = tf.Variable(
            0,
            dtype=tf.int32,
            trainable=False,
            name="accum_counter",
        )
        self.built = False

    def build(self, optimizer, model):
        self.built = True
        self.optimizer = optimizer
        self.model = model

        # reinitialize gradient accumulator
        self.gradient_accumulation = [
            tf.Variable(
                tf.zeros_like(v, dtype=tf.float32),
                trainable=False,
                name="accum_" + str(i),
            )
            for i, v in enumerate(self.model.trainable_variables)
        ]

    def apply_gradients(self, gradients):
        self.accum_step_counter.assign_add(1)

        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(
                gradients[i],
                read_value=False,
            )

        # apply accumulated gradients
        tf.cond(
            tf.equal(self.accum_step_counter, self.accum_steps),
            true_fn=self.apply_accu_gradients,
            false_fn=lambda: None,
        )

    def apply_accu_gradients(self):
        """Performs gradient update and resets slots afterwards."""
        # apply accumulated gradients
        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(
                self.gradient_accumulation[i]
                / tf.cast(self.accum_steps, dtype=tf.float32),
                read_value=False,
            )

        self.optimizer.apply_gradients(
            zip(self.gradient_accumulation, self.model.trainable_variables)
        )

        # reset
        self.accum_step_counter.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(
                tf.zeros_like(self.model.trainable_variables[i], dtype=tf.float32),
                read_value=False,
            )
