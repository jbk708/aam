import tensorflow as tf


@tf.keras.saving.register_keras_serializable(package="TransformerLearningRateSchedule")
class TransformerLearningRateSchedule(
    tf.keras.optimizers.schedules.LearningRateSchedule
):
    def __init__(self, warmup_steps=100, decay_method="cosine", initial_lr=3e-4):
        super(TransformerLearningRateSchedule, self).__init__()

        self.warmup_steps = warmup_steps
        self.decay_method = decay_method
        self.initial_lr = initial_lr

    def __call__(self, step):
        # Linear warmup
        learning_rate = tf.cast(
            self.initial_lr * tf.math.minimum(step / self.warmup_steps, 1.0),
            dtype=tf.float32,
        )

        if self.decay_method == "cosine":
            # Cosine decay after warmup
            cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self.initial_lr,
                first_decay_steps=self.warmup_steps,
            )
            learning_rate = tf.cond(
                step < self.warmup_steps,
                lambda: learning_rate,
                lambda: cosine_decay(step - self.warmup_steps),
            )
        elif self.decay_method == "inv_sqrt":
            # Inverse Square Root decay after warmup (used in the original Transformer paper)
            inv_sqrt_decay = self.initial_lr * tf.math.rsqrt(
                tf.cast(step - self.warmup_steps + 1, tf.float32)
            )
            learning_rate = tf.cond(
                step < self.warmup_steps, lambda: learning_rate, lambda: inv_sqrt_decay
            )

        return learning_rate

    def get_config(self):
        config = {}
        config.update(
            {
                "warmup_steps": self.warmup_steps,
                "decay_method": self.decay_method,
                "initial_lr": self.initial_lr,
            }
        )
        return config


def cos_decay_with_warmup(lr, warmup_steps=5000, decay_steps=1000):
    # # Learning rate schedule: Warmup followed by cosine decay
    # lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    #     initial_learning_rate=lr, first_decay_steps=warmup_steps
    # )
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0,
        decay_steps=decay_steps,
        warmup_target=lr,
        warmup_steps=warmup_steps,
    )
    return lr_schedule


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf

    cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0,
        decay_steps=4000,
        warmup_target=0.0003,
        warmup_steps=100,
    )

    # Generate learning rates for a range of steps
    steps = np.arange(4000, dtype=np.float32)
    learning_rates = [cosine_decay(step).numpy() for step in steps]

    # Plot the learning rate schedule
    plt.figure(figsize=(10, 6))
    plt.plot(steps, learning_rates)
    plt.xlabel("Training Step")
    plt.ylabel("Learning Rate")
    plt.title("Transformer Learning Rate Schedule")
    plt.legend()
    plt.grid(True)
    plt.show()
