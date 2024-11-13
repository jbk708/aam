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
        # self.log_steps = tf.constant(100, dtype=tf.int32, name="accum_steps")
        # self.log_step_counter = tf.Variable(
        #     0,
        #     dtype=tf.int32,
        #     trainable=False,
        #     name="accum_counter",
        # )
        self.built = False
        # self.file_writer = tf.summary.create_file_writer(
        #     "/home/kalen/aam-research-exam/research-exam/healty-age-regression/results/logs/gradients"
        # )

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
                synchronization=tf.VariableSynchronization.ON_READ,
                aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
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
        self.optimizer.apply_gradients(
            zip(self.gradient_accumulation, self.model.trainable_variables)
        )

        # self.log_step_counter.assign_add(1)
        # tf.cond(
        #     tf.equal(self.log_step_counter, self.log_steps),
        #     true_fn=self.log_gradients,
        #     false_fn=lambda: None,
        # )

        # reset
        self.accum_step_counter.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(
                tf.zeros_like(self.model.trainable_variables[i], dtype=tf.float32),
                read_value=False,
            )

    # def log_gradients(self):
    #     self.log_step_counter.assign(0)
    #     with self.file_writer.as_default():
    #         for grad, var in zip(
    #             self.gradient_accumulation, self.model.trainable_variables
    #         ):
    #             if grad is not None:
    #                 tf.summary.histogram(
    #                     var.name + "/gradient", grad, step=self.optimizer.iterations
    #                 )
