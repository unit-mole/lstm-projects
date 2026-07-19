
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="HAR")
class AttentionLayer(tf.keras.layers.Layer):
    """Temporal attention layer used by the saved HAR model."""
    def build(self, input_shape):
        self.W = self.add_weight(name="attention_weight", shape=(input_shape[-1], 1), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="attention_bias", shape=(input_shape[1], 1), initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        scores = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        weights = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(inputs * weights, axis=1)

    def get_config(self):
        return super().get_config()
