import tensorflow as tf
from tensorflow.keras import backend as K
from keras.layers import Layer

class AttentionTanh(Layer):
    def __init__(self, units=32, return_sequences=True, **kwargs):
        self.units = units
        self.return_sequences = return_sequences
        super(AttentionTanh, self).__init__(**kwargs)

    def build(self, input_shape):
        self.Q = self.add_weight(name="att_weight_Q", shape=(input_shape[-1], self.units),
                                 initializer="normal")
        self.K = self.add_weight(name="att_weight_K", shape=(input_shape[-1], self.units),
                                 initializer="normal")
        self.V = self.add_weight(name="att_weight_V", shape=(input_shape[-1], self.units),
                                 initializer="normal")
        super(AttentionTanh, self).build(input_shape)

    def call(self, x):
        xQ = tf.math.tanh(K.dot(x, self.Q))
        xK = tf.math.tanh(K.dot(x, self.K))
        xV = tf.math.tanh(K.dot(x, self.V))

        scores = K.dot(xQ, tf.transpose(xK)) / tf.math.sqrt(tf.cast(tf.shape(x)[-1], dtype='float32'))
        weights = tf.nn.softmax(scores, axis=-1)
        weighted_input = tf.matmul(weights, xV)

        if self.return_sequences:
            return weighted_input
        else:
            return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            output_shape = input_shape[:2] + (self.units,)
            return output_shape
        else:
            output_shape = input_shape[0], self.units
            return output_shape

    def get_config(self):
        config = {'units': self.units, 'return_sequences': self.return_sequences}
        base_config = super(AttentionTanh, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class AttentionRelu(Layer):
    def __init__(self, units=32, return_sequences=True, **kwargs):
        self.units = units
        self.return_sequences = return_sequences
        super(AttentionRelu, self).__init__(**kwargs)

    def build(self, input_shape):
        self.Q = self.add_weight(name="att_weight_Q", shape=(input_shape[-1], self.units),
                                 initializer="normal")
        self.K = self.add_weight(name="att_weight_K", shape=(input_shape[-1], self.units),
                                 initializer="normal")
        self.V = self.add_weight(name="att_weight_V", shape=(input_shape[-1], self.units),
                                 initializer="normal")
        super(AttentionRelu, self).build(input_shape)

    def call(self, x):
        xQ = tf.math.tanh(K.dot(x, self.Q))
        xK = tf.math.tanh(K.dot(x, self.K))
        xV = tf.math.tanh(K.dot(x, self.V))

        scores = K.dot(xQ, tf.transpose(xK)) / tf.math.sqrt(tf.cast(tf.shape(x)[-1], dtype='float32'))
        weights = tf.nn.softmax(scores, axis=-1)
        weighted_input = tf.matmul(weights, xV)
        
        if self.return_sequences:
            return weighted_input
        else:
            return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            output_shape = input_shape[:2] + (self.units,)
            return output_shape
        else:
            output_shape = input_shape[0], self.units
            return output_shape

    def get_config(self):
        config = {'units': self.units, 'return_sequences': self.return_sequences}
        base_config = super(AttentionRelu, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))