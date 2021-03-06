import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import regularizers, constraints, initializers, activations
from tensorflow.python.keras.layers import Layer, LSTM


class AttentionLayer(Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(
            name='W_a',
            shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
            initializer='uniform',
            trainable=True)
        self.U_a = self.add_weight(
            name='U_a',
            shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
            initializer='uniform',
            trainable=True)
        self.V_a = self.add_weight(
            name='V_a',
            shape=tf.TensorShape((input_shape[0][2], 1)),
            initializer='uniform',
            trainable=True)

        super(AttentionLayer,
              self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, verbose=False):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs
        if verbose:
            print('encoder_out_seq>', encoder_out_seq.shape)
            print('decoder_out_seq>', decoder_out_seq.shape)

        def energy_step(inputs, states):
            """ Step function for computing energy for a single decoder state """

            assert_msg = "States must be a list. However states {} is of type {}".format(
                states, type(states))
            assert isinstance(states, list) or isinstance(states,
                                                          tuple), assert_msg
            """ Some parameters required for shaping tensors"""
            batch_size = encoder_out_seq.shape[0]
            en_seq_len, en_hidden = encoder_out_seq.shape[
                1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]
            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch_size*en_seq_len, latent_dim
            reshaped_enc_outputs = K.reshape(
                encoder_out_seq, (batch_size * en_seq_len, en_hidden))
            # <= batch_size*en_seq_len, latent_dim
            W_a_dot_s = K.reshape(
                K.dot(reshaped_enc_outputs, self.W_a),
                (batch_size, en_seq_len, en_hidden))
            if verbose:
                print('wa.s>', W_a_dot_s.shape)
            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a),
                                      1)  # <= batch_size, 1, latent_dim
            if verbose:
                print('Ua.h>', U_a_dot_h.shape)
            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            reshaped_Ws_plus_Uh = K.tanh(
                K.reshape(W_a_dot_s + U_a_dot_h,
                          (batch_size * en_seq_len, en_hidden)))
            if verbose:
                print('Ws+Uh>', reshaped_Ws_plus_Uh.shape)
            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.reshape(
                K.dot(reshaped_Ws_plus_Uh, self.V_a), (batch_size, en_seq_len))
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            if verbose:
                print('ei>', e_i.shape)

            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """
            # <= batch_size, hidden_size
            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
            if verbose:
                print('ci>', c_i.shape)
            return c_i, [c_i]

        # We are not using initial states, but need to pass something to K.rnn funciton
        fake_state_c = K.zeros(
            shape=(encoder_out_seq.shape[0], encoder_out_seq.shape[-1]))
        fake_state_e = K.zeros(
            shape=(encoder_out_seq.shape[0], encoder_out_seq.shape[1]))
        """ Computing energy outputs """
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step,
            decoder_out_seq,
            [fake_state_e],
        )
        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(
            context_step,
            e_outputs,
            [fake_state_c],
        )

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1],
                            input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1],
                            input_shape[0][1]))
        ]


class LuongAttention(tf.keras.Model):
    def __init__(self, rnn_size):
        super(LuongAttention, self).__init__()
        self.wa = tf.keras.layers.Dense(rnn_size)

    def call(self, decoder_output, encoder_output):
        # Dot score: h_t (dot) Wa (dot) h_s
        # encoder_output shape: (batch_size, max_len, rnn_size)
        # decoder_output shape: (batch_size, 1, rnn_size)
        # score will have shape: (batch_size, 1, max_len)
        score = tf.matmul(
            decoder_output, self.wa(encoder_output), transpose_b=True)

        # alignment vector a_t
        alignment = tf.nn.softmax(score, axis=2)

        # context vector c_t is the average sum of encoder output
        context = tf.matmul(alignment, encoder_output)

        return context, alignment


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, hidden_size)
        score = self.V(
            tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights