from tensorflow.python.keras.layers import Embedding, Masking, RepeatVector, Reshape
from tensorflow.python.keras.layers import Dense, Input, Dense, Dropout, Activation, BatchNormalization
from tensorflow.python.keras.layers import Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from tensorflow.python.keras.layers import CuDNNLSTM, Concatenate, dot
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.regularizers import l1, l2, L1L2
from tensorflow.python.keras import backend as K

from attention_decoder import AttentionLayer
from sequence_encoder import RNABaseDataGenerator
from vocabulary import RNACoder, KerasDataGenerator

import numpy as np


class RNA2STRUCTURE:
    def __init__(self,
                 max_sequence_length=1000,
                 return_probabilities=False,
                 trainable=True):

        self.max_sequence_length = max_sequence_length
        self.return_probabilities = return_probabilities
        self.trainable = trainable
        self.Coder = RNACoder(self.max_sequence_length)
        self.N_words = len(self.Coder.ALPHABET)

    def load(self,
             nodes=100,
             reg=1e-8,
             dropout=0.5,
             batch_size=16,
             attention=True):
        if attention:
            self.Attention_Model(
                nodes=nodes, reg=reg, dropout=dropout, batch_size=batch_size)
        else:
            self.Base_Model(
                nodes=nodes, reg=reg, dropout=dropout, batch_size=batch_size)

    def load_weights(self,
                     full_path,
                     encoder_path,
                     decoder_path,
                     attention=True):
        try:
            if attention:
                custom_objects = {'AttentionLayer': AttentionLayer}
                self.full_model.load_weights(
                    full_path)
                self.encoder_model.load_weights(
                    encoder_path)
                self.decoder_model.load_weights(
                    decoder_path)
            else:
                self.full_model = load_model(full_path)
                self.encoder_model = load_model(encoder_path)
                self.decoder_model = load_model(decoder_path)

        except Exception as E:
            print("Exception", E)
            print("You must first load a model using load().")
            import sys
            sys.exit()

    def Base_Model(self, nodes=100, reg=1e-8, dropout=0.5, batch_size=16):
        reg = L1L2(reg)

        encoder_input = Input(
            shape=(self.max_sequence_length, self.N_words), dtype='float32')

        mask = Masking(mask_value=0.0)(encoder_input)
        mask = BatchNormalization()(mask)

        l_lstm = LSTM(
            nodes,
            input_shape=(self.max_sequence_length, self.N_words),
            kernel_regularizer=reg)(mask)
        l_lstm = Dense(self.max_sequence_length * self.N_words)(l_lstm)
        l_lstm = Reshape((self.max_sequence_length, self.N_words))(l_lstm)
        decoded_sequence = TimeDistributed(
            Dense(self.N_words, activation='softmax'))(l_lstm)

        full_model = Model(encoder_input, decoded_sequence)
        self.full_model = full_model

    def Attention_Model(self, nodes=100, reg=0.0, dropout=0.0, batch_size=16):
        reg = L1L2(reg)

        # Encoder

        encoder_input = Input(
            batch_shape=(batch_size, self.max_sequence_length, self.N_words),
            dtype='float32')
        #encoder_batch_norm = BatchNormalization()
        #mask = encoder_batch_norm(encoder_input)

        encoder = Bidirectional(
            GRU(
                nodes,
                #stateful=True,
                return_state=True,
                return_sequences=True,
                recurrent_dropout=dropout,
                kernel_regularizer=reg,
                kernel_initializer='he_normal',
                recurrent_initializer='he_normal',
                name='encoder_gru'),
            #merge_mode="concat",
            name='bidirectional_encoder')

        encoder_out, encoder_fwd_state, encoder_back_state = encoder(
            encoder_input)

        # Decoder

        decoder_input = Input(
            batch_shape=(batch_size, self.max_sequence_length - 1,
                         self.N_words),
            dtype='float32')
        #decoder_batch_norm = BatchNormalization()
        #mask = decoder_batch_norm(decoder_input)

        decoder = Bidirectional(
            GRU(
                nodes,
                #stateful=True,
                return_state=True,
                return_sequences=True,
                recurrent_dropout=dropout,
                kernel_regularizer=reg,
                kernel_initializer='he_normal',
                recurrent_initializer='he_normal',
                name='decoder_gru'),
            #merge_mode="concat",
            name='bidirectional_decoder')

        decoder_out, decoder_fwd_state, decoder_back_state = decoder(
            decoder_input,
            initial_state=[encoder_fwd_state, encoder_back_state])

        # Attention
        attn_layer = AttentionLayer(name='attention_layer')
        attn_out, attn_states = attn_layer([encoder_out, decoder_out])

        decoder_combined_context = Concatenate(
            axis=-1, name='concat_layer')([decoder_out, attn_out])

        # Dense
        dense_1 = Dense(nodes, activation="tanh")
        dense_time_1 = TimeDistributed(dense_1)
        dense_2 = Dense(self.N_words, activation='softmax')
        dense_time_2 = TimeDistributed(dense_2)
        decoded_sequence = dense_time_2(decoder_combined_context)

        full_model = Model([encoder_input, decoder_input], decoded_sequence)

        """ Encoder (Inference) model """
        encoder_inf_inputs = Input(
            batch_shape=(1, self.max_sequence_length, self.N_words),
            name='encoder_inf_inputs')
        #encoder_inf_inputs_masked = encoder_batch_norm(encoder_inf_inputs)

        encoder_inf_out, encoder_inf_fwd_state, encoder_inf_back_state = encoder(
            encoder_inf_inputs)

        encoder_model = Model(
            inputs=encoder_inf_inputs,
            outputs=[
                encoder_inf_out, encoder_inf_fwd_state, encoder_inf_back_state
            ])

        """ Decoder (Inference) model """
        decoder_inf_inputs = Input(
            batch_shape=(1, 1, self.N_words), name='decoder_word_inputs')
        #decoder_inf_inputs_masked = decoder_batch_norm(decoder_inf_inputs)

        encoder_inf_states = Input(
            batch_shape=(1, self.max_sequence_length, 2 * nodes),
            name='encoder_inf_states')

        decoder_init_fwd_state = Input(
            batch_shape=(1, nodes), name='decoder_fwd_init')

        decoder_init_back_state = Input(
            batch_shape=(1, nodes), name='decoder_back_init')

        decoder_inf_out, decoder_inf_fwd_state, decoder_inf_back_state = decoder(
            decoder_inf_inputs,
            initial_state=[decoder_init_fwd_state, decoder_init_back_state])

        # Attention
        attn_inf_out, attn_inf_states = attn_layer(
            [encoder_inf_states, decoder_inf_out])
        decoder_inf_concat = Concatenate(
            axis=-1, name='concat')([decoder_inf_out, attn_inf_out])

        # Output
        decoder_inf_pred = TimeDistributed(dense_2)(decoder_inf_concat)

        decoder_model = Model(
            inputs=[
                encoder_inf_states, decoder_init_fwd_state,
                decoder_init_back_state, decoder_inf_inputs
            ],
            outputs=[
                decoder_inf_pred, attn_inf_states, decoder_inf_fwd_state,
                decoder_inf_back_state
            ])

        self.full_model = full_model
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model

        #return full_model, encoder_model, decoder_model

    def InferBasePairs(self, encoder_input, dec_inp):
        """
        Infer logic
        :param encoder_model: keras.Model
        :param decoder_model: keras.Model
        :param test_en_seq: sequence of word ids
        :param en_vsize: int
        :param fr_vsize: int
        :return:
        """

        amax = np.argmax(encoder_input, axis=2)
        try:
            OUTPUT_LENGTH = np.where(amax[0] == 0)[0][0]
        except:
            OUTPUT_LENGTH = len(amax[0]) + 3

        decoded_basepairs = '3' # s.o.s. char
        decoder_input = np.zeros((1, 1, self.N_words))
        decoder_input[0, 0, self.Coder.base_to_int[decoded_basepairs]] = 1.0

        enc_outs, enc_fwd_state, enc_back_state = self.encoder_model.predict(
            encoder_input)
        dec_fwd_state, dec_back_state = enc_fwd_state, enc_back_state

        attention_weights = []
        stop_condition = False

        c = 1
        while not stop_condition:

            pred = self.decoder_model.predict(
                [enc_outs, dec_fwd_state, dec_back_state, decoder_input])
            dec_out, attention, dec_fwd_state, dec_back_state = pred

            #print(dec_out, np.argmax(dec_out, axis=2)[0,0])

            dec_ind = np.argmax(dec_out, axis=2)[0, 0]
            sampled_char = self.Coder.int_to_base[dec_ind]

            #decoder_input = dec_inp[0][c, :]
            #decoder_input = decoder_input.reshape((1, 1, self.N_words))
            #print('a', decoder_input)

            decoder_input = np.zeros((1, 1, self.N_words))
            decoder_input[0, 0, dec_ind] = 1.0
            #print('b', decoder_input)

            attention_weights.append((dec_ind, attention))
            decoded_basepairs += sampled_char

            if (sampled_char == '.' or len(decoded_basepairs) >=
                self.max_sequence_length or c >= 52):
                stop_condition = True

            c += 1

        

        return decoded_basepairs, attention_weights
