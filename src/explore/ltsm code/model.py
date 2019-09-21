import keras
from keras import backend as K
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Embedding, Masking, RepeatVector, Reshape
from keras.layers import Dense, Input, Dense, Dropout, Activation, BatchNormalization
from keras.layers import Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.regularizers import l1, l2, L1L2





model = Sequential()
model.add(LSTM(..., input_shape=(...)))
model.add(RepeatVector(...))
model.add(LSTM(..., return_sequences=True))
model.add(TimeDistributed(Dense(...)))

def basicModel(batch_size = 128):
    model = Sequential()
    model.add(LSTM(nodes, input_shape = (sequence_length, words), kernel_regularizer = reg))
    model.add(Dense(seqeunce length decoded * words))
    model.Reshape(sequence length dec, words)
    model.add(TimeDistributed(Dense(words, activation = 'softmax')))


def Base_Model(self, nodes=100, reg=1e-8, dropout=0.5, batch_size=16):

    reg = L1L2(reg)

    encoder_input = Input(
        shape=(self.max_sequence_length_enc, self.N_words), dtype='float32')

    mask = Masking(mask_value=0.0)(encoder_input)
    mask = BatchNormalization()(mask)

    l_lstm = LSTM(
        nodes,
        input_shape=(self.max_sequence_length_enc, self.N_words),
        kernel_regularizer=reg)(mask)
    l_lstm = Dense(self.max_sequence_length_dec * self.N_words)(l_lstm)
    l_lstm = Reshape((self.max_sequence_length_dec, self.N_words))(l_lstm)
    decoded_sequence = TimeDistributed(
        Dense(self.N_words, activation='softmax'))(l_lstm)
