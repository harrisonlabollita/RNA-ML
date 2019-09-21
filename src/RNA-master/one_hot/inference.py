import numpy as np, argparse
import os, sys, pickle, psutil, random

from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.optimizers import SGD, Adam
from tensorflow.python.keras.models import load_model

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

from vocabulary import RNACoder, KerasDataGenerator
from attention_decoder import AttentionLayer
from model import RNA2STRUCTURE
from train import load_generators

from visualize_attention import plot_attention_weights


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--nb_epoch',
        type=int,
        default=1000,
        help='Number of epochs, default 100')
    parser.add_argument(
        '--batch_size', type=int, default=1, help='Batch size, default 1')
    parser.add_argument(
        '--split', type=float, default=0.3, help='Train-test split')
    parser.add_argument(
        '--max_seq', type=float, default=100, help='Max length of strand')
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=64,
        help='Encoder/decoder hidden size, default 64')
    parser.add_argument(
        '--l1l2',
        type=float,
        default=0,
        help='l1l2 regularization parameter , default 0')
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.0,
        help='Dropout parameter , default 0')
    parser.add_argument(
        '--lr', type=float, default=0.001, help='Learning rate, default 1e-3')
    parser.add_argument(
        '--attention', type=bool, default=True, help='Use attention model')
    parser.add_argument(
        '--load_weights', type=bool, default=False, help='Load model weights')
    parser.add_argument(
        '--save_weights', type=str, default='', help='Load model weights')
    parser.add_argument(
        '--sequence_data',
        type=str,
        default='data/training/sequences.txt',
        help='Filepath leading to text file containing RNA data')
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = load_args()

    DATA_FILEPATH = str(args.sequence_data)

    MAX_SEQ_LENGTH_ENC = int(args.max_seq)
    MAX_SEQ_LENGTH_DEC = int(args.max_seq) + 2
    MAX_SEQ_CUTOFF = MAX_SEQ_LENGTH_ENC
    HIDDEN_DIM = int(args.hidden_dim)

    N_epochs = int(args.nb_epoch)
    VALIDATION_SPLIT = float(args.split)
    BATCH_SIZE = int(args.batch_size)

    LEARNING_RATE = float(args.lr)
    REGULARIZATION = float(args.l1l2)
    DROPOUT = float(args.dropout)

    ATTENTION = True  #bool(args.attention)
    TEACHER_FORCER = True

    LOAD_WEIGHTS = bool(args.load_weights)

    """ 
        Set up directory for saving attention maps 
    """
    HERE = os.path.realpath(os.path.join(os.path.realpath(__file__), '..'))
    if not os.path.exists(os.path.join(HERE, 'attention_maps')):
        os.makedirs(os.path.join(HERE, 'attention_maps'))

    ###############################################################

    """ 
        Load model 
        
        -- BATCH_SIZE must equal that used for training the model
        -- Must call model and load trained weights (rather than saved model)
            due to serialization issues with return_state in RNN models in 
            current Keras.
    """

    # create a directory if it doesn't already exist
    HERE = os.path.realpath(os.path.join(os.path.realpath(__file__), '..'))
    if not os.path.exists(os.path.join(HERE, 'attention_maps')):
        os.makedirs(os.path.join(HERE, 'attention_maps'))

    WEIGHTS_FPATH = 'models/full_model_{}_{}.h5'.format(
        MAX_SEQ_LENGTH_ENC, MAX_SEQ_LENGTH_DEC)
    WEIGHTS_ENCODER = 'models/encoder_{}_{}.h5'.format(MAX_SEQ_LENGTH_ENC,
                                                       MAX_SEQ_LENGTH_DEC)
    WEIGHTS_DECODER = 'models/decoder_{}_{}.h5'.format(MAX_SEQ_LENGTH_ENC,
                                                       MAX_SEQ_LENGTH_DEC)
    LOG_FPATH = 'models/training_log_{}_{}.log'.format(MAX_SEQ_LENGTH_ENC,
                                                       MAX_SEQ_LENGTH_DEC)

    RNA = RNA2STRUCTURE(MAX_SEQ_LENGTH_ENC, MAX_SEQ_LENGTH_DEC, trainable=False)

    RNA.load(
        nodes=HIDDEN_DIM,
        reg=REGULARIZATION,
        dropout=DROPOUT,
        batch_size=BATCH_SIZE)

    RNA.load_weights(
        WEIGHTS_FPATH,
        WEIGHTS_ENCODER,
        WEIGHTS_DECODER,
        attention=ATTENTION)

    full_model = RNA.full_model
    encoder_model = RNA.encoder_model
    decoder_model = RNA.decoder_model

    adam = Adam(lr=LEARNING_RATE, clipnorm=1)

    for model in [full_model, encoder_model, decoder_model]:
        model.compile(
            loss='categorical_crossentropy',
            sample_weight_mode='temporal',
            optimizer=adam,
            metrics=['accuracy'])
        model.summary()

    """ 
        Load training / testing data and encoder/decoder helper 
        
        Reset batch_size = 1 for inference. 
    """
    MAX_SEQ_CUTOFF = 25
    
    BATCH_SIZE = 1
    baseCoder, train_gen, test_gen, train_steps, val_steps = load_generators(
        DATA_FILEPATH, MAX_SEQ_LENGTH_ENC-1, MAX_SEQ_LENGTH_DEC, MAX_SEQ_CUTOFF,
        BATCH_SIZE, VALIDATION_SPLIT, TEACHER_FORCER)

    c = 0
    for (x, y, z) in test_gen:
        xs, xt = x

        pred_seq, attention_weights = RNA.InferBasePairs(xs)

        y_decode = y.argmax(axis=2)[0]
        y_decode = "".join([baseCoder.int_to_base[xp] for xp in y_decode])

        x_decode = xs.argmax(axis=2)[0]
        x_decode = "".join([baseCoder.int_to_base[xp] for xp in x_decode])

        print('input   ', x_decode)
        print('infer   ', pred_seq)
        print('true    ', '3' + y_decode)
        print('==========================================')

        attention_weights = attention_weights[:MAX_SEQ_LENGTH_DEC]
        input_seq_att = x_decode[:MAX_SEQ_LENGTH_ENC]
        input_seq_att = input_seq_att[:-input_seq_att.count('-')]

        decode_seq_att = pred_seq[1:MAX_SEQ_LENGTH_DEC - 1]

        plot_attention_weights(attention_weights, input_seq_att,
                               decode_seq_att)

        c += 1
        if c % 50 == 0:
            break
