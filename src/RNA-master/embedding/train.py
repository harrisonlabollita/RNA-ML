import numpy as np, argparse
import os, sys, pickle, psutil, random

from model import RNA2STRUCTURE

import tensorflow as tf
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.optimizers import SGD, Adam
from tensorflow.python.keras.backend import categorical_crossentropy

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

from vocabulary import RNACoder, KerasDataGenerator, load_generators


def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--nb_epoch',
        type=int,
        default=1000,
        help='Number of epochs, default 100')

    parser.add_argument(
        '--batch_size', type=int, default=16, help='Batch size, default 16')

    parser.add_argument(
        '--split', type=float, default=0.3, help='Train-test split')

    parser.add_argument(
        '--max_seq', type=float, default=100, help='Max length of strand')

    parser.add_argument(
        '--embedding',
        type=int,
        default=0,
        help='Use model with Embedding layers')

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


class LearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs={}):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


def anneal_lr(epoch_no):
    return 0.001 * (1.0 / (1 + 2.0 * np.sqrt(epoch_no)))


def softmax_cross_entropy_with_logits(y_true, y_pred):
    mask = 1. - K.cast(K.equal(K.argmax(y_true, axis=-1), 0), K.floatx())
    return mask * tf.nn.softmax_cross_entropy_with_logits(
        labels=y_true, logits=y_pred)


def masked_categorical_crossentropy(y_true, y_pred):
    #mask = 1. - K.cast(K.equal(K.argmax(y_true, axis=-1), 0), K.floatx())
    return categorical_crossentropy(y_true, y_pred)


def fractional_accuracy(y_true, y_pred):
    equal = K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1))
    X = K.mean(K.sum(K.cast(equal, tf.float32), axis=-1))
    not_equal = K.not_equal(
        K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1))
    Y = K.mean(K.sum(K.cast(not_equal, tf.float32), axis=-1))
    return X / (X + Y)


def categorical_accuracy(y_true, y_pred):
    #mask = 1. - K.cast(K.equal(K.argmax(y_true, axis=-1), 0), K.floatx())
    return K.mean(
        K.cast(
            K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)),
            K.floatx()))


def get_callbacks(WEIGHTS_FPATH, LOG_FPATH, monitor):
    callbacks = [
        ModelCheckpoint(
            WEIGHTS_FPATH,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=True,
            mode='auto'),
        EarlyStopping(monitor=monitor, patience=3),

        #LearningRateScheduler(anneal_lr),
        #LearningRateTracker(),
        ReduceLROnPlateau(
            monitor=monitor, factor=0.2, patience=2, min_lr=1e-7, mode='auto'),
        CSVLogger(LOG_FPATH, separator=' ', append=True),
    ]
    return callbacks


if __name__ == '__main__':

    args = load_args()

    DATA_FILEPATH = str(args.sequence_data)

    MAX_SEQ_CUTOFF = int(args.max_seq)
    MAX_SEQ_LENGTH_ENC = int(args.max_seq) + 1
    MAX_SEQ_LENGTH_DEC = int(args.max_seq) + 2
    HIDDEN_DIM = int(args.hidden_dim)

    N_epochs = int(args.nb_epoch)
    VALIDATION_SPLIT = float(args.split)
    BATCH_SIZE = int(args.batch_size)

    LEARNING_RATE = float(args.lr)
    REGULARIZATION = float(args.l1l2)
    DROPOUT = float(args.dropout)

    ATTENTION = True
    TEACHER_FORCER = True
    USE_EMBEDDING = bool(args.embedding)

    LOAD_WEIGHTS = bool(args.load_weights)

    SAVE_TAG = '{}_{}_{}'.format(MAX_SEQ_LENGTH_ENC, MAX_SEQ_LENGTH_DEC,
                                 HIDDEN_DIM)
    WEIGHTS_FPATH = 'models/full_model_{}.h5'.format(SAVE_TAG)
    WEIGHTS_ENCODER = 'models/encoder_{}.h5'.format(SAVE_TAG)
    WEIGHTS_DECODER = 'models/decoder_{}.h5'.format(SAVE_TAG)
    LOG_FPATH = 'models/training_log_{}.log'.format(SAVE_TAG)

    ###############################################################
    """ Load training / testing data and encoder/decoder helper """

    baseCoder, train_gen, test_gen, train_steps, val_steps = load_generators(
        DATA_FILEPATH, MAX_SEQ_LENGTH_ENC, MAX_SEQ_LENGTH_DEC, MAX_SEQ_CUTOFF,
        BATCH_SIZE, VALIDATION_SPLIT, USE_EMBEDDING, TEACHER_FORCER)

    ###############################################################
    """ Load model """

    RNA = RNA2STRUCTURE(MAX_SEQ_LENGTH_ENC, MAX_SEQ_LENGTH_DEC)

    RNA.load(
        nodes=HIDDEN_DIM,
        reg=REGULARIZATION,
        dropout=DROPOUT,
        embedding=USE_EMBEDDING,
        batch_size=BATCH_SIZE)

    if LOAD_WEIGHTS:
        RNA.load_weights(
            WEIGHTS_FPATH,
            WEIGHTS_ENCODER,
            WEIGHTS_DECODER,
            attention=ATTENTION)

    full_model = RNA.full_model
    encoder_model = RNA.encoder_model
    decoder_model = RNA.decoder_model

    adam = Adam(lr=LEARNING_RATE, clipnorm=1.0)

    for model in [full_model, encoder_model, decoder_model]:
        model.compile(
            loss='categorical_crossentropy',
            sample_weight_mode='temporal',
            optimizer=adam,
            metrics=[categorical_accuracy])
        model.summary()

    plot_model(full_model, to_file='models/full_arch.png', show_shapes=True)
    plot_model(
        encoder_model, to_file='models/encoder_arch.png', show_shapes=True)
    plot_model(
        decoder_model, to_file='models/decoder_arch.png', show_shapes=True)

    ###############################################################
    """ Train the model """

    callbacks = get_callbacks(WEIGHTS_FPATH, LOG_FPATH,
                              "val_categorical_accuracy")

    try:

        full_model.fit_generator(
            train_gen,
            steps_per_epoch=train_steps,
            validation_data=test_gen,
            validation_steps=val_steps,
            epochs=N_epochs,
            callbacks=callbacks,
            verbose=1,
            workers=min(8,psutil.cpu_count()),
            use_multiprocessing=True,
        )

        encoder_model.save_weights(WEIGHTS_ENCODER)
        decoder_model.save_weights(WEIGHTS_DECODER)

    except KeyboardInterrupt as e:

        encoder_model.save_weights(WEIGHTS_ENCODER)
        decoder_model.save_weights(WEIGHTS_DECODER)
        print('Model training stopped early.')

    ###############################################################
    ''' 
    Inference 

    c = 0
    for (x, y, z) in test_gen:
        xs, xt = x

        full_pred = full_model.predict([xs, xt])
        decoded = full_pred.argmax(axis=2)[0]
        decoded = "".join([baseCoder.int_to_base[xp] for xp in decoded])

        xs = xs[0].reshape((1, MAX_SEQ_LENGTH_ENC, baseCoder.N_CHARS))
        xt = xt[0].reshape((1, MAX_SEQ_LENGTH_DEC - 1, baseCoder.N_CHARS))
        y = y[0].reshape((1, MAX_SEQ_LENGTH_DEC - 1, baseCoder.N_CHARS))

        pred_seq, attention_weights = RNA.InferBasePairs(xs)

        y_decode = y.argmax(axis=2)[0]
        y_decode = "".join([baseCoder.int_to_base[xp] for xp in y_decode])

        x_decode = xs.argmax(axis=2)[0]
        x_decode = "".join([baseCoder.int_to_base[xp] for xp in x_decode])

        print('input   ', x_decode)
        print('predict ', '3' + decoded)
        print('infer   ', pred_seq)
        print('true    ', '3' + y_decode)
        print('==========================================')

        c += 1
        if c % 10 == 0:
            break
    '''