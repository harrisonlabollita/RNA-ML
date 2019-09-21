import numpy as np
import random
from tensorflow.python.keras.utils import Sequence

class RNACoder:
    def __init__(self, MAX_SEQ_LENGTH_ENC, MAX_SEQ_LENGTH_DEC, ALPHABET=None):
        self.MAX_SEQ_LENGTH_ENC = MAX_SEQ_LENGTH_ENC
        self.MAX_SEQ_LENGTH_DEC = MAX_SEQ_LENGTH_DEC
        self.ALPHABET = ALPHABET

        if not self.ALPHABET:
            self.ALPHABET = [
                '-',
                'U',
                'C',
                'A',
                'G',
                'Y',
                'R',
                'N',
                'W',
                'S',
                'M',
                'K',
                'D',
                'V',
                'H',
                'X',
                '3',
                '5',
            ]
            # self.ALPHABET = [
            #     '-',
            #     'U',
            #     'C',
            #     'A',
            #     'G',
            #     'X',
            #     '3',
            #     '5',
            # ]

        self.N_CHARS = len(self.ALPHABET)
        self.int_to_base = {
            x: y
            for (x, y) in zip(list(range(len(self.ALPHABET))), self.ALPHABET)
        }
        self.base_to_int = {
            x: y
            for (x, y) in zip(self.ALPHABET, list(range(len(self.ALPHABET))))
        }

    def one_hot_encode(self, sequence, size):
        encoding = list()
        for i in range(size):
            try:
                value = sequence[i]
            except:
                value = '-'
            vector = [0.0 for _ in range(self.N_CHARS)]
            vector[self.base_to_int[value]] = 1.0
            encoding.append(vector)
        return np.array(encoding)

    def one_hot_decode(self, encoded_seq):
        return [np.argmax(vector) for vector in encoded_seq]

    def decode_hot2text(self, encoded_seq):
        decoded = self.one_hot_decode(encoded_seq)
        return "".join([self.int_to_base[k] for k in decoded])

    def get_pair(self, sequence_in, sequence_out):
        sequence_in = sequence_in + '3'
        sequence_out = '3' + sequence_out + '5'

        X = self.one_hot_encode(sequence_in, size=self.MAX_SEQ_LENGTH_ENC)
        y = self.one_hot_encode(sequence_out, size=self.MAX_SEQ_LENGTH_DEC)

        X = X.reshape((1, X.shape[0], X.shape[1]))
        y = y.reshape((1, y.shape[0], y.shape[1]))

        return X, y

    def encode(self, sequence, size):
        encoding = list()
        for i in range(size):
            try:
                value = sequence[i]
            except:
                value = '-'
            vector = self.base_to_int[value]
            encoding.append(vector)
        return np.array(encoding)

    def get_pair_idx(self, sequence_in, sequence_out):
        sequence_in = sequence_in + '3'
        sequence_out = '3' + sequence_out + '5'

        X = self.encode(sequence_in, size=self.MAX_SEQ_LENGTH_ENC)
        y = self.encode(sequence_out, size=self.MAX_SEQ_LENGTH_DEC)

        X = X.reshape((1, X.shape[0]))
        y = y.reshape((1, y.shape[0]))

        return X, y


class KerasDataGenerator(Sequence):
    def __init__(self,
                 sequences,
                 batch_size,
                 split,
                 MAX_SEQ_LENGTH_ENC,
                 MAX_SEQ_LENGTH_DEC,
                 ALPHABET=None,
                 train=True,
                 embedding=True,
                 teacher=False,
                 shuffle=False):

        self.ids = sequences
        self.batch_size = batch_size
        self.split = split
        self.shuffle = shuffle

        self.MAX_SEQ_LENGTH_ENC = MAX_SEQ_LENGTH_ENC
        self.MAX_SEQ_LENGTH_DEC = MAX_SEQ_LENGTH_DEC
        self.ALPHABET = ALPHABET
        self.EMBEDDING = embedding

        self.train = train
        self.teacher_forcing = teacher
        self.Coder = RNACoder(self.MAX_SEQ_LENGTH_ENC, self.MAX_SEQ_LENGTH_DEC)
        self.N_CHARS = len(self.Coder.ALPHABET)
        self.load()

    def load(self):
        if self.train:
            self.ids = self.ids[:self.split]
        else:
            self.ids = self.ids[self.split:]
        self.index_range = list(range(self.__len__()))
        if self.shuffle:
            random.seed(random.randint(1, 1000))
            random.shuffle(self.index_range)
        else:
            self.ids = sorted(self.ids, key=lambda x: len(x[0]))
        self.epoch = 0

    def __len__(self):
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, idx):
        total_indicies = self.__len__()

        reshuffle = False
        if (idx + 1 == total_indicies):
            reshuffle = self.shuffle
            self.epoch += 1
        if reshuffle:
            random.seed(random.randint(1, 1000))
            random.shuffle(self.index_range)

        idx = self.index_range[idx]
        batch = self.ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.prepare_batch(batch)

    def prepare_batch(self, batch):

        batch_x = []
        batch_t = []
        batch_y = []
        weights = []

        for (seq_in, seq_out) in batch:

            if self.teacher_forcing:

                if self.EMBEDDING:
                    x_encode, X_decode = self.Coder.get_pair_idx(seq_in, seq_out)
                    _, _x_decode = self.Coder.get_pair(seq_in, seq_out)
                    x_decode = X_decode[:, :-1]
                    y_encode = _x_decode[:, 1:, :]
                else:
                    x_encode, _x_decode = self.Coder.get_pair(seq_in, seq_out)
                    x_decode = _x_decode[:, :-1, :]
                    y_encode = _x_decode[:, 1:, :]

                batch_x.append(x_encode)
                batch_t.append(x_decode)
                batch_y.append(y_encode)

            else:
                # Get seq_x and seq_y one-hot states
                x_encode, y_encode = self.Coder.get_pair(seq_in, seq_out)
                batch_x.append(x_encode)
                batch_y.append(y_encode)

            # Get the sample_weights.
            w = np.ones(y_encode.shape[1])

            # Mask padding chars '-'
            #w[self.MAX_SEQ_LENGTH_DEC:] = 0.0
            # Mask 80% of no-bp padding chars X
            #wildcards = [i for i, l in enumerate(seq_out) if l == 'X']
            # wildcards = [i for i, l in enumerate(seq_out[1:]) if l != '-']
            # if len(wildcards) >= 5:
            #     w_mask = random.sample(wildcards, np.int(0.8 * len(wildcards)))
            #     w[w_mask] = 0.0

            weights.append(w)

        batch_x = np.vstack(batch_x).astype(np.float32)
        batch_y = np.vstack(batch_y).astype(np.float32)
        weights = np.vstack(weights).astype(np.float32)

        if self.teacher_forcing:
            batch_t = np.vstack(batch_t).astype(np.float32)
            return ([batch_x, batch_t], batch_y, weights)

        return (batch_x, batch_y, weights)


def load_generators(DATA_FILEPATH, MAX_SEQ_LENGTH_ENC,
                    MAX_SEQ_LENGTH_DEC, MAX_SEQ_CUTOFF, BATCH_SIZE, VALIDATION_SPLIT,
                    EMBEDDING, TEACHER_FORCER):

    with open(DATA_FILEPATH, "r") as fid:
        structures = [line.strip("\n").split(" ") for line in fid.readlines()]
        structures = sorted(structures, key=lambda x: len(x[0]))

    structures = [x for x in structures if len(x[0]) <= MAX_SEQ_CUTOFF]
    structures = [x for x in structures if x[1].count('X') != len(x[1])]
    #structures = [x for x in structures if len(list(set(x[0]))) == 4]

    print("There are {} sequences with length <= {}".format(
        len(structures), MAX_SEQ_CUTOFF))

    #total_chars = sum([len(x[0]) + 3 for x in structures])
    #total_chars_plus_pad = float((MAX_SEQ_LENGTH_ENC) * len(structures))
    #frac_pad_char = total_chars_plus_pad / (total_chars + total_chars_plus_pad)
    #print("Fraction of padding character '-': {}".format(frac_pad_char))

    # Shuffle before passing to generator
    # Generator will resort the splits if shuffle = False
    random.seed(1234)
    random.shuffle(structures)

    N_samples = len(structures)
    N_batches = np.ceil(N_samples / BATCH_SIZE)
    split = int(N_samples * (1 - VALIDATION_SPLIT))

    baseCoder = RNACoder(MAX_SEQ_LENGTH_ENC, MAX_SEQ_LENGTH_DEC)

    train_gen = KerasDataGenerator(
        structures,
        BATCH_SIZE,
        split,
        MAX_SEQ_LENGTH_ENC,
        MAX_SEQ_LENGTH_DEC,
        embedding=EMBEDDING,
        teacher=TEACHER_FORCER,
        shuffle=True)

    test_gen = KerasDataGenerator(
        structures,
        BATCH_SIZE,
        split,
        MAX_SEQ_LENGTH_ENC,
        MAX_SEQ_LENGTH_DEC,
        train=False,
        embedding=EMBEDDING,
        teacher=TEACHER_FORCER,
        shuffle=True)

    train_steps = np.ceil((1 - VALIDATION_SPLIT) * N_batches)
    val_steps = np.ceil(VALIDATION_SPLIT * N_batches)

    return baseCoder, train_gen, test_gen, train_steps, val_steps