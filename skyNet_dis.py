from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from attentionLayer import Wrapper


class SkyNet_Dis:
    def __init__(self, params, mode, iter):
        self.params = params
        self.iter = iter
        self.mode = mode
        self.build_graph()

    def build_graph(self):
        if self.mode == 'train':
            self.mode = tf.estimator.ModeKeys.TRAIN
        elif self.mode == 'infer':
            self.mode = tf.estimator.ModeKeys.PREDICT
        elif self.mode == 'eval':
            self.mode = tf.estimator.ModeKeys.EVAL
        else:
            raise ValueError("Please choose mode: train, eval or infer")
        tf.compat.v1.logging.info("Building {} model...".format(self.mode))

        self.src, self.src_length, self.trgt, self.trgt_length = self.iter
        self.batch_size = tf.shape(self.src_length)[0]
        enc_outputs, enc_state = self._build_encoder()
        self.logits = self._build_decoder(enc_outputs, enc_state)

        self.p = tf.nn.sigmoid(self.logits)

    def _build_encoder(self):
        dropout = self.params.dropout if self.mode == tf.estimator.ModeKeys.TRAIN else 0.0

        with tf.compat.v1.variable_scope("encoder") as scope:
            cell = tf.compat.v1.nn.rnn_cell.LSTMCell(self.params.num_units)

            if dropout > 0.0:
                cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=(1.0 - dropout))

            enc_outputs, enc_state = tf.compat.v1.nn.dynamic_rnn(cell, self.src, sequence_length=self.src_length,
                                                                 dtype=tf.float32)

        return enc_outputs, enc_state

    def _build_decoder(self, enc_outputs, enc_state):
        with tf.compat.v1.variable_scope("decoder") as decoder_scope:
            cell, decoder_initial_state = self._build_decoder_cell(enc_outputs, enc_state, self.src_length)

            initial_inputs = tf.fill([self.batch_size, self.params.input_dim], 0.0)
            logits, _ = cell.call(inputs=initial_inputs, state=decoder_initial_state)
        return logits

    def _build_decoder_cell(self, enc_outputs, enc_state, src_length):
        if self.params.time_major:
            memory = tf.transpose(enc_outputs, [1, 0, 2])
        else:
            memory = enc_outputs

        cell = tf.compat.v1.nn.rnn_cell.LSTMCell(self.params.num_units)

        cell = Wrapper(cell, self.params.num_units, memory, memory_sequence_length=src_length, name="attention")

        decoder_initial_state = cell.zero_state(self.batch_size, dtype=tf.float32).clone(cell_state=enc_state)

        return cell, decoder_initial_state

    def compute_loss(self):

        self.max_time = tf.shape(self.src)[1]

        distance = tf.pow(self.src, 2)
        distance = tf.reduce_sum(distance, axis=-1)
        scores = 1 / tf.pow(distance, 0.5)

        zero = tf.zeros_like(scores)
        scores = tf.where(tf.math.is_inf(scores), x=zero, y=scores)

        base = tf.reduce_sum(scores, axis=-1)
        base = tf.tile(tf.expand_dims(base, 1), [tf.constant(1), self.max_time])
        scores = scores / base

        self.multilabel = tf.reduce_sum(tf.one_hot(self.trgt, self.max_time + 1, axis=-1), 1)
        self.multilabel = self.multilabel[:, 1:]  # get ride of index = 0 column

        p = tf.nn.sigmoid(self.logits)
        loss1 = -tf.compat.v1.log(p + 1e-10)
        loss1 = tf.multiply(loss1, scores)
        loss1 = tf.multiply(loss1, self.multilabel)
        loss2 = -tf.compat.v1.log(1 - p + 1e-10)
        loss2 = tf.multiply(loss2, scores)
        loss2 = tf.multiply(loss2, 1 - self.multilabel)
        loss = loss1 + loss2
        trgt_weight = tf.sequence_mask(self.src_length, self.max_time, dtype=self.logits.dtype)
        self.loss = tf.reduce_sum(loss * trgt_weight) / tf.compat.v1.to_float(self.batch_size)

        return self.loss
