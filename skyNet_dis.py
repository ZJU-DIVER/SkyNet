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
            self.mode = tf.contrib.learn.ModeKeys.TRAIN
        elif self.mode == 'infer':
            self.mode = tf.contrib.learn.ModeKeys.INFER
        elif self.mode == 'eval':
            self.mode = tf.contrib.learn.ModeKeys.EVAL
        else:
            raise ValueError("Please choose mode: train, eval or infer")
        tf.logging.info("Building {} model...".format(self.mode))

        self.src, self.src_length, self.trgt, self.src_real, self.trgt_real, self.src_real_length = self.iter
        self.batch_size = tf.shape(self.src_length)[0]
        enc_outputs, enc_state = self._build_encoder()
        self.logits = self._build_decoder(enc_outputs, enc_state)
        self.p = tf.nn.sigmoid(self.logits)

    def _build_encoder(self):
        dropout = self.params.dropout if self.mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0

        with tf.variable_scope("encoder") as scope:
            cell = tf.contrib.rnn.LSTMBlockCell(self.params.num_units)

            if dropout > 0.0:
                cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=(1.0 - dropout))

            enc_outputs, enc_state = tf.nn.dynamic_rnn(cell, self.src, sequence_length=self.src_length,
                                                       dtype=tf.float32)

        return enc_outputs, enc_state

    def _build_decoder(self, enc_outputs, enc_state):
        with tf.variable_scope("decoder") as decoder_scope:
            cell, decoder_initial_state = self._build_decoder_cell(enc_outputs, enc_state, self.src_length)

            initial_inputs = tf.fill([self.batch_size, self.params.input_dim], 0.0)
            logits, _ = cell.call(inputs=initial_inputs, state=decoder_initial_state)
        return logits

    def _build_decoder_cell(self, enc_outputs, enc_state, src_length):
        if self.params.time_major:
            memory = tf.transpose(enc_outputs, [1, 0, 2])
        else:
            memory = enc_outputs

        cell = tf.contrib.rnn.LSTMBlockCell(self.params.num_units)

        cell = Wrapper(cell, self.params.num_units, memory, memory_sequence_length=src_length, name="attention")

        decoder_initial_state = cell.zero_state(self.batch_size, dtype=tf.float32).clone(cell_state=enc_state)

        return cell, decoder_initial_state

    def compute_loss(self):
        distance = tf.pow(self.src_real, 2)
        distance = tf.reduce_sum(distance, axis=-1)
        scores = 1 / tf.pow(distance, 4)
        max_time = tf.shape(self.src_real)[1]
        base = tf.cast(max_time / 100, tf.float32)
        scores = scores / (base + scores)
        sky = tf.reduce_sum(tf.one_hot(self.trgt_real - 1, max_time, axis=-1, dtype=tf.float32), 1)
        scores = tf.maximum(scores, sky)

        p = tf.nn.sigmoid(self.logits)
        p = p[:, 1:]
        loss1 = -tf.log(p + 1e-10)
        loss1 = tf.multiply(loss1, scores)
        loss2 = -tf.log(1 - p + 1e-10)
        loss2 = tf.multiply(loss2, 1 - scores)
        loss = loss1 + loss2
        trgt_weight = tf.sequence_mask(self.src_real_length, max_time, dtype=self.logits.dtype)
        self.loss = tf.reduce_sum(loss * trgt_weight) / tf.to_float(self.batch_size)

        return self.loss
