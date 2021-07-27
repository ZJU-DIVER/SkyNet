from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from attentionLayer import Wrapper


class PointerNet:
    def __init__(self, params, mode, iter):
        self.params = params
        self.mode = mode
        self.iter = iter
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

        self.src, self.src_length, self.trgt_in, self.trgt, self.trgt_length = self.iter
        self.batch_size = tf.shape(self.src_length)[0]
        enc_outputs, enc_state = self._build_encoder()
        self.logits, self.predicted = self._build_decoder(enc_outputs, enc_state)

    def _build_encoder(self):  # same part
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
            cell, decoder_initial_state = self._build_decoder_cell(enc_outputs, enc_state,
                                                                   self.src_length)  # cell is a LSTM layer

            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                dec_inputs = self.trgt_in  # decoder input, which is [0,0] + [the value of skyline points]  and + [0,0]

                if self.params.time_major:
                    dec_inputs = tf.transpose(dec_inputs)

                helper = tf.contrib.seq2seq.TrainingHelper(dec_inputs,
                                                           self.trgt_length)  # [0,0] + [the value of skyline points]  ///  m+1

                decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, decoder_initial_state)

                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, scope=decoder_scope)
                logits = outputs.rnn_output
                sample_ids = outputs.sample_id  # sample_ids is only one index len(sample_ids) = 1

            else:
                def initialize_fn():
                    finished = tf.tile([False], [self.batch_size])
                    start_inputs = tf.fill([self.batch_size, self.params.input_dim], 0.0)
                    return (finished, start_inputs)

                def sample_fn(time, outputs, state):
                    del time, state
                    sample_ids = tf.argmax(outputs, axis=-1, output_type=tf.int32)
                    return sample_ids

                def next_inputs_fn(time, outputs, state, sample_ids):
                    del outputs
                    finished1 = tf.greater(time, tf.cast(12, tf.int32))
                    # finished2 = tf.equal(sample_ids, 0) #pointing to first input point
                    # finished = tf.logical_or(finished1, finished2)
                    idx = tf.reshape(tf.stack([tf.range(self.batch_size, dtype=tf.int32), sample_ids], axis=1),
                                     (-1, 2))
                    next_inputs = tf.gather_nd(self.src, idx)
                    return (finished1, next_inputs, state)

                helper = tf.contrib.seq2seq.CustomHelper(initialize_fn, sample_fn, next_inputs_fn)
                decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, decoder_initial_state)
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, scope=decoder_scope)
                logits = outputs.rnn_output
                sample_ids = outputs.sample_id

        return logits, sample_ids

    def _build_decoder_cell(self, enc_outputs, enc_state, src_length):  # same part
        if self.params.time_major:
            memory = tf.transpose(enc_outputs, [1, 0, 2])
        else:
            memory = enc_outputs

        cell = tf.contrib.rnn.LSTMBlockCell(self.params.num_units)

        cell = Wrapper(cell, self.params.num_units, memory, memory_sequence_length=src_length, name="attention")

        decoder_initial_state = cell.zero_state(self.batch_size, dtype=tf.float32).clone(
            cell_state=enc_state)

        return cell, decoder_initial_state

    def compute_loss(self):
        trgt_out = self.trgt
        self.max_time = tf.shape(trgt_out)[1]
        self.crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=trgt_out, logits=self.logits)
        trgt_weight = tf.sequence_mask(self.trgt_length, self.max_time, dtype=self.logits.dtype)
        self.loss = tf.reduce_sum(self.crossent * trgt_weight) / tf.to_float(self.batch_size)
        return self.loss
