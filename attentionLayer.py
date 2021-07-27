import tensorflow as tf

class Wrapper(tf.contrib.seq2seq.AttentionWrapper):
  """Customized AttentionWrapper for PointerNet."""

  def __init__(self,cell, attention_size, memory, memory_sequence_length=None,initial_cell_state=None,name=None):



    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(attention_size, memory, memory_sequence_length=memory_sequence_length, probability_fn=lambda x: x)

    # According to the paper, no need to concatenate the input and attention
    # Therefore, we make cell_input_fn to return input only
    cell_input_fn=lambda input, attention: input

    super(Wrapper, self).__init__(cell,
                                         attention_mechanism=attention_mechanism,
                                         attention_layer_size=None,
                                         alignment_history=False,
                                         cell_input_fn=cell_input_fn,
                                         output_attention=True,
                                         initial_cell_state=initial_cell_state,
                                         name=name)
  @property
  def output_size(self):
    return self.state_size.alignments

  def call(self, inputs, state):
    _, next_state = super(Wrapper, self).call(inputs, state)
    return next_state.alignments, next_state

