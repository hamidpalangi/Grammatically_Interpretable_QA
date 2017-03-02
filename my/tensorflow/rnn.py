import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn as _dynamic_rnn, \
    bidirectional_dynamic_rnn as _bidirectional_dynamic_rnn
from tensorflow.python.ops.rnn import bidirectional_rnn as _bidirectional_rnn

from my.tensorflow import flatten, reconstruct


def dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None,
                dtype=None, parallel_iterations=None, swap_memory=False,
                time_major=False, scope=None):
    assert not time_major  # TODO : to be implemented later!
    flat_inputs = flatten(inputs, 2)  # [-1, J, d]
    flat_len = None if sequence_length is None else tf.cast(flatten(sequence_length, 0), 'int64')

    flat_outputs, final_state = _dynamic_rnn(cell, flat_inputs, sequence_length=flat_len,
                                             initial_state=initial_state, dtype=dtype,
                                             parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                                             time_major=time_major, scope=scope)

    outputs = reconstruct(flat_outputs, inputs, 2)
    return outputs, final_state


def bw_dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None,
                   dtype=None, parallel_iterations=None, swap_memory=False,
                   time_major=False, scope=None):
    assert not time_major  # TODO : to be implemented later!

    flat_inputs = flatten(inputs, 2)  # [-1, J, d]
    flat_len = None if sequence_length is None else tf.cast(flatten(sequence_length, 0), 'int64')

    flat_inputs = tf.reverse(flat_inputs, 1) if sequence_length is None \
        else tf.reverse_sequence(flat_inputs, sequence_length, 1)
    flat_outputs, final_state = _dynamic_rnn(cell, flat_inputs, sequence_length=flat_len,
                                             initial_state=initial_state, dtype=dtype,
                                             parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                                             time_major=time_major, scope=scope)
    flat_outputs = tf.reverse(flat_outputs, 1) if sequence_length is None \
        else tf.reverse_sequence(flat_outputs, sequence_length, 1)

    outputs = reconstruct(flat_outputs, inputs, 2)
    return outputs, final_state


def bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=None,
                              initial_state_fw=None, initial_state_bw=None,
                              dtype=None, parallel_iterations=None,
                              swap_memory=False, time_major=False, scope=None):
    assert not time_major

    flat_inputs = flatten(inputs, 2)  # [-1, J, d]
    flat_len = None if sequence_length is None else tf.cast(flatten(sequence_length, 0), 'int64')

    (flat_fw_outputs, flat_bw_outputs), final_state = \
        _bidirectional_dynamic_rnn(cell_fw, cell_bw, flat_inputs, sequence_length=flat_len,
                                   initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw,
                                   dtype=dtype, parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                                   time_major=time_major, scope=scope)

    fw_outputs = reconstruct(flat_fw_outputs, inputs, 2)
    bw_outputs = reconstruct(flat_bw_outputs, inputs, 2)
    # FIXME : final state is not reshaped!
    return (fw_outputs, bw_outputs), final_state


def bidirectional_rnn(cell_fw, cell_bw, inputs,
                      initial_state_fw=None, initial_state_bw=None,
                      dtype=None, sequence_length=None, scope=None):

    flat_inputs = flatten(inputs, 2)  # [-1, J, d]
    flat_len = None if sequence_length is None else tf.cast(flatten(sequence_length, 0), 'int64')

    (flat_fw_outputs, flat_bw_outputs), final_state = \
        _bidirectional_rnn(cell_fw, cell_bw, flat_inputs, sequence_length=flat_len,
                           initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw,
                           dtype=dtype, scope=scope)

    fw_outputs = reconstruct(flat_fw_outputs, inputs, 2)
    bw_outputs = reconstruct(flat_bw_outputs, inputs, 2)
    # FIXME : final state is not reshaped!
    return (fw_outputs, bw_outputs), final_state

# def bidirectional_dynamic_rnn_4reg(cell_fw, cell_bw, inputs, sequence_length=None,
#                               initial_state_fw=None, initial_state_bw=None,
#                               dtype=None, parallel_iterations=None,
#                               swap_memory=False, time_major=False, scope=None):
#     assert not time_major
#
#     flat_inputs = flatten(inputs, 2)  # [-1, J, d]
#     flat_len = None if sequence_length is None else tf.cast(flatten(sequence_length, 0), 'int64')
#
#     ( (flat_fw_aF, flat_fw_aR, flat_fw_outputs) , (flat_bw_aF, flat_bw_aR, flat_bw_outputs) ), final_state = \
#         _bidirectional_dynamic_rnn(cell_fw, cell_bw, flat_inputs, sequence_length=flat_len,
#                                    initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw,
#                                    dtype=dtype, parallel_iterations=parallel_iterations, swap_memory=swap_memory,
#                                    time_major=time_major, scope=scope)
#
#     fw_outputs = reconstruct(flat_fw_outputs, inputs, 2)
#     bw_outputs = reconstruct(flat_bw_outputs, inputs, 2)
#     fw_aF = reconstruct(flat_fw_aF, inputs, 2)
#     bw_aF = reconstruct(flat_bw_aF, inputs, 2)
#     fw_aR = reconstruct(flat_fw_aR, inputs, 2)
#     bw_aR = reconstruct(flat_bw_aR, inputs, 2)
#     # FIXME : final state is not reshaped!
#     return ( (fw_aF, fw_aR, fw_outputs) , (bw_aF, bw_aR, bw_outputs) ), final_state

def bidirectional_dynamic_rnn_4reg(cell_fw, cell_bw, inputs, nSymbols, nRoles, dimT, sequence_length=None,
                              initial_state_fw=None, initial_state_bw=None,
                              dtype=None, parallel_iterations=None,
                              swap_memory=False, time_major=False, scope=None):
    assert not time_major

    flat_inputs = flatten(inputs, 2)  # [-1, J, d]
    flat_len = None if sequence_length is None else tf.cast(flatten(sequence_length, 0), 'int64')

    ( tmp_fw , tmp_bw ), final_state = \
        _bidirectional_dynamic_rnn(cell_fw, cell_bw, flat_inputs, sequence_length=flat_len,
                                   initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw,
                                   dtype=dtype, parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                                   time_major=time_major, scope=scope)

    # **** for later reference: START ****
    # flat_fw_aF, flat_fw_aR, flat_fw_outputs = tf.split_v(value=tmp_fw, size_splits=[nSymbols, nRoles, dimT], split_dim=2)
    # flat_bw_aF, flat_bw_aR, flat_bw_outputs = tf.split_v(value=tmp_bw, size_splits=[nSymbols, nRoles, dimT], split_dim=2)

    # Bsize = tf.shape(tmp_fw)[0]
    # SeqLen = tf.shape(tmp_fw)[1]
    # flat_fw_aF = tf.slice(tmp_fw, begin=[0, 0, 0], size=[Bsize, SeqLen, nSymbols])
    # flat_fw_aR = tf.slice(tmp_fw, begin=[0, 0, nSymbols], size=[Bsize, SeqLen, nRoles])
    # flat_fw_outputs = tf.slice(tmp_fw, begin=[0, 0, nSymbols+nRoles], size=[Bsize, SeqLen, dimT])
    # flat_bw_aF = tf.slice(tmp_bw, begin=[0, 0, 0], size=[Bsize, SeqLen, nSymbols])
    # flat_bw_aR = tf.slice(tmp_bw, begin=[0, 0, nSymbols], size=[Bsize, SeqLen, nRoles])
    # flat_bw_outputs = tf.slice(tmp_bw, begin=[0, 0, nSymbols+nRoles], size=[Bsize, SeqLen, dimT])

    # If you wonder why I am doing something like below which is not as computationally efficient as above commented codes,
    # please take a look at above commented codes and try them for yourself. It gives some errors which is resulted from dynamic
    # time dimension in the tensors.
    # This is just a hack and should be improved for speed in the future which will require modifying
    # internal TensorFlow functions "_bidirectional_dynamic_rnn" and "dynamic_rnn".
    # **** for later reference: END ****

    dim_all = nSymbols + nRoles + dimT
    fw_splits = tf.split(2, dim_all, tmp_fw)
    bw_splits = tf.split(2, dim_all, tmp_bw)
    flat_fw_aF = tf.concat(2, fw_splits[:nSymbols])
    flat_fw_aR = tf.concat(2, fw_splits[nSymbols:nSymbols+nRoles])
    flat_fw_outputs = tf.concat(2, fw_splits[nSymbols+nRoles:dim_all])
    flat_bw_aF = tf.concat(2, bw_splits[:nSymbols])
    flat_bw_aR = tf.concat(2, bw_splits[nSymbols:nSymbols+nRoles])
    flat_bw_outputs = tf.concat(2, bw_splits[nSymbols+nRoles:dim_all])

    fw_outputs = reconstruct(flat_fw_outputs, inputs, 2)
    bw_outputs = reconstruct(flat_bw_outputs, inputs, 2)
    fw_aF = reconstruct(flat_fw_aF, inputs, 2)
    bw_aF = reconstruct(flat_bw_aF, inputs, 2)
    fw_aR = reconstruct(flat_fw_aR, inputs, 2)
    bw_aR = reconstruct(flat_bw_aR, inputs, 2)
    # FIXME : final state is not reshaped!

    return ( (fw_aF, fw_aR, fw_outputs) , (bw_aF, bw_aR, bw_outputs) ), final_state