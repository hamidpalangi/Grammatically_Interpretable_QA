import tensorflow as tf
from my.tensorflow.general import mask

def Reg_eq_1_4(a, Qside, mask_it):
    """
    Regularization in eq. (1.4).
    First item: a^2 * (1-a)^2 = a^2 * (1 + a^2 - 2*a) = a^2 + a^2*a^2 - 2*a*a^2
    There is also an a^2 in the second item. Therefore we have 3 a^2 terms for which
    it is more computationally efficient to calculate a^2 just once (represented as a2 below).

    :param Qside: Bolean, is "a" a tensor from query side or passage side.
    :param a: input tensor of shape:
    1. If Qside is True: [batchsize, time, dim].
    2. 1. If Qside is False: [batchsize, max # of sentences in the passage, time, dim].
    "time" means the length of the sentence.
    "dim": as an example, for "fw_u_aF" the dim is number of fillers.
    :param mask: the mask for input tensor "a".

    :return: One item in regularization term in eq. (1.4) of TPR_ver0_0.pdf document. The output shape is [batchsize].
    """
    a2 = tf.multiply(a, a)
    item1 = a2 + tf.multiply(a2, a2) - 2 * tf.multiply(a, a2)

    if Qside:
        item1 = tf.reduce_sum(item1, 2) # sum over entries of each vector.
        tmp = tf.reduce_sum(a2, 2) - 1
        item2 = tf.multiply(tmp, tmp)
        out = mask(item1 + item2, mask_it)
        out = tf.reduce_sum(out, 1) # sum over time dimension.
    else:
        item1 = tf.reduce_sum(item1, 3) # sum over entries of each vector.
        tmp = tf.reduce_sum(a2, 3) - 1
        item2 = tf.multiply(tmp, tmp)
        out = mask(item1 + item2, mask_it)
        out = tf.reduce_sum(out, 2) # sum over time dimension (length of each sentence).
        out = tf.reduce_sum(out, 1)  # sum over all sentences in the passage (max # of sentences in the passage).
    return out