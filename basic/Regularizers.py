import tensorflow as tf

def Reg_eq_1_4(a):
    """
    Regularization in eq. (1.4).
    First item: a^2 * (1-a)^2 = a^2 * (1 + a^2 - 2*a) = a^2 + a^2*a^2 - 2*a*a^2
    There is also an a^2 in the second item. Therefore we have 3 a^2 terms for which
    it is more computationally efficient to calculate a^2 just once (represented as a2 below).
    :param a: input tensor of shape [batchsize, time, dim]. E.g., for "fw_u_aF" the dim is number of fillers.
    :return: One item in regularization term in eq. (1.4) of TPR_ver0_0.pdf document. The output shape is [batchsize].
    """
    a2 = tf.multiply(a, a)
    item1 = a2 + tf.multiply(a2, a2) - 2 * tf.multiply(a, a2)
    item1 = tf.reduce_sum(item1, 2)
    tmp = tf.reduce_sum(a2, 2) - 1
    item2 = tf.multiply(tmp, tmp)
    out = tf.reduce_sum(item1 + item2, 1)
    return out