import tensorflow as tf
import numpy as np

def norm_vis(T, which_words, data_type, summaries):
    """
    To visualize norm of fillers and roles vectors.
    This function is aimed to be used under "evaluator.py"
    """
    for word in which_words:
        # question side
        name = "{}/norm/fw_u_aF/[0]["+str(word)+"]"
        summaries.append( tf.Summary(
            value=[tf.Summary.Value(tag=name.format(data_type),
                                    simple_value=float(np.linalg.norm(T["fw_u_aF"][0][word])))]) )
        name = "{}/norm/fw_u_aR/[0]["+str(word)+"]"
        summaries.append( tf.Summary(
            value=[tf.Summary.Value(tag=name.format(data_type),
                                    simple_value=float(np.linalg.norm(T["fw_u_aR"][0][word])))]) )
        name = "{}/norm/bw_u_aF/[0][" + str(word) + "]"
        summaries.append( tf.Summary(
            value=[tf.Summary.Value(tag=name.format(data_type),
                                    simple_value=float(np.linalg.norm(T["bw_u_aF"][0][word])))]) )
        name = "{}/norm/bw_u_aR/[0]["+str(word)+"]"
        summaries.append( tf.Summary(
            value=[tf.Summary.Value(tag=name.format(data_type),
                                    simple_value=float(np.linalg.norm(T["bw_u_aR"][0][word])))]) )
        # context side
        name = "{}/norm/fw_h_aF/[0][0]["+str(word)+"]"
        summaries.append( tf.Summary(
            value=[tf.Summary.Value(tag=name.format(data_type),
                                    simple_value=float(np.linalg.norm(T["fw_h_aF"][0][0][word])))]) )
        name = "{}/norm/fw_h_aR/[0][0]["+str(word)+"]"
        summaries.append( tf.Summary(
            value=[tf.Summary.Value(tag=name.format(data_type),
                                    simple_value=float(np.linalg.norm(T["fw_h_aR"][0][0][word])))]) )
        name = "{}/norm/bw_h_aF/[0][0]["+str(word)+"]"
        summaries.append( tf.Summary(
            value=[tf.Summary.Value(tag=name.format(data_type),
                                    simple_value=float(np.linalg.norm(T["bw_h_aF"][0][0][word])))]) )
        name = "{}/norm/bw_h_aR/[0][0]["+str(word)+"]"
        summaries.append( tf.Summary(
            value=[tf.Summary.Value(tag=name.format(data_type),
                                    simple_value=float(np.linalg.norm(T["bw_h_aR"][0][0][word])))]) )
    return summaries

def norm_vis2(T, which_words):
    """
    To visualize norm of fillers and roles vectors.
    This function is aimed to be used under "model.py" and called in "main.py" through "trainer.py".
    """
    for word in which_words:
        # question side
        name = "{}/norm/fw_u_aF/[0]["+str(word)+"]"
        tf.summary.scalar(name, euclidean_norm(T["fw_u_aF"][0][word]))
        name = "{}/norm/fw_u_aR/[0]["+str(word)+"]"
        tf.summary.scalar(name, euclidean_norm(T["fw_u_aR"][0][word]))
        name = "{}/norm/bw_u_aF/[0][" + str(word) + "]"
        tf.summary.scalar(name, euclidean_norm(T["bw_u_aF"][0][word]))
        name = "{}/norm/bw_u_aR/[0]["+str(word)+"]"
        tf.summary.scalar(name, euclidean_norm(T["bw_u_aR"][0][word]))
        # context side
        name = "{}/norm/fw_h_aF/[0][0]["+str(word)+"]"
        tf.summary.scalar(name, euclidean_norm(T["fw_h_aF"][0][0][word]))
        name = "{}/norm/fw_h_aR/[0][0]["+str(word)+"]"
        tf.summary.scalar(name, euclidean_norm(T["fw_h_aR"][0][0][word]))
        name = "{}/norm/bw_h_aF/[0][0]["+str(word)+"]"
        tf.summary.scalar(name, euclidean_norm(T["bw_h_aF"][0][0][word]))
        name = "{}/norm/bw_h_aR/[0][0]["+str(word)+"]"
        tf.summary.scalar(name, euclidean_norm(T["bw_h_aR"][0][0][word]))

def sparsity_vis(T, which_words):
    """
    To visualize sparsity of fillers and roles vectors.
    """
    for word in which_words:
        # question side
        name = "{}/sparsity/fw_u_aF/[0]["+str(word)+"]"
        tf.summary.scalar(name, tf.nn.zero_fraction(T["fw_u_aF"][0][word]))
        name = "{}/sparsity/fw_u_aR/[0]["+str(word)+"]"
        tf.summary.scalar(name, tf.nn.zero_fraction(T["fw_u_aR"][0][word]))
        name = "{}/sparsity/bw_u_aF/[0][" + str(word) + "]"
        tf.summary.scalar(name, tf.nn.zero_fraction(T["bw_u_aF"][0][word]))
        name = "{}/sparsity/bw_u_aR/[0]["+str(word)+"]"
        tf.summary.scalar(name, tf.nn.zero_fraction(T["bw_u_aR"][0][word]))
        # context side
        name = "{}/sparsity/fw_h_aF/[0][0]["+str(word)+"]"
        tf.summary.scalar(name, tf.nn.zero_fraction(T["fw_h_aF"][0][0][word]))
        name = "{}/sparsity/fw_h_aR/[0][0]["+str(word)+"]"
        tf.summary.scalar(name, tf.nn.zero_fraction(T["fw_h_aR"][0][0][word]))
        name = "{}/sparsity/bw_h_aF/[0][0]["+str(word)+"]"
        tf.summary.scalar(name, tf.nn.zero_fraction(T["bw_h_aF"][0][0][word]))
        name = "{}/sparsity/bw_h_aR/[0][0]["+str(word)+"]"
        tf.summary.scalar(name, tf.nn.zero_fraction(T["bw_h_aR"][0][0][word]))

def tensorHist(T, which_words):
    """
    To visualize distribution of units activations of fillers and roles vectors.
    """
    for word in which_words:
        # question side
        name = "fw_u_aF/[0][" + str(word) + "]"
        tf.summary.histogram(name, T["fw_u_aF"][0][word])
        name = "fw_u_aR/[0][" + str(word) + "]"
        tf.summary.histogram(name, T["fw_u_aR"][0][word])
        name = "bw_u_aF/[0][" + str(word) + "]"
        tf.summary.histogram(name, T["bw_u_aF"][0][word])
        name = "bw_u_aR/[0][" + str(word) + "]"
        tf.summary.histogram(name, T["bw_u_aR"][0][word])
        # context side
        name = "fw_h_aF/[0][0][" + str(word) + "]"
        tf.summary.histogram(name, T["fw_h_aF"][0][0][word])
        name = "fw_h_aR/[0][0][" + str(word) + "]"
        tf.summary.histogram(name, T["fw_h_aR"][0][0][word])
        name = "bw_h_aF/[0][0][" + str(word) + "]"
        tf.summary.histogram(name, T["bw_h_aF"][0][0][word])
        name = "bw_h_aR/[0][0][" + str(word) + "]"
        tf.summary.histogram(name, T["bw_h_aR"][0][0][word])

def euclidean_norm(a):
    a = tf.square(a)
    a = tf.reduce_sum(a)
    return tf.sqrt(a)