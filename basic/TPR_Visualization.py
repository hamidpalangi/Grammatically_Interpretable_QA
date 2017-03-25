import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


def forceAspect(ax, aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)

def sentence2role_vis(data_set, idxs, tensor_dict, config, tensor2vis):
    question = data_set.data["q"][config.which_q]
    q_len = len(question)
    fw_u_aR = tensor_dict[tensor2vis][config.which_q][:q_len]
    # Visualize each question
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(tensor2vis)
    cax = ax.matshow(fw_u_aR, interpolation='none', cmap=plt.cm.ocean_r)
    fig.colorbar(cax)
    ax.set_yticklabels([""] + question, fontsize=8)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    # Minor ticks
    ax.set_xticks(np.arange(-.5, config.nRoles, 1), minor=True)
    ax.set_yticks(np.arange(-.5, q_len, 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=2)
    plt.show()
    forceAspect(ax, aspect=1)
    plt.savefig(config.TPRvis_dir + "/dataID_" + str(idxs[config.which_q]) + "_" + tensor2vis + ".png")