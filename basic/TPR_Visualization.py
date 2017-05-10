import tensorflow as tf
import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nltk.tag import StanfordPOSTagger
from basic.clustering import do_cluster
from sklearn.metrics.pairwise import cosine_similarity

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

def sentence2role_filler_vis(data_set, idxs, tensor_dict, config, tensor2vis, spans):
    if tensor2vis in ["fw_u_aR", "bw_u_aR", "fw_u_aF", "bw_u_aF"]: # Question side
        question = data_set.data["q"][config.which_q]
        q_len = len(question)
        T = tensor_dict[tensor2vis][config.which_q][:q_len]
        # Visualize each question
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel(tensor2vis)
        ax.set_ylabel("QUESTION")
    else: # Context side
        if config.Just_Answer_vis:
            ans_start = spans[config.which_q][0][1]
            ans_end = spans[config.which_q][1][1]
            q_len = ans_end - ans_start
            question = data_set.data["x"][config.which_q][0][ans_start:ans_end]
            T = tensor_dict[tensor2vis][config.which_q][0][ans_start:ans_end]
            # Visualize each answer
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel(tensor2vis + "\n Question: " + str(data_set.data["q"][config.which_q]), fontsize=8)
            ax.set_ylabel("PREDICTED ANSWER BY TRAINED MODEL")
        else:
            question = data_set.data["x"][config.which_q][0]
            q_len = len(question)
            if q_len > 15:
                q_len = 15
            T = tensor_dict[tensor2vis][config.which_q][0][:q_len]
            # Visualize each question
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel(tensor2vis + "\n Question: " + str(data_set.data["q"][config.which_q]), fontsize=8)
            ax.set_ylabel("FIRST 15 WORDS OF THE PASSAGE.")


    cax = ax.matshow(T, interpolation='none', cmap=plt.cm.ocean_r)
    fig.colorbar(cax)
    ax.set_yticklabels([""] + question, fontsize=8)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    # Minor ticks
    if tensor2vis in ["fw_u_aR", "bw_u_aR", "fw_h_aR", "bw_h_aR"]:
        width = config.nRoles
    else:
        width = config.nSymbols
    ax.set_xticks(np.arange(-.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-.5, q_len, 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=2)
    plt.show()
    forceAspect(ax, aspect=1)
    plt.savefig(config.TPRvis_dir + "/dataID_" + str(idxs[config.which_q]) + "_" + tensor2vis + ".png")

def write2csv(data_set, idxs, tensor_dict, config, tensor2vis):
    fl = open(config.TPRvis_dir + "/" + tensor2vis + "_nRoles_" + str(config.nRoles) + "_test_set.csv", "a")
    nQuestions = len(data_set.data["q"])
    for which_q in range(nQuestions):
        if tensor2vis in ["fw_u_aR", "bw_u_aR", "fw_u_aF", "bw_u_aF"]: # Question side
            question = data_set.data["q"][which_q]
            q_len = len(question)
            T = tensor_dict[tensor2vis][which_q][:q_len]
        out = [[]]*q_len
        for i in range(q_len):
            out[i] = [idxs[which_q]] + [question[i]] + T[i].tolist()
        writer = csv.writer(fl)
        for row in out:
            writer.writerow(row)
    fl.close()

def write2csv_withPOS(data_set, idxs, tensor_dict, config, tensor2vis, posBatch):
    fl = open(config.TPRvis_dir + "/" + tensor2vis + "_nRoles_" + str(config.nRoles) + "_test_set.csv", "a")
    nQuestions = len(data_set.data["q"])
    for which_q in range(nQuestions):
        if tensor2vis in ["fw_u_aR", "bw_u_aR", "fw_u_aF", "bw_u_aF"]: # Question side
            question = data_set.data["q"][which_q]
            q_len = len(question)
            T = tensor_dict[tensor2vis][which_q][:q_len]
        out = [[]]*q_len
        pos = posBatch[which_q]
        for i in range(q_len):
            out[i] = [idxs[which_q]] + [question[i]] + [pos[i][1]] + T[i].tolist()
        writer = csv.writer(fl)
        for row in out:
            writer.writerow(row)
    fl.close()

def getPOS_fromBatch(data_set, config):
    nQuestions = len(data_set.data["q"])
    out = [[]] * nQuestions
    pos_tagger = StanfordPOSTagger(config.stanford_model, config.stanford_jar, encoding='utf8')
    for which_q in range(nQuestions):
        if config.QuestionSideVis: # Question side
            question = data_set.data["q"][which_q]
        pos = pos_tagger.tag(question)
        out[which_q] = pos
    return out

def cluster(num, X, config):
    do_cluster(num, X, config)
    return True

def do_Fa_F_vis(data_set, idxs, tensor_dict, config, tensor2vis, F_name):
    """
    This function finds the cosine similarity between F * a_F(t) for each word and each filler embedding vector (each
    column of F) where t is the word number in the sentence. It creates 3 outputs:
    1. One excel file including the cosine similarity scores where each row shows one word and each column is a filler.
    2. One excel file where assigned filler ID per word is written, i.e., one row per word + its assigned filler ID.
    The word-filler assignment is based on the maximum cosine similarity.
    3. Several excel files, where per filler one excel file is created.
    All words assigned to that filler are in the excel file corresponding to it along
    with the cosine similarity, the index of the word in the sentence, and, the index of the sentence in the whole dataset.
    The word-filler assignment is done based on maximum cosine similarity.
    :param data_set: contains the input sentences.
    :param idxs: index of the sentence in the dataset.
    :param tensor_dict: contains the tensors we need, e.g., learned F matrix and a_F(t) vectors.
    :param config: includes config & settings
    :param tensor2vis: the name of tensor we want to visualize.
    :param F_name: the of tensor that contains the trained F matrix.
    :return:
    """
    f2 = open(config.TPRvis_dir + "/" + tensor2vis + "_vis_Fa_F_test_set.csv", "a")
    f3 = open(config.TPRvis_dir + "/" + tensor2vis + "_MAX_vis_Fa_F_fillerID_per_word_test_set.csv", "a")
    writer2 = csv.writer(f2)
    writer3 = csv.writer(f3)
    F = tensor_dict[F_name]
    nQuestions = len(data_set.data["q"])
    for which_q in range(nQuestions):
        if tensor2vis in ["fw_u_aR", "bw_u_aR", "fw_u_aF", "bw_u_aF"]: # Question side, context side is to do.
            question = data_set.data["q"][which_q]
            q_len = len(question)
            T = tensor_dict[tensor2vis][which_q][:q_len]
        out2 = [[]] * q_len
        out3 = [[]] * q_len
        for i in range(q_len):
            F_embed_vec = T[i].dot(F)
            similarities = cosine_similarity( F , F_embed_vec.reshape(1,-1) )
            similarities = similarities.squeeze() # remove the unnecessary extra dimension from sklearn.
            similarities = np.round(similarities, decimals=4)
            max_idx = np.argmax(similarities) # index of the filler with maximum similarity.
            max_val = max(similarities)
            out = [question[i]] + [max_val] + [i] + [idxs[which_q]]
            out2[i] = [idxs[which_q]] + [question[i]] + similarities.tolist()
            out3[i] = [idxs[which_q]] + [question[i]] + [max_idx] + [max_val]
            fl = open(config.TPRvis_dir + "/" + tensor2vis + "_MAX_vis_Fa_F_test_set_" + "Filler_" + str(max_idx) + ".csv", "a")
            writer1 = csv.writer(fl)
            writer1.writerow(out)
            fl.close()
        for row in out2:
            writer2.writerow(row)
        for row in out3:
            writer3.writerow(row)
    f2.close()
    f3.close()

def B_avg(data_set, tensor_dict, tensor2vis, B):
    """
    This is basically B = sum( a_F(t) * (a_R(t))^T ) / number_of_tokens_in_test_set,
    i.e., binding matrix averaged over whole test set.
    :param data_set: contains the input sentences.
    :param tensor_dict: contains the tensors we need, e.g., a_R(t) and a_F(t) vectors.
    :param tensor2vis: the name of tensor(s) we want to visualize.
    :param B: Binding matrix summed till this minibatch. It is summed over the whole test set
    and then divided by total number of tokens in the test set.
    :return:
            B from this minibatch added to the B passed to this function. Later the averaged B is written
            to a csv file in "main.py".
    """
    aF_name, aR_name = tensor2vis
    assert aF_name in ["fw_u_aR", "bw_u_aR", "fw_u_aF", "bw_u_aF"] # Question side, context side is to do.
    assert aR_name in ["fw_u_aR", "bw_u_aR", "fw_u_aF", "bw_u_aF"] # Question side, context side is to do.
    nQuestions = len(data_set.data["q"])
    for which_q in range(nQuestions):
        question = data_set.data["q"][which_q]
        q_len = len(question)
        aF = tensor_dict[aF_name][which_q][:q_len]
        aR = tensor_dict[aR_name][which_q][:q_len]
        for i in range(q_len):
            B += np.expand_dims(aF[i], axis=1).dot(np.expand_dims(aR[i], axis=0))
    return B
