import os

import tensorflow as tf

from basic.main import main as m

flags = tf.app.flags

# Names and directories
flags.DEFINE_string("model_name", "basic", "Model name [basic]")
flags.DEFINE_string("data_dir", "data/squad", "Data dir [data/squad]")
flags.DEFINE_string("run_id", "0", "Run ID [0]")
flags.DEFINE_string("out_base_dir", "out", "out base dir [out]")
flags.DEFINE_string("forward_name", "single", "Forward name [single]")
flags.DEFINE_string("answer_path", "", "Answer path []")
flags.DEFINE_string("eval_path", "", "Eval path []")
flags.DEFINE_string("load_path", "", "Load path []")
flags.DEFINE_string("shared_path", "", "Shared path []")

# Device placement
flags.DEFINE_string("device", "/cpu:0", "default device for summing gradients. [/cpu:0]")
flags.DEFINE_string("device_type", "gpu", "device for computing gradients (parallelization). cpu | gpu [gpu]")
flags.DEFINE_integer("num_gpus", 1, "num of gpus or cpus for computing gradients [1]")

# Essential training and test options
flags.DEFINE_string("mode", "test", "trains | test | forward [test]")
flags.DEFINE_boolean("load", True, "load saved data? [True]")
flags.DEFINE_bool("single", False, "supervise only the answer sentence? [False]")
flags.DEFINE_boolean("debug", False, "Debugging mode? [False]")
flags.DEFINE_bool('load_ema', True, "load exponential average of variables when testing?  [True]")
flags.DEFINE_bool("eval", True, "eval? [True]")

# Training / test parameters
# flags.DEFINE_integer("batch_size", 60, "Batch size [60]")
flags.DEFINE_integer("batch_size", 40, "Batch size [40]") # due to memory limitation when using TPR.
flags.DEFINE_integer("val_num_batches", 100, "validation num batches [100]")
flags.DEFINE_integer("test_num_batches", 0, "test num batches [0]")
flags.DEFINE_integer("num_epochs", 12, "Total number of epochs for training [12]")
flags.DEFINE_integer("num_steps", 20000, "Number of steps [20000]")
flags.DEFINE_integer("load_step", 0, "load step [0]")
flags.DEFINE_float("init_lr", 0.5, "Initial learning rate [0.5]")
flags.DEFINE_float("input_keep_prob", 0.8, "Input keep prob for the dropout of LSTM weights [0.8]")
flags.DEFINE_float("keep_prob", 0.8, "Keep prob for the dropout of Char-CNN weights [0.8]")
flags.DEFINE_float("wd", 0.0, "L2 weight decay for regularization [0.0]")
flags.DEFINE_integer("hidden_size", 100, "Hidden size [100]")
flags.DEFINE_integer("char_out_size", 100, "char-level word embedding size [100]")
flags.DEFINE_integer("char_emb_size", 8, "Char emb size [8]")
flags.DEFINE_string("out_channel_dims", "100", "Out channel dims of Char-CNN, separated by commas [100]")
flags.DEFINE_string("filter_heights", "5", "Filter heights of Char-CNN, separated by commas [5]")
flags.DEFINE_bool("finetune", False, "Finetune word embeddings? [False]")
flags.DEFINE_bool("highway", True, "Use highway? [True]")
flags.DEFINE_integer("highway_num_layers", 2, "highway num layers [2]")
flags.DEFINE_bool("share_cnn_weights", True, "Share Char-CNN weights [True]")
flags.DEFINE_bool("share_lstm_weights", True, "Share pre-processing (phrase-level) LSTM weights [True]")
flags.DEFINE_float("var_decay", 0.999, "Exponential moving average decay for variables [0.999]")

# Optimizations
flags.DEFINE_bool("cluster", False, "Cluster data for faster training [False]")
flags.DEFINE_bool("len_opt", False, "Length optimization? [False]")
flags.DEFINE_bool("cpu_opt", False, "CPU optimization? GPU computation can be slower [False]")

# Logging and saving options
flags.DEFINE_boolean("progress", True, "Show progress? [True]")
flags.DEFINE_integer("log_period", 100, "Log period [100]")
flags.DEFINE_integer("eval_period", 1000, "Eval period [1000]")
flags.DEFINE_integer("save_period", 1000, "Save Period [1000]")
flags.DEFINE_integer("max_to_keep", 20, "Max recent saves to keep [20]")
flags.DEFINE_bool("dump_eval", True, "dump eval? [True]")
flags.DEFINE_bool("dump_answer", True, "dump answer? [True]")
flags.DEFINE_bool("vis", False, "output visualization numbers? [False]")
flags.DEFINE_bool("dump_pickle", True, "Dump pickle instead of json? [True]")
flags.DEFINE_float("decay", 0.9, "Exponential moving average decay for logging values [0.9]")

# Thresholds for speed and less memory usage
flags.DEFINE_integer("word_count_th", 10, "word count th [100]")
flags.DEFINE_integer("char_count_th", 50, "char count th [500]")
flags.DEFINE_integer("sent_size_th", 400, "sent size th [64]")
flags.DEFINE_integer("num_sents_th", 8, "num sents th [8]")
flags.DEFINE_integer("ques_size_th", 30, "ques size th [32]")
flags.DEFINE_integer("word_size_th", 16, "word size th [16]")
flags.DEFINE_integer("para_size_th", 256, "para size th [256]")

# Advanced training options
flags.DEFINE_bool("lower_word", True, "lower word [True]")
flags.DEFINE_bool("squash", False, "squash the sentences into one? [False]")
flags.DEFINE_bool("swap_memory", True, "swap memory? [True]")
flags.DEFINE_string("data_filter", "max", "max | valid | semi [max]")
flags.DEFINE_bool("use_glove_for_unk", True, "use glove for unk [False]")
flags.DEFINE_bool("known_if_glove", True, "consider as known if present in glove [False]")
flags.DEFINE_string("logit_func", "tri_linear", "logit func [tri_linear]")
flags.DEFINE_string("answer_func", "linear", "answer logit func [linear]")
flags.DEFINE_string("sh_logit_func", "tri_linear", "sh logit func [tri_linear]")

# Ablation options
flags.DEFINE_bool("use_char_emb", True, "use char emb? [True]")
flags.DEFINE_bool("use_word_emb", True, "use word embedding? [True]")
flags.DEFINE_bool("q2c_att", True, "question-to-context attention? [True]")
flags.DEFINE_bool("c2q_att", True, "context-to-question attention? [True]")
flags.DEFINE_bool("dynamic_att", False, "Dynamic attention [False]")

# TPR parameters
flags.DEFINE_bool("LSTMandTPR", False, "Use TPR + LSTM concatenated in phrase embedding layer?"
                                       "Note that here TPR and LSTM are two independent cells. [True]")
flags.DEFINE_bool("justTPR", False, "Use just TPR not LSTM in phrase embedding layer? [False]")
flags.DEFINE_bool("justLSTM", False, "Use just LSTM not TPR in phrase embedding layer? [False]")
flags.DEFINE_bool("TPRLSTMCell", False, "Use newly defined mixed TPR-LSTM cell in phrase embedding layer? [False]")
flags.DEFINE_bool("share_tpr_weights", True, "Share TPR weights between query side and text side? [True]")
flags.DEFINE_integer("nSymbols", 100, "# of Symbols in TPR [100]")
flags.DEFINE_integer("dSymbols", 10, "size of Symbol embedding in TPR [10]")
flags.DEFINE_integer("nRoles", 20, "# of Roles in TPR [20]")
flags.DEFINE_integer("dRoles", 10, "size of Role embedding in TPR [10]")
flags.DEFINE_bool("TPRregularizer1", False, "Use regularization in eq. (1.4) of 'TPR_ver0_0.pdf'? [False]")
flags.DEFINE_float("cF", 0.00001, "Filler regularization weight [0.00001]")
flags.DEFINE_float("cR", 0.00001, "Role regularization weight [0.00001]")

# Resume training
flags.DEFINE_bool("resumeTrain", False, "Resume training from the iteration specified in the checkpoint? [False]")

# TPR Visualization Parameters [also check parameters under "Logging and saving options" above]
flags.DEFINE_string("which_words", "1,5", "which words to track for visualization in question / context side [1,5]")
flags.DEFINE_bool("TPRvis", False, "TPR visualization? [False]")
flags.DEFINE_bool("JustLastIterVis", False, "Show activations of just last iteration? [False]")
flags.DEFINE_integer("which_q", 0, "which member from each minibatch in test to use for aR and aF visualization [0]")
flags.DEFINE_string("which_tensors2vis", "fw_u_aR,bw_u_aR,fw_u_aF,bw_u_aF",
                    "which tensors to visualize ['fw_u_aR','bw_u_aR','fw_u_aF','bw_u_aF']")
flags.DEFINE_bool("QuestionSideVis", True, "Is this Question side visualization or Context side? "
                                           "Currently just used in POS tagger ('getPOS_fromBatch' in 'TPR_Visualization.py')[True]")
flags.DEFINE_bool("Just_Answer_vis", True,
                  "Just visualize the answer part of passage determined by start and end words."
                  "If False, will visualize just first 15 words of passage. [True]")
flags.DEFINE_bool("write2csv", False, "Write aF and aR per word in the test set to a csv file. [False]")
flags.DEFINE_bool("POStagger", False, "Write aF and aR per word + their POS tags from "
                                      "Stanford parser in the test set to a csv file. [False]")
flags.DEFINE_string("stanford_jar",
                    "/home/hpalangi/QA/TPR_Stuff/Codes/TPR_ver1.0/stanford-postagger-2016-10-31/stanford-postagger.jar",
                    "For POS tagger.")
flags.DEFINE_string("stanford_model",
                    "/home/hpalangi/QA/TPR_Stuff/Codes/TPR_ver1.0/stanford-postagger-2016-10-31/models/english-bidirectional-distsim.tagger",
                    "For POS tagger.")
flags.DEFINE_bool("F_vis", False, "visualize fillers matrix F. [False]")
flags.DEFINE_bool("R_vis", False, "visualize roles matrix R. [False]")
flags.DEFINE_string("nClusters_F", "5", "# of clusters for clustering of trained F in TPR for visualization purposes. "
                                          "Can be a list of number of candidate clusters. Clustering is performed for each "
                                          "one and the best one is selected based on Silhouette score [20]")
flags.DEFINE_string("nClusters_R", "5", "# of clusters for clustering of trained R in TPR for visualization purposes. "
                                         "Can be a list of number of candidate clusters. Clustering is performed for each "
                                         "one and the best one is selected based on Silhouette score [5]")
flags.DEFINE_bool("clustered_F", False, "used internally. [False]")
flags.DEFINE_bool("clustered_R", False, "used internally. [False]")
flags.DEFINE_bool("Fa_F_vis", False, "Finds cosine similarity between each filler (each column of F) and"
                                     "F*a_F for each word. For now writes all of them in an excel file where "
                                     "each row represents a word and each column represents a filler. "
                                     "If set to True, 3 types of outputs are calculated and saved in folder "
                                     "TPRvis in path '<path-to-codes>/out/basic/<run_id_number>/TPRvis':"
                                     "1. Averaged bindning matrix B over the whole dataset."
                                     "2. List of assigned words to each filler based on "
                                     "cosine similarity and simple max rule. One csv file per filler. "
                                     "3. One csv file including all tokens in the test set + the assigned "
                                     "filler to that token.[False]")
flags.DEFINE_integer("nWordsTest", 120950, "Total number of words in the test set [120950]")
flags.DEFINE_bool("EMperQ", False, "Write EM, F1, paragraph, and 3 answers per query for the whole validation set as "
                                   "different columns of an output excel file. [False]")

def main(_):
    config = flags.FLAGS
    config.which_words = [int(w) for w in config.which_words.split(",")]
    config.which_tensors2vis = [tensor for tensor in config.which_tensors2vis.split(",")]
    if config.JustLastIterVis:
        config.log_period = config.num_steps
    config.nClusters_F = [int(i) for i in config.nClusters_F.split(",")]
    config.nClusters_R = [int(i) for i in config.nClusters_R.split(",")]
    config.out_dir = os.path.join(config.out_base_dir, config.model_name, str(config.run_id).zfill(2))

    m(config)

if __name__ == "__main__":
    tf.app.run()
