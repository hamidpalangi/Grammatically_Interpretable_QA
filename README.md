# Tensor Product Representations for Machine Reading Comprehension
- This repository includes all necessary programs to implement and reproduce the results reported in the following [paper](http://arxiv.org/abs/1705.08432) (please use it to refer):
```
@article{TPR_QA,
  author    = {Hamid Palangi and Paul Smolensky and Xiaodong He and Li Deng},
  title     = {Question-Answering with Grammatically-Interpretable Representations},
  year      = {2017},
  url       = {http://arxiv.org/abs/1705.08432},
}
```
- The codes are written on the top of [BIDAF](https://github.com/allenai/bi-att-flow) model which we used as our baseline. We tried to preserve the same code structure as BIDAF.

## FAQ
- **What does this model bring to the table?**
  - We often use CNNs, RNNs (e.g., LSTMs), or more complicated models constructed from these basic neural network structures to create good representations (features) for our target task. But if someone ask us what does each of those entries in the generated representations (feature vectors) mean, we usually do not have a concrete answer. In other words, it is challenging to *interpret* those entries. Here we propose a model that can automatically learn grammatically / semantically interpretable representations through the Question-Answering (QA) task on [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) dataset. More details in the paper. 

- **Does your model performs the best on SQuAD leaderboard?**
  - Not yet. The main goal of this work is not beating the best model on [SQuAD leaderboard](https://rajpurkar.github.io/SQuAD-explorer/) (r-net from MSR Asia at the time of writing this readme document) but to add interpretability to QA systems.

- **I do not want to read the whole paper, can you give me / point me to some examples of interpretable representations your model creates?**
  - Please check section 5 of the paper.
  
- **How easy it is to grab TPRN cells proposed and implemented here and use them in my codes / models?**
  - TPRN is implemented in a way that you can simply add it to your library of different recurrent units (e.g., LSTM, GRU, etc). Simply import a TPRN cell you need from [my/tensorflow/tpr.py](my/tensorflow/tpr.py). Use `TPRCell` if you are not going to use quantization function `Q` in the paper. Use `TPRCellReg` if you need to get `aR` and `aF` vectors out for the quantization function. 

- **How can I run the codes and produce your results by myself?**
  1. Requirements and pre-processing are similar to BIDAF. Follow sections 0 and 1 in [BIDAF](https://github.com/allenai/bi-att-flow). 
  
  2. **Training**:  
  Each experiment for TPRN model took about 13 hours on a single Nividia Tesla P100 GPU. Model converged after 20K updates. To train the model run:
  ```
  python -m basic.cli --mode train --noload --len_opt --cluster --justTPR True --TPRregularizer1 True --TPRvis True --cF 0.00001 --cR 0.00001 --nRoles 20 --nSymbols 100 --batch_size 60 --run_id 00 |& tee <path-to-your-log-files-location>/EXP00.txt
  ```
  `cF` and `cR` are weights for quantization function `Q` in the paper when added to the overall cost function. `cF` is for symbol related terms and `cR` is for role related terms. `nRoles` and `nSymbols` are number of roles and symbols respectively in the TPRN model.
  
    **Note**: If you are training to just observe the reported F1 and Exact Match (EM) numbers use above line. If you are training to observe (reproduce) the interpretability results please use `--batch_size 40`. There is nothing special about batch size 40 but since we used a model trained with batch size 40 for interpretability experiments we advise to use it for reproducibility purposes.
  
  3. **Test**:  
  To test the trained model run:
  ```
  python -m basic.cli --len_opt --cluster --justTPR True --TPRregularizer1 True --load_path "out/basic/00/save/basic-20000" --batch_size 60 --run_id 00
  ```
  Above is BIDAF evaluator which is harsher than SQuAD official evaluator. For official SQuAD evaluator run the following after running BIDAF evaluator:
  ```
  python squad/evaluate-v1.1.py $HOME/data/squad/dev-v1.1.json out/basic/<MODEL_NUMBER>/answer/test-<UPDATE_NUMBER>.json
  ```
  In above example `MODEL_NUMBER` is `00` and `UPDATE_NUMBER` is `20000`.
  
  4. **Interpretability Results**:  
  To regenerate results reported in section 5.2 and 8 (supplementary materials) run:
  ```
  python -m basic.cli --mode test --justTPR True --TPRregularizer1 True --TPRvis True --Fa_F_vis True --vis True --batch_size 60 --load_path "out/basic/00/save/basic-20000" --run_id 00 |& tee <path-to-your-log-files-location>/EXP00_Interpret.txt
  ```
  This will generate the following `.csv` files in your `out/basic/00/TPRvis` directory: 
    - 100 files named `fw_u_aF_MAX_vis_Fa_F_test_set_Filler_k.csv` where `k = 0, 1, 2, ..., 99`. Each file includes all words from the validation set of SQuAD assigned to that Symbol using the procedure described in section 5.2.1 of the paper. For example, the file with `k = 4` includes all words assigned to symbol 4. The structure of each file is: 
      - Column A: Token
      - Column B: Cosine Similarity
      - Column C: Token number in the query
      - Column D: Query number in the validation set

      Now open the file for a symbol in excel (or any other editor you use), sort the rows based on column B (cosine similarity) from largest to smallest, and then remove rows of all duplicate tokens in column A. What you get should are the words assigned to that symbol sorted from the word with highest similarity to the symbol to the word with lowest similarity.  

      **Note**: Please note that after training your model, the symbols IDs might be different or you might observe new patterns created by the model. This is because we do not provide our model with any external information about the semantic or syntactic structure of input dataset and it figures them out by itself through end-to-end training. *If you want to observe exactly our interpretability results, please use our pretrained model (explained in the last question below)*.

   - A file named `fw_u_aF_MAX_vis_Fa_F_fillerID_per_word_test_set` that includes the whole validation set tokens and the symbol ID assigned to it. The structure of this file is as follows:
      - Column A: Query number in the validation set
      - Column B: Token
      - Column C: Symbol ID
      - Column D: Cosine Similarity
      
  - A file named `fw_u_aF_vis_Fa_F_test_set.csv` that includes the cosine similarity between each token and all of 100 symbols. This helps to explore interesting patterns in symbol / word assignments. The structure of this file is as follows:
      - Column A: Query number in the validation set
      - Column B: Token
      - Rest of 100 columns: Cosine similarity between the token and each of 100 symbols. Cosine similarities are calculated in the way described in section 5.2.1 of the paper.
      
  - A file named `B_averaged_whole_test_set.csv` which is the binding matrix averaged over the whole validation set. It is not used for the results of this paper and you can neglect it. 
      
- **I do not want to train the model again. Where is your trained model to exactly regenerate interpretability results reported in the paper?**
  - You can download it from [here](https://www.dropbox.com/s/a4j5ob40spgptr8/29.zip?dl=0). Please note that this is the trained model with batch size 40 that is used for interpretability results. 
  - After downloading above zip file, unzip it and copy it in `out/basic`. Now run the following to get the exact interpretability results reported in the sections 5.2 and 8 of the paper in `out/basic/29/TPRvis`:
  ```
  python -m basic.cli --mode test --justTPR True --TPRregularizer1 True --TPRvis True --Fa_F_vis True --vis True --batch_size 60 --load_path "out/basic/29/save/basic-20000" --run_id 29 |& tee <path-to-your-log-files-location>/EXP29_Interpret.txt
  ```
  
