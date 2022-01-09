# Introduction
This code is the implementation of the paper "Reviewer assignment based on sentence pair modeling".

If you find this code useful, please cite the following paper:
```
@article{DBLP:journals/ijon/DuanTZWCZ19,
  author    = {Zhen Duan and
               Shicheng Tan and
               Shu Zhao and
               Qianqian Wang and
               Jie Chen and
               Yanping Zhang},
  title     = {Reviewer assignment based on sentence pair modeling},
  journal   = {Neurocomputing},
  volume    = {366},
  pages     = {97--108},
  year      = {2019},
  url       = {https://doi.org/10.1016/j.neucom.2019.06.074},
  doi       = {10.1016/j.neucom.2019.06.074},
  timestamp = {Thu, 10 Oct 2019 09:29:24 +0200},
  biburl    = {https://dblp.org/rec/journals/ijon/DuanTZWCZ19.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

Model download: Baidu Netdisk: https://pan.baidu.com/s/1XKVRCRwRg6-zFs-o5VklPA  Password: t447

Code dependency: (Due to frequent environmental changes, I am not sure if these are the oldest versions.)

- Python=3.6.7
- bert-serving-client==1.6.2
- bert-serving-server==1.6.2
- sklearn==0.20.1
- tensorflow-gpu==1.12.0
- tqdm==4.26.0

# Data required by the code

BERT句词向量目录: 可以重新训练

词向量地址: 不用BERT和随机向量时使用

相似概率矩阵地址: 可有可无, 可以自己计算出, 非论文需要

模型地址: 可有可无, 可加载已有模型. 有模型后不再需要词向量, BERT还是需要的.

Datasets:

- 审稿人论文地址 (reviewers.txt): must have

- 稿件论文地址 (manuscripts.txt): must have

- 标签地址 (groundtruth.txt): must have

# Trained model: synthetic dataset

batchSize = 122

## model/elsevier: W2V-CNN

macro-P: 0.5370, macro-R: 0.3245, macro-F1: 0.4046, MAP: 0.4118, NDCG: 0.5758, bpref: 0.7809

## model/elsevier2: W2V-biLSTM

macro-P: 0.5280, macro-R: 0.3148, macro-F1: 0.3944, MAP: 0.4113, NDCG: 0.5725, bpref: 0.7788

## model/elsevier3: biLSTM

macro-P: 0.3960, macro-R: 0.2291, macro-F1: 0.2902, MAP: 0.2589, NDCG: 0.4241, bpref: 0.7011

## model/elsevier4: W2V-biLSTM-CNN

macro-P: 0.5200, macro-R: 0.3152, macro-F1: 0.3925, MAP: 0.4053, NDCG: 0.5654, bpref: 0.7768

## model/elsevier8: BERT-CNN

macro-P: 0.5435, macro-R: 0.3275, macro-F1: 0.4087, MAP: 0.4137, NDCG: 0.5790, bpref: 0.7833, 

## 10/elsevier1: CNN

macro-P: 0.4220, macro-R: 0.2526, macro-F1: 0.3160, MAP: 0.2846, NDCG: 0.4463, bpref: 0.7184, 

## 10/elsevier3: biLSTM-CNN

macro-P: 0.3935, macro-R: 0.2300, macro-F1: 0.2903, MAP: 0.2634, NDCG: 0.4273, bpref: 0.7014,

# Trained model: public dataset

## arxiv1: BERT-CNN

macro-P: 0.5453, macro-R: 0.4055, macro-F1: 0.4651, MAP: 0.4206, NDCG: 0.6028, bpref: 0.8015, 

## arxiv2: W2V-biLSTM-CNN

macro-P: 0.5693, macro-R: 0.4223, macro-F1: 0.4849, MAP: 0.4876, NDCG: 0.6577, bpref: 0.8362,

## arxiv3: W2V-biLSTM

macro-P: 0.5555, macro-R: 0.4124, macro-F1: 0.4734, MAP: 0.4856, NDCG: 0.6464, bpref: 0.8316, 

## arxiv4: W2V-CNN

macro-P: 0.6319, macro-R: 0.4703, macro-F1: 0.5393, MAP: 0.5784, NDCG: 0.7198, bpref: 0.8773,

## arxiv5: CNN

macro-P: 0.3991, macro-R: 0.2955, macro-F1: 0.3395, MAP: 0.2948, NDCG: 0.4699, bpref: 0.7288,

## arxiv6: biLSTM

macro-P: 0.4251, macro-R: 0.3154, macro-F1: 0.3621, MAP: 0.3221, NDCG: 0.4982, bpref: 0.7468, 

## arxiv7: biLSTM+CNN

macro-P: 0.4208, macro-R: 0.3126, macro-F1: 0.3587, MAP: 0.3254, NDCG: 0.5081, bpref: 0.7550, 

# Trained model: InsuranceQA

## InsuranceQA_model: W2V-biLSTM

## InsuranceQA_model4: W2V-biLSTM-CNN

## InsuranceQA_model6: W2V-CNN

## InsuranceQA_model11: BERT-CNN

