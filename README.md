# BLOGER
[![arXiv](https://img.shields.io/badge/arXiv-2510.21242-red.svg)](https://arxiv.org/abs/2510.21242)

This is the pytorch implementation of the paper at SIGIR 2026:
> [Bi-Level Optimization for Generative Recommendation: Bridging Tokenization and Generation](https://arxiv.org/abs/2510.21242))
> 
> Yimeng Bai, Chang Liu, Yang Zhang, Dingxian Wang, Frank Yang, Andrew Rabinovich, Wenge Rong, Fuli Feng.

## Usage

### Data
The experimental datasets should be preprocessed into **JSON format**. You may refer to this [example data](https://github.com/HonghuiBao2000/LETTER/tree/master/data) for guidance. 

Next, you need to extract semantic embeddings for each item description. We provide a script `./data/get_text_emb.py` for this purpose. This script uses LLaMA to convert item textual descriptions into dense vectors, which are then used as inputs to RQ-VAE.

### Training & Evaluation
#### 1. Pretrain the RQ-VAE Model (TIGER/LETTER)
```
python run_gr_id.py
```
#### 2. Train with Bi-Level Optimization
Make sure the pretrained RQ-VAE checkpoint is available before running this step. 
```
python run_gr_rec_blo.py
```
