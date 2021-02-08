"""
@Time : 2021/2/814:06
@Auth : 周俊贤
@File ：config.py
@DESCRIPTION:

"""

from typing import Optional

import torch

# General
hidden_size: int = 512
dec_hidden_size: Optional[int] = 512
embed_size: int = 512
pointer = True

# Data
max_vocab_size = 20000
embed_file: Optional[str] = None  # use pre-trained embeddings
source = 'big_samples'    # use value: train or  big_samples 
data_path: str = './data/dev.txt'
val_data_path = './data/dev.txt'
test_data_path = './data/test.txt'
stop_word_file = './data/HIT_stop_words.txt'
max_src_len: int = 300  # exclusive of special tokens such as EOS
max_tgt_len: int = 100  # exclusive of special tokens such as EOS
truncate_src: bool = True
truncate_tgt: bool = True
min_dec_steps: int = 30
max_dec_steps: int = 100
enc_rnn_dropout: float = 0.5
enc_attn: bool = True
dec_attn: bool = True
dec_in_dropout = 0
dec_rnn_dropout = 0
dec_out_dropout = 0


# Training
trunc_norm_init_std = 1e-4
eps = 1e-31
learning_rate = 0.001
lr_decay = 0.0
initial_accumulator_value = 0.1
epochs = 8
batch_size = 16
coverage = False
fine_tune = False
scheduled_sampling = False
weight_tying = False
max_grad_norm = 2.0
is_cuda = True
DEVICE = torch.device("cuda" if is_cuda else "cpu")
LAMBDA = 1
output_dir = "./output"
if pointer:
    if coverage:
        if fine_tune:
            model_name = 'ft_pgn'
        else:
            model_name = 'cov_pgn'
    elif scheduled_sampling:
        model_name = 'ss_pgn'
    elif weight_tying:
        model_name = 'wt_pgn'
    else:
        if source == 'big_samples':
            model_name = 'pgn_big_samples'
        else:    
            model_name = 'pgn'
else:
    model_name = 'baseline'


# Beam search
beam_size: int = 3
alpha = 0.2
beta = 0.2
gamma = 0.6
