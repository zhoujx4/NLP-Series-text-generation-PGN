"""
@Time : 2021/2/718:56
@Auth : 周俊贤
@File ：dataset.py
@DESCRIPTION:

"""

import sys
import pathlib
from collections import Counter
from typing import Callable

import torch
from torch.utils.data import Dataset

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))
from utils.utils import simple_tokenizer, count_words, sort_batch_by_len, source2ids, abstract2ids
from data.vocab import Vocab
from utils import config

class PairDataset(object):
    def __init__(self,
                 filename,
                 tokenize=simple_tokenizer,
                 max_src_len=None,
                 max_tgt_len=None,
                 truncate_src=False,
                 truncate_tgt=False):
        print("Reading dataset %s..." % filename, end=' ', flush=True)
        self.filename = filename
        self.pairs = []
        with open(filename, 'rt', encoding='utf-8') as f:
            next(f)
            for i, line in enumerate(f):
                pair = line.strip().split('<sep>')
                src = tokenize(pair[0])
                if max_src_len and len(src) > max_src_len:
                    if truncate_src:
                        src = src[:max_src_len]
                    else:
                        continue
                tgt = tokenize(pair[1])
                if max_tgt_len and len(tgt) > max_tgt_len:
                    if truncate_tgt:
                        tgt = tgt[:max_tgt_len]
                    else:
                        continue
                self.pairs.append((src, tgt))
        print("一共有 %d 对 pairs." % len(self.pairs))

    def build_vocab(self, embed_file) -> Vocab:
        word_counts = Counter()
        count_words(word_counts, [src + tgr for src, tgr in self.pairs])
        vocab = Vocab()
        for word, count in word_counts.most_common(config.max_vocab_size):
            vocab.add_words([word])
        if embed_file is not None:
            count = vocab.load_embeddings(embed_file)
            print("%d 个词加载了预训练的词向量" % count)
        return vocab

class SampleDataset(Dataset):
    def __init__(self, PairDataset, vocab):
        self.src_sents = [x[0] for x in PairDataset]
        self.trg_sents = [x[1] for x in PairDataset]
        self.vocab = vocab
        self._len = len(PairDataset)

    def __getitem__(self, index):
        x, oov = source2ids(self.src_sents[index], self.vocab)
        return {
            'x': [self.vocab.SOS] + x + [self.vocab.EOS],
            'OOV': oov,
            'len_OOV': len(oov),
            'y': [self.vocab.SOS] + abstract2ids(self.trg_sents[index], self.vocab, oov) + [self.vocab.EOS],
            'x_len': len(self.src_sents[index]),
            'y_len': len(self.trg_sents[index])}

    def __len__(self):
        return self._len

def collate_fn(batch):
    def padding(indice, max_length, pad_idx=0):
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    data_batch = sort_batch_by_len(batch)

    x = data_batch["x"]
    x_max_length = max([len(t) for t in x])
    y = data_batch["y"]
    y_max_length = max([len(t) for t in y])

    OOV = data_batch["OOV"]
    len_OOV = torch.tensor(data_batch["len_OOV"])

    x_padded = padding(x, x_max_length)
    y_padded = padding(y, y_max_length)

    x_len = torch.tensor(data_batch["x_len"])
    y_len = torch.tensor(data_batch["y_len"])
    return x_padded, y_padded, x_len, y_len, OOV, len_OOV  # 除了 OOV 是个列表外， 其它都是tensor