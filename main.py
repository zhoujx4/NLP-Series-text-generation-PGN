"""
@Time : 2021/2/89:16
@Auth : 周俊贤
@File ：main.py
@DESCRIPTION:

"""
import copy
import os

import numpy as np
from torch import optim
from torch.utils.data import DataLoader
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from dataset.dataset import PairDataset, SampleDataset, collate_fn
from model.model import PGN
from utils import config
from utils.utils import ScheduledSampler, config_info, ProgressBar

def train(train_iter, model, v, teacher_forcing):
    """
    Args:
        dataset (dataset.PairDataset)
        val_dataset (dataset.PairDataset)
        v (vocab.Vocab)
        start_epoch (int, optional)
    """
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    #
    pbar_batch = ProgressBar(n_total=len(train_iter), desc="Traing_batch")
    batch_loss = 0
    batch_losses = []
    for batch, data in enumerate(train_iter):
        x, y, x_len, y_len, oov, len_oovs = data
        if config.is_cuda:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            x_len = x_len.to(DEVICE)
            y_len = y_len.to(DEVICE)
            len_oovs = len_oovs.to(DEVICE)

        loss = model(x,
                     x_len,
                     y,
                     len_oovs,
                     batch=batch,
                     num_batches=num_batches,
                     teacher_forcing=teacher_forcing)
        batch_losses.append(loss.item())
        loss.backward()
        clip_grad_norm_(model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(model.attention.parameters(), config.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        batch_loss += loss.item()
        pbar_batch(batch, {"batch_loss": batch_loss / (batch + 1)})

    batch_losses = np.mean(batch_losses)
    return batch_losses

def evaluate(model, eval_iter):
    """
    Args:
        model (torch.nn.Module)
        val_data (dataset.PairDataset)
    """
    val_loss = []
    model.eval()
    with torch.no_grad():
        DEVICE = config.DEVICE
        for batch, data in enumerate(tqdm(eval_iter)):
            x, y, x_len, y_len, oov, len_oovs = data
            if config.is_cuda:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                x_len = x_len.to(DEVICE)
                y_len = y_len.to(DEVICE)
                len_oovs = len_oovs.to(DEVICE)
            loss = model(x,
                         x_len,
                         y,
                         len_oovs,
                         batch=batch,
                         num_batches=len(eval_iter),
                         teacher_forcing=True)
            val_loss.append(loss.item())
    return np.mean(val_loss)

def main():
    DEVICE = torch.device('cuda') if config.is_cuda else torch.device('cpu')
    dataset = PairDataset(config.data_path,
                          max_src_len=config.max_src_len,
                          max_tgt_len=config.max_tgt_len,
                          truncate_src=config.truncate_src,
                          truncate_tgt=config.truncate_tgt)
    val_dataset = PairDataset(config.val_data_path,
                              max_src_len=config.max_src_len,
                              max_tgt_len=config.max_tgt_len,
                              truncate_src=config.truncate_src,
                              truncate_tgt=config.truncate_tgt)
    vocab = dataset.build_vocab(embed_file=config.embed_file)
    train_data = SampleDataset(dataset.pairs, vocab)
    val_data = SampleDataset(val_dataset.pairs, vocab)
    train_iter = DataLoader(dataset=train_data,
                            batch_size=config.batch_size,
                            shuffle=True,
                            collate_fn=collate_fn)
    eval_iter = DataLoader(dataset=val_data,
                           batch_size=config.batch_size,
                           shuffle=True,
                           pin_memory=True, drop_last=True,
                           collate_fn=collate_fn)

    #
    model = PGN(vocab)
    model.to(DEVICE)

    #
    best_val_loss = np.inf
    start_epoch = 0
    num_epochs = len(range(start_epoch, config.epochs))
    scheduled_sampler = ScheduledSampler(num_epochs)
    if config.scheduled_sampling:
        print('动态 Teather forcing 模式打开')
    pbar_epoch = ProgressBar(n_total=len(train_iter), desc="Traing_epoch")
    for epoch in range(start_epoch, config.epochs):
        model.train()
        # Teacher Forcing模式
        if config.scheduled_sampling:
            teacher_forcing = scheduled_sampler.teacher_forcing(epoch - start_epoch)
        else:
            teacher_forcing = True
        print('teacher_forcing = {}'.format(teacher_forcing))
        # 训练
        batch_loss = train(train_iter, model, vocab, teacher_forcing)
        pbar_epoch(epoch, {"epoch_loss": batch_loss})
        # 验证
        val_loss = evaluate(model, eval_iter)
        print('validation loss:{}'.format(val_loss))
        if (val_loss < best_val_loss):
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.module if hasattr(model, "module") else model)
            torch.save(best_model.state_dict(), os.path.join(config.output_dir, "best_model.pkl"))

if __name__ == "__main__":
    main()