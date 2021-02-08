"""
@Time : 2021/2/89:29
@Auth : 周俊贤
@File ：model.py
@DESCRIPTION:

"""

import os
import sys
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))
from utils import config
from utils.utils import replace_oovs

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size,):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded)
        return output, hidden

class Attention(nn.Module):
    def __init__(self, hidden_units):
        super(Attention, self).__init__()
        self.Wh = nn.Linear(2*hidden_units, 2*hidden_units, bias=False)
        self.Ws = nn.Linear(2*hidden_units, 2*hidden_units)
        self.wc = nn.Linear(1, 2*hidden_units, bias=False)
        self.v = nn.Linear(2*hidden_units, 1, bias=False)

    def forward(self, decoder_states, encoder_output, x_padding_masks, coverage_vector):
        """
        Args:
            decoder_states (tuple): each with shape (1, batch_size, hidden_units)
            encoder_output (Tensor): shape (batch_size, seq_len, hidden_units).
            x_padding_masks (Tensor): shape (batch_size, seq_len).
            coverage_vector (Tensor): shape (batch_size, seq_len).

        Returns:
            context_vector (Tensor): shape (batch_size, 2*hidden_units).
            attention_weights (Tensor): shape (batch_size, seq_length).
            coverage_vector (Tensor): shape (batch_size, seq_length).
        """
        # 把 decoder 的 hidden 向量和 ceil 向量拼接起来作为状态向量
        h_dec, c_dec = decoder_states
        s_t = torch.cat([h_dec, c_dec], dim=2)  # (1, batch_size, 2*hidden_units)
        s_t = s_t.transpose(0, 1)  # (batch_size, 1, 2*hidden_units)
        s_t = s_t.expand_as(encoder_output).contiguous()  # (batch_size, seq_length, 2*hidden_units)

        # 计算attention
        encoder_features = self.Wh(encoder_output.contiguous())  # Wh h_* (batch_size, seq_length, 2*hidden_units)
        decoder_features = self.Ws(s_t)  # Ws s_t (batch_size, seq_length, 2*hidden_units)
        att_inputs = encoder_features + decoder_features  # (batch_size, seq_length, 2*hidden_units)
        # 增加 coverage 向量.
        if config.coverage:
            coverage_features = self.wc(coverage_vector.unsqueeze(2))
            att_inputs = att_inputs + coverage_features

        # 求attention概率分布
        score = self.v(torch.tanh(att_inputs))  # (batch_size, seq_length, 1)
        attention_weights = F.softmax(score, dim=1).squeeze(2)  # (batch_size, seq_length)
        attention_weights = attention_weights * x_padding_masks
        normalization_factor = attention_weights.sum(1, keepdim=True)  # Normalize attention weights after excluding padded positions.
        attention_weights = attention_weights / normalization_factor
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_output)  # (batch_size, 1, 2*hidden_units)
        context_vector = context_vector.squeeze(1)  # (batch_size, 2*hidden_units)

        # Update coverage vector.
        if config.coverage:
            coverage_vector = coverage_vector + attention_weights

        return context_vector, attention_weights, coverage_vector

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, enc_hidden_size=None, is_cuda=True):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.DEVICE = torch.device('cuda') if is_cuda else torch.device('cpu')
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

        self.W1 = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.W2 = nn.Linear(self.hidden_size, vocab_size)
        if config.pointer:
            self.w_gen = nn.Linear(self.hidden_size * 4 + embed_size, 1)

    def forward(self, x_t, decoder_states, context_vector):
        """
        Args:
            x_t (Tensor): shape (batch_size, 1).
            decoder_states (tuple): (h_n, c_n), each with shape (1, batch_size, hidden_units) for each.
            context_vector (Tensor): shape (batch_size,2*hidden_units).
        Returns:
            p_vocab (Tensor): shape (batch_size, vocab_size).
            docoder_states (tuple): The lstm states in the decoder.Each with shapes (1, batch_size, hidden_units).
            p_gen (Tensor): shape (batch_size, 1).
        """
        decoder_emb = self.embedding(x_t)
        decoder_output, decoder_states = self.lstm(decoder_emb, decoder_states)

        # 拼接 状态向量 和 上下文向量
        decoder_output = decoder_output.view(-1, config.hidden_size)
        concat_vector = torch.cat([decoder_output, context_vector], dim=-1)  # (batch_size, 3*hidden_units)

        #
        FF1_out = self.W1(concat_vector)  # (batch_size, hidden_units)
        FF2_out = self.W2(FF1_out)  # (batch_size, vocab_size)
        p_vocab = F.softmax(FF2_out, dim=1)  # (batch_size, vocab_size)

        # Concatenate h and c to get s_t and expand the dim of s_t.
        h_dec, c_dec = decoder_states
        s_t = torch.cat([h_dec, c_dec], dim=2)  # (1, batch_size, 2*hidden_units)

        p_gen = None
        if config.pointer:
            x_gen = torch.cat([context_vector, s_t.squeeze(0), decoder_emb.squeeze(1)], dim=-1)
            p_gen = torch.sigmoid(self.w_gen(x_gen))
        return p_vocab, decoder_states, p_gen

class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

    def forward(self, hidden):
        h, c = hidden
        h_reduced = torch.sum(h, dim=0, keepdim=True)
        c_reduced = torch.sum(c, dim=0, keepdim=True)
        return (h_reduced, c_reduced)


class PGN(nn.Module):
    def __init__(self, v):
        super(PGN, self).__init__()
        self.v = v
        self.DEVICE = config.DEVICE
        self.attention = Attention(config.hidden_size)
        self.encoder = Encoder(len(v), config.embed_size, config.hidden_size)
        self.decoder = Decoder(len(v), config.embed_size, config.hidden_size)
        self.reduce_state = ReduceState()

    def get_final_distribution(self, x, p_gen, p_vocab, attention_weights, max_oov):
        """
        Args:
            x: (batch_size, seq_len)
            p_gen: (batch_size, 1)
            p_vocab: (batch_size, vocab_size)
            attention_weights: (batch_size, seq_len)
            max_oov: (Tensor or int): The maximum sequence length in the batch.
        Returns:
            final_distribution (Tensor): shape (batch_size, )
        """

        if not config.pointer:
            return p_vocab

        batch_size = x.size()[0]
        # Clip the probabilities.
        p_gen = torch.clamp(p_gen, 0.001, 0.999)
        p_vocab_weighted = p_gen * p_vocab  # Get the weighted probabilities.
        attention_weighted = (1 - p_gen) * attention_weights  # (batch_size, seq_len)

        # 得到 词典 和 oov 的总体概率分布
        # extended_size = len(self.v) + max_oovs
        extension = torch.zeros((batch_size, max_oov)).float().to(self.DEVICE)
        p_vocab_extended = torch.cat([p_vocab_weighted, extension], dim=1)  # (batch_size, extended_vocab_size)

        #
        final_distribution = p_vocab_extended.scatter_add_(dim=1, index=x, src=attention_weighted)

        return final_distribution

#     @timer('model forward')
    def forward(self, x, x_len, y, len_oovs, batch, num_batches, teacher_forcing):
        """
        Args:
            x (Tensor): shape (batch_size, seq_len)
            x_len ([int): Sequence length of the current batch.
            y (Tensor): shape (bacth_size, y_len)
            len_oovs (Tensor)
            batch (int)
            num_batches(int)
            teacher_forcing(bool)
        """
        x_copy = replace_oovs(x, self.v)
        x_padding_masks = torch.ne(x, 0).float()
        encoder_output, encoder_states = self.encoder(x_copy)
        decoder_states = self.reduce_state(encoder_states)
        coverage_vector = torch.zeros(x.size()).to(self.DEVICE)
        step_losses = []
        x_t = y[:, 0]
        for t in range(y.shape[1]-1):
            if teacher_forcing:
                x_t = y[:, t]
            x_t = replace_oovs(x_t, self.v)
            y_t = y[:, t+1]
            context_vector, attention_weights, coverage_vector = self.attention(decoder_states, encoder_output, x_padding_masks, coverage_vector)
            p_vocab, decoder_states, p_gen = self.decoder(x_t.unsqueeze(1), decoder_states, context_vector)
            final_dist = self.get_final_distribution(x, p_gen, p_vocab, attention_weights, torch.max(len_oovs))
            x_t = torch.argmax(final_dist, dim=1).to(self.DEVICE)
            if not config.pointer:
                y_t = replace_oovs(y_t, self.v)
            target_probs = torch.gather(final_dist, 1, y_t.unsqueeze(1))
            target_probs = target_probs.squeeze(1)
            mask = torch.ne(y_t, 0).float()
            loss = -torch.log(target_probs + config.eps)
            if config.coverage:
                ct_min = torch.min(attention_weights, coverage_vector)
                cov_loss = torch.sum(ct_min, dim=1)
                loss = loss * config.LAMBDA * cov_loss
            mask = mask.float()
            loss = loss * mask
            step_losses.append(loss)

        sample_losses = torch.sum(torch.stack(step_losses, 1), 1)
        seq_len_mask = torch.ne(y, 0).float()
        batch_seq_len = torch.sum(seq_len_mask, dim=1)
        batch_loss = torch.mean(sample_losses / batch_seq_len)
        return batch_loss
