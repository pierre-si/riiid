import math

import torch
from torch import nn
import torch.nn.functional as F
    
    
class RiiidEmbedding(nn.Module):
    def __init__(self, maximums, pad_idx=0, emb_size=16, dim=128):
        super().__init__()
        self.emb_size = emb_size
        self.question_emb = nn.Embedding(maximums['question_id']+1, emb_size, padding_idx = pad_idx)
        self.part_emb = nn.Embedding(maximums['part']+1, emb_size, padding_idx = pad_idx)
        self.answer_emb = nn.Embedding(maximums['answered_correctly']+1, emb_size, padding_idx=pad_idx)
        self.cont_emb = nn.Sequential(
            nn.Linear(2, emb_size),
            nn.LayerNorm(emb_size)
            )
        self.merge = nn.Linear(4*emb_size, dim)#, bias=False)

    def forward(self, x_cat, x_cont):
        if len(x_cat.size()) == 2:
            x_cat = x_cat.unsqueeze(0)
            x_cont = x_cont.unsqueeze(0)
        # x_cat: N×L×3
        # x_cat[:, :, 0]: N×L
        q_emb = self.question_emb(x_cat[:, :, 0]) # N×L×D
        p_emb = self.part_emb(x_cat[:, :, 1])
        a_emb = self.answer_emb(x_cat[:, :, 2])
        cont_emb = self.cont_emb(x_cont)
        emb = torch.cat([q_emb, p_emb, a_emb, cont_emb], dim=2)
        emb = self.merge(emb)
        return emb

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LastQueryTransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(LastQueryTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(LastQueryTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask = None, src_key_padding_mask = None, seq_lengths= None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if seq_lengths is None: # when using left padding
            queries = src[-1:, :, :]
        else:
            batch_indexes = torch.arange(src.size()[1])
            seq_indexes = seq_lengths - 1
            queries = src[seq_indexes, batch_indexes].unsqueeze(0)

        src2 = self.self_attn(queries, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2) # scr2's first dimension (length) is broadcasted
        src = self.norm1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src = src + self.dropout2(src2)
        # src = self.norm2(src)
        return src

class Riiid(nn.Module):
    def __init__(self, maximums, pad_idx = 0, dropout = 0.1):
        super(Riiid, self).__init__()
        self.pad_idx = pad_idx
        self.emb = RiiidEmbedding(maximums, pad_idx=pad_idx, dim=128)
        #self.pos_encoder = PositionalEncoding(d_model=128, dropout=dropout, max_len=16409)
        
        self.encoder_layer = LastQueryTransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=256, dropout=dropout)

        bidirectional = False
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=False, bidirectional=bidirectional) # batch_first is False by default.

        dnn_in = 256 if bidirectional else 128
        self.dnn = nn.Sequential(
            nn.Linear(dnn_in, 128),
            nn.ReLU(),
            #nn.BatchNorm1d(256),
            nn.Linear(128, 1),
        )
                
    def forward(self, x_cat, x_cont, seq_lengths=None):
        pad_mask = self.make_padding_mask(x_cat)
        x = self.emb(x_cat, x_cont)
        #x = self.pos_encoder(x)
        x = x.transpose(1, 0) # pytorch MHA requires input to be S×N×E (S: source sequence length)
        x = self.encoder_layer(x, src_key_padding_mask=pad_mask, seq_lengths=seq_lengths)
        x = nn.utils.rnn.pack_padded_sequence(x, seq_lengths)
        #x = self.lstm(x)[0][-1] # output: S × N × hidden_size, thus N × hidden
        x = self.lstm(x)[1][0][-1] # (h_n, c_n)[0][0], h_n: n_layers*n_directions (=1) × N × hidden_size
        #x = x.transpose(1, 0)

        x = self.dnn(x) # N × 1

        return x 

    def make_padding_mask(self, x_cat):
        pad_mask = (x_cat[:, :, 0] == self.pad_idx)
        return pad_mask
