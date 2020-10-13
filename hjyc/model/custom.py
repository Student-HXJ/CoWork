import torch
import torch.nn as nn
import math
import random
from torch.autograd import Variable
import torch.nn.functional as F


def conv_stract(*args):
    layer = []
    layer.append(nn.Conv2d(*args))
    layer.append(nn.LeakyReLU(negative_slope=0.05))
    layer.append(nn.Dropout(p=0.05))
    return layer


class Conv_Stract(nn.Module):

    def __init__(self, steps=12, units=128, img_size=(50, 50), init_channel=1):
        super(Conv_Stract, self).__init__()
        self.steps = steps

        img_size = (img_size[0] - 4, img_size[1] - 4)
        img_size2 = ((img_size[0] + 1) // 2, (img_size[1] + 1) // 2)

        self.conv_op = nn.Sequential(*conv_stract(init_channel, 32, 3), *conv_stract(32, 4, 3),
                                     nn.MaxPool2d(2, 2, padding=(img_size[0] % 2, img_size[1] % 2)))
        self.dense = nn.Linear(int(4 * img_size2[0] * img_size2[1]), units)
        self.acti = nn.LeakyReLU(negative_slope=0.05)

    def forward(self, x):
        # need TimeDistributed, but the batch=1, so not use.
        x = self.conv_op(x)
        x = x.contiguous().view(self.steps, -1)
        x = self.dense(x)
        output = self.acti(x)
        return output


class Encoder(nn.Module):

    def __init__(self, hidden_size=128, n_layers=1, dropout=0.05, step=12):
        super(Encoder, self).__init__()
        self.step = step
        # encoder
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.bat = nn.BatchNorm1d(hidden_size)

    def forward(self, src, hidden=None):
        src = src.contiguous().view(self.step, 1, -1)
        output, hidden = self.gru(src, hidden)
        output = output.contiguous().view(self.step, -1)
        output = self.bat(output)
        return output, hidden


class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):

    def __init__(self, hidden_size=128, output_size=50 * 50, n_layers=1, dropout=0.05):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.step = 12

        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + hidden_size, hidden_size, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        input = input.contiguous().view(self.step, 1, -1)
        encoder_outputs = encoder_outputs.contiguous().view(self.step, 1, -1)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)  # (B,1,T)
        # (B,1,T) -> (T,B,1) * (T,B,N) --> (T,B,N)
        context = attn_weights.transpose(0, 1).transpose(0, 2) * encoder_outputs
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([input, context], 2)
        output, hidden = self.gru(rnn_input, last_hidden)  # (T,B,N)
        dense_input = torch.cat([output, context], 2)
        output = self.out(dense_input.contiguous().view(self.step, -1))
        return output.contiguous().view(-1, 50, 50), hidden, attn_weights


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, step=12):
        super(Seq2Seq, self).__init__()
        self.step = step
        self.conv_stract = Conv_Stract(steps=step)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src):
        x = src.contiguous().view(self.step, 1, 50, 50)
        x = self.conv_stract(src)
        encoder_output, hidden = self.encoder(x)
        output, hidden, attn_weights = self.decoder(x, hidden, encoder_output)
        return output


class Res(nn.Module):

    def __init__(self, encoder, img_size=(50, 50), step=12, units=128):
        super(Res, self).__init__()
        self.img_size = img_size
        self.step = step
        self.conv_stract = Conv_Stract(steps=step, img_size=img_size, init_channel=2)
        self.encoder = encoder
        # decoder
        self.seq1 = nn.Sequential(
            nn.Linear(units, units), nn.LeakyReLU(negative_slope=0.05), nn.Dropout(p=0.05), nn.BatchNorm1d(units))
        self.gru = nn.GRU(units, units, 1)
        self.seq2 = nn.Sequential(
            nn.BatchNorm1d(units), nn.Dropout(p=0.05), nn.Linear(units, img_size[0] * img_size[1]),
            nn.LeakyReLU(negative_slope=0.05))

    def forward(self, src):
        x = torch.cat(src, 1)
        x = self.conv_stract(x)
        x, hidden = self.encoder(x)
        x = self.seq1(x)
        x = x.unsqueeze(1)
        x, _ = self.gru(x, hidden)
        x = x.squeeze(1)
        x = self.seq2(x)
        return x.contiguous().view(-1, *self.img_size)


class Custom_Model(nn.Module):

    def __init__(self, step=12):
        super(Custom_Model, self).__init__()
        encoder = Encoder()
        decoder = Decoder()
        self.seq2seq = Seq2Seq(encoder, decoder)
        self.res = Res(Encoder(), step=step)

    def forward(self, vp, vc):
        vc_e = self.seq2seq(vp)
        ec = torch.sub(vc, vc_e.unsqueeze(1))
        ef_ = self.res([vc, ec])
        vf_e = self.seq2seq(vc)
        add = torch.add(vf_e, ef_)
        return add


class TimeDistributed(nn.Module):

    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps
        x_reshaped = x.contiguous().view(x.size(0) * x.size(1), x.size(2), x.size(3))
        y = self.module(x_reshaped)

        # We have to reshape Y
        # (samples, timesteps, output_size)
        y = y.contiguous().view(x.size(0), x.size(1), x.size(2), x.size(3))
        return y
