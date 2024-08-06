#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

# This is an implementation from scratch of a "GPT", that is a model
# composed of several causal self-attention blocks. It is equipped
# with a caching mechanism for keys and values to avoid a O(N^3) cost
# for auto-regression.

import math

import torch

from torch import nn
from torch.nn import functional as F

######################################################################

# A BracketedSequence is a BxTx... tensor with a first and a nb time
# steps to compute.

# Modules able to process it expect that they will have to process a
# first bracket starting at t=0, followed by a succession of brackets
# that move forward in time, do not overlap, and cover the axis T with
# no holes.
#
# Although it is more general, for a classical prompt-conditioned
# auto-regressive process it will be a first bracket starting at 0 and
# of arbitrary length for the "prompt", followed by brackets of length
# 1 for the successive tokens.
#
# Modules able to process brackets may implement a cache that is
# resetted when the input bracket starts at t=0


class BracketedSequence:
    def __init__(self, x, first=None, nb=None):
        self.x = x
        self.first = 0 if first is None else first
        self.nb = x.size(1) if nb is None else nb

    def slice(self):
        return self.x[:, self.first : self.first + self.nb]

    def complete(self):
        return self.first == 0 and self.nb == self.x.size(1)


######################################################################


class CacheWrapper(nn.Module):
    def __init__(self, *f):
        super().__init__()
        self.f = f[0] if len(f) == 1 else nn.Sequential(*f)

    def forward(self, bs):
        if bs.first == 0:
            y = self.f(bs.slice())
            self.cache_y = y.new(*((y.size(0), bs.x.size(1)) + y.size()[2:]))
            self.cache_y[:, bs.first : bs.first + bs.nb] = y
        else:
            self.cache_y[:, bs.first : bs.first + bs.nb] = self.f(bs.slice())

        return BracketedSequence(self.cache_y, bs.first, bs.nb)


##############################


class WithResidual(nn.Module):
    def __init__(self, *f):
        super().__init__()
        self.f = f[0] if len(f) == 1 else nn.Sequential(*f)

    def forward(self, bs):
        return BracketedSequence(bs.x + self.f(bs).x, bs.first, bs.nb)


##############################


class AddPositionalEncoding(nn.Module):
    def __init__(self, len_max):
        super().__init__()
        self.len_max = len_max

    # [Vaswani et al 2018] PE_{t,2i} = sin(t/(L^{2i/D})), PE_{t,2i+1} = cos(t/(L^{2i/D}))

    def forward(self, bs):
        if bs.first == 0:
            t = torch.arange(bs.x.size(1), dtype=bs.x.dtype, device=bs.x.device)[
                :, None
            ]
            j = torch.arange(bs.x.size(2), dtype=bs.x.dtype, device=bs.x.device)[
                None, :
            ]
            k = j % 2
            self.pe = torch.sin(
                t / (self.len_max ** ((j - k) / bs.x.size(2))) + math.pi / 2 * k
            )
            self.cache_y = bs.x.new(bs.x.size())

        self.cache_y[:, bs.first : bs.first + bs.nb] = (
            bs.slice() + self.pe[bs.first : bs.first + bs.nb]
        )

        return BracketedSequence(self.cache_y, bs.first, bs.nb)


##############################


class EncoderHead(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, bs):
        z = self.fc(bs.x).mean(dim=1)
        return z, bs.x.shape


class DecoderBottom(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, z_shape):
        z, shape = z_shape
        y = self.fc(z)[:, None, :].expand(shape)
        return BracketedSequence(y)


##############################


class QKVAttention(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_qk,
        dim_v,
        nb_heads=1,
        causal=False,
        attention_dropout=0.0,
    ):
        super().__init__()

        def randw(*d):
            return nn.Parameter(torch.randn(*d) / math.sqrt(d[-1]))

        self.causal = causal
        self.attention_dropout = attention_dropout
        self.record_attention = False

        self.w_q = randw(nb_heads, dim_qk, dim_in)
        self.w_k = randw(nb_heads, dim_qk, dim_in)
        self.w_v = randw(nb_heads, dim_v, dim_in)
        self.w_o = randw(dim_v * nb_heads, dim_in)

    def forward(self, bs_q):
        x_q = bs_q.x

        assert (
            self.causal or bs_q.complete()
        ), "Partial evaluation is only possible for causal models"

        if bs_q.first == 0:
            self.cache_k = x_q.new_zeros(
                x_q.size(0), self.w_k.size(0), x_q.size(1), self.w_k.size(1)
            )
            self.cache_v = x_q.new_zeros(
                x_q.size(0), self.w_v.size(0), x_q.size(1), self.w_v.size(1)
            )
            self.cache_y = x_q.new_zeros(x_q.size(0), x_q.size(1), self.w_o.size(1))

        q = torch.einsum(
            "ntc,hdc->nhtd", x_q[:, bs_q.first : bs_q.first + bs_q.nb], self.w_q
        )

        self.cache_k[:, :, bs_q.first : bs_q.first + bs_q.nb] = torch.einsum(
            "ntc,hdc->nhtd", x_q[:, bs_q.first : bs_q.first + bs_q.nb], self.w_k
        )
        self.cache_v[:, :, bs_q.first : bs_q.first + bs_q.nb] = torch.einsum(
            "ntc,hdc->nhtd", x_q[:, bs_q.first : bs_q.first + bs_q.nb], self.w_v
        )

        a = torch.einsum(
            "nhtd,nhsd->nhts", q, self.cache_k[:, :, : bs_q.first + bs_q.nb]
        ) / math.sqrt(self.w_q.size(1))

        if self.causal:
            if bs_q.first == 0:
                self.cache_attzero = (
                    torch.arange(x_q.size(1), device=q.device)[None, None, :, None]
                    < torch.arange(x_q.size(1), device=q.device)[None, None, None, :]
                )
            a = a.masked_fill(
                self.cache_attzero[
                    :, :, bs_q.first : bs_q.first + bs_q.nb, : bs_q.first + bs_q.nb
                ],
                float("-inf"),
            )

        a = a.softmax(dim=3)

        if self.record_attention:
            self.a = a

        a = F.dropout(a, self.attention_dropout, self.training)

        y = torch.einsum(
            "nhts,nhsd->nthd", a, self.cache_v[:, :, : bs_q.first + bs_q.nb]
        ).flatten(2)

        self.cache_y[:, bs_q.first : bs_q.first + bs_q.nb] = y @ self.w_o

        return BracketedSequence(self.cache_y, bs_q.first, bs_q.nb)


##############################


class NoiseInjector(nn.Module):
    def __init__(self, identifier=None):
        super().__init__()
        self.noise_std = 0.0
        self.identifier = identifier

    def forward(self, x):
        if self.noise_std > 0:
            x = x * (
                1 - 2 * (torch.rand(x.size(), device=x.device) < self.noise_std).long()
            )
        return x


##############################


class MyGPT(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        dim_model,
        dim_keys,
        dim_hidden,
        nb_heads,
        nb_blocks,
        causal=False,
        autoencoder_dim=-1,
        dropout=0.0,
        len_max=1e5,
    ):
        super().__init__()

        assert dim_model % nb_heads == 0

        self.temperature = 1.0

        self.embedding = nn.Sequential(
            CacheWrapper(nn.Embedding(vocabulary_size, dim_model), nn.Dropout(dropout)),
            AddPositionalEncoding(len_max),
        )

        trunk_blocks = []

        for b in range(nb_blocks):
            trunk_blocks += [
                WithResidual(
                    CacheWrapper(
                        nn.LayerNorm((dim_model,)),
                        NoiseInjector(identifier=("attention", b)),
                    ),
                    QKVAttention(
                        dim_in=dim_model,
                        dim_qk=dim_keys,
                        dim_v=dim_model // nb_heads,
                        nb_heads=nb_heads,
                        causal=causal,
                        attention_dropout=dropout,
                    ),
                ),
                WithResidual(
                    CacheWrapper(
                        nn.LayerNorm((dim_model,)),
                        NoiseInjector(identifier=("ffw", b)),
                        nn.Linear(in_features=dim_model, out_features=dim_hidden),
                        nn.ReLU(),
                        nn.Linear(in_features=dim_hidden, out_features=dim_model),
                        nn.Dropout(dropout),
                    ),
                ),
            ]

        self.trunk = nn.Sequential(*trunk_blocks)

        self.readout = CacheWrapper(
            nn.Linear(in_features=dim_model, out_features=vocabulary_size)
        )

        # -------------------------------------------------------
        if autoencoder_dim > 0:
            self.encoder = nn.Sequential(
                *(
                    trunk_blocks[: nb_blocks // 2]
                    + [EncoderHead(dim_model, autoencoder_dim)]
                )
            )

            self.decoder = nn.Sequential(
                *(
                    [
                        DecoderBottom(autoencoder_dim, dim_model),
                        AddPositionalEncoding(len_max),
                    ]
                    + trunk_blocks[nb_blocks // 2 :]
                )
            )
        # -------------------------------------------------------

        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Embedding):
                    m.weight.normal_(mean=0, std=2e-2)
                elif isinstance(m, nn.LayerNorm):
                    m.bias.zero_()
                    m.weight.fill_(1.0)

    def forward(self, bs):
        bs = BracketedSequence(F.pad(bs.x, (1, -1)), bs.first, bs.nb)
        bs = self.embedding(bs)
        bs = self.trunk(bs)
        bs = self.readout(bs)
        bs.x[:, bs.first : bs.first + bs.nb] /= self.temperature
        return bs

    def encode(self, bs):
        bs = self.embedding(bs)
        z = self.encoder(bs)
        return z

    def decode(self, z_shape):
        bs = self.decoder(z_shape)
        bs = self.readout(bs)
        return bs

    def partial_forward(self, bs, start_layer=None, end_layer=None):
        if start_layer is None:
            # print(f"GENERATE {bs.first} {bs.first+bs.nb}")
            bs = BracketedSequence(F.pad(bs.x, (1, -1)), bs.first, bs.nb)
            bs = self.embedding(bs)
            if end_layer is not None:
                return self.trunk[:end_layer](bs)
            else:
                bs = self.trunk(bs)
                bs = self.readout(bs)
                return bs
        else:
            bs = self.trunk[start_layer:](bs)
            bs = self.trunk(bs)
            bs = self.readout(bs)
            return bs

    def reset_transformations(self):
        self.temperature = 1.0
        for m in self.modules():
            if isinstance(m, NoiseInjector):
                m.noise_std = 0.0

    def set_noise_injection(self, noise_std, identifier=None):
        for m in self.modules():
            if isinstance(m, NoiseInjector):
                if identifier is None or identifier == m.identifier:
                    m.noise_std = noise_std

    def record_attention(self, v=True):
        for m in self.modules():
            if isinstance(m, QKVAttention):
                m.record_attention = v

    def retrieve_attention(self):
        a = []
        for m in self.modules():
            if isinstance(m, QKVAttention):
                a.append(m.a)
        return a


######################################################################

if __name__ == "__main__":
    print("Basic check.")

    vocabulary_size = 3
    x = torch.randint(vocabulary_size, (1, 5))

    model = MyGPT(
        vocabulary_size=vocabulary_size,
        dim_model=4,
        dim_keys=2,
        dim_hidden=2,
        nb_heads=2,
        nb_blocks=2,
        dropout=0.1,
        causal=True,
    )

    model.eval()
    y1 = model(BracketedSequence(x)).x
    y2 = torch.randn_like(y1)
    for s in range(x.size(1)):
        z = model(BracketedSequence(x, s, 1))
        y2[:, s] = z.slice()

    print(f"error={((y1 - y2).norm() / (y1.norm() + y2.norm())).item()}")

######################################################################
