#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: define the classes of Transformer neural network

import copy
import collections
import numpy as np

import paddle
from .common import Linear, Dropout
from .norm import LayerNorm
from .. import functional as F
from ... import tensor
from ...fluid import layers
from .. import Layer, LayerList
from ...framework import ParamAttr
from paddle.fluid.data_feeder import convert_dtype
from paddle.nn.Layers import _convert_attention_mask, MultiHeadAttention

__all__ = []


class RecFormerEncoderLayer(Layer):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 weight_attr=None,
                 bias_attr=None):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerEncoderLayer, self).__init__()

        assert d_model > 0, ("Expected d_model to be greater than 0, "
                             "but recieved {}".format(d_model))
        assert nhead > 0, ("Expected nhead to be greater than 0, "
                           "but recieved {}".format(nhead))
        assert dim_feedforward > 0, (
            "Expected dim_feedforward to be greater than 0, "
            "but recieved {}".format(dim_feedforward))

        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 2)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 2)

        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0])

        self.linear1 = Linear(
            d_model, dim_feedforward, weight_attrs[1], bias_attr=bias_attrs[1])
        self.dropout = Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = Linear(
            dim_feedforward, d_model, weight_attrs[1], bias_attr=bias_attrs[1])
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)

    def forward(self, src, memory):
        src_mask = _convert_attention_mask(src_mask, src.dtype)
        memory_mask = _convert_attention_mask(memory_mask, memory.dtype)

        residual = src
        l_m = memory.shape[1]
        l_s = src.shape[1]
        concat_src = paddle.concat(x=[memory, src], axis=1)
        if self.normalize_before:
            concat_src = self.norm1(concat_src)

        # Add cache for encoder for the usage like UniLM
        concat_src = self.self_attn(concat_src, concat_src, concat_src, None)
        concat_src = residual + self.dropout1(concat_src)

        if not self.normalize_before:
            concat_src = self.norm1(concat_src)

        residual = concat_src
        if self.normalize_before:
            concat_src = self.norm2(concat_src)
        concat_src = self.linear2(self.dropout(self.activation(self.linear1(concat_src))))
        concat_src = residual + self.dropout2(concat_src)
        if not self.normalize_before:
            concat_src = self.norm2(concat_src)
        return paddle.split(concat_src, [l_m, l_s])

class RecFormerEncoder(Layer):
    """
    Define Recursive Encoder
    """

    def __init__(self, encoder_layer, num_layers):
        super(RecFormerEncoder, self).__init__()
        self.layers = LayerList([(encoder_layer if i == 0 else
                                  type(encoder_layer)(**encoder_layer._config))
                                 for i in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, memories)
        """
        src:  A tensor of shape [Batch, Sequence, Hidden]
        memories:  A tensor of shape [Batch, NumLayers, Sequence, Hidden]
        """
        assert(memories.shape[1] == self.num_layers), "The memories have different layers"

        new_memories = []
        new_memories.append(src)
        for i, mod in enumerate(self.layers):
            new_memories.append(mod(new_memories[-1], memories[i]))

        return new_memories[:, 1:, :, :]

class RecFormerDecoder(Layer):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(RecFormerDecoder, self).__init__()
        self.layers = LayerList([(decoder_layer if i == 0 else
                                  type(decoder_layer)(**decoder_layer._config))
                                 for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memories, tgt_mask=None, memory_mask=None, cache=None):
        """
        tgt:  A tensor of shape [Batch, Sequence, Hidden]
        memories:  A tensor of shape [Batch, NumLayers, Sequence, Hidden]
        """
        tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)
        memory_mask = _convert_attention_mask(memory_mask, memory.dtype)

        output = tgt
        new_caches = []
        for i, mod in enumerate(self.layers):
            if cache is None:
                output = mod(output,
                             memories[i],
                             tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             cache=None)
            else:
                output, new_cache = mod(output,
                            memories[i],
                            tgt_mask=tgt_mask,
                            memory_mask=memory_mask,
                            cache=cache[i])
                new_caches.append(new_cache)

class RecFormer(Layer):
    def __init__(self,
                 d_model=512,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 segment_len=128,
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 weight_attr=None,
                 bias_attr=None):

        super(RecFormerDecoder, self).__init__()

        assert d_model > 0, ("Expected d_model to be greater than 0, "
                             "but recieved {}".format(d_model))
        assert nhead > 0, ("Expected nhead to be greater than 0, "
                           "but recieved {}".format(nhead))
        assert dim_feedforward > 0, (
            "Expected dim_feedforward to be greater than 0, "
            "but recieved {}".format(dim_feedforward))

        if isinstance(bias_attr, (list, tuple)):
            if len(bias_attr) == 1:
                encoder_bias_attr = [bias_attr[0]] * 2
                decoder_bias_attr = [bias_attr[0]] * 3
            elif len(bias_attr) == 2:
                encoder_bias_attr = bias_attr
                decoder_bias_attr = [bias_attr[0], bias_attr[0], bias_attr[-1]]
            elif len(bias_attr) == 3:
                encoder_bias_attr = [bias_attr[0], bias_attr[-1]]
                decoder_bias_attr = bias_attr
            else:
                assert False, (
                    "length of bias_attr should be 1 or 2 or 3 when it is a list/tuple"
                )
        else:
            encoder_bias_attr = bias_attr
            decoder_bias_attr = bias_attr

        if isinstance(weight_attr, (list, tuple)):
            if len(weight_attr) == 1:
                encoder_weight_attr = [weight_attr[0]] * 2
                decoder_weight_attr = [weight_attr[0]] * 3
            elif len(weight_attr) == 2:
                encoder_weight_attr = weight_attr
                decoder_weight_attr = [
                    weight_attr[0], weight_attr[0], weight_attr[-1]
                ]
            elif len(weight_attr) == 3:
                encoder_weight_attr = [weight_attr[0], weight_attr[-1]]
                decoder_weight_attr = weight_attr
            else:
                assert False, (
                    "length of weight_attr should be 1 or 2 or 3 when it is a list/tuple"
                )
        else:
            encoder_weight_attr = weight_attr
            decoder_weight_attr = weight_attr

        self.encoder_layer = RecFormerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation,
            attn_dropout, act_dropout, normalize_before,
            encoder_weight_attr, encoder_bias_attr)
        self.encoder = RecFormerEncoder(encoder_layer, self.n_layer)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation,
            attn_dropout, act_dropout, normalize_before,
            decoder_weight_attr, decoder_bias_attr)
        decoder_norm = LayerNorm(d_model)
        self.decoder = RecFormerDecoder(decoder_layer, num_layers,
            decoder_norm)

        self.d_model = d_model
        self.nhead = nhead
        self.segment_len = segment_len
        self.num_layers = num_layers

    def forward(self, src, tgt):
        """
        src [Batch, SrcSeqLen, Hidden]
        tgt [Batch, TgtSeqLen, Hidden]
        """
        batch_size = tgt.shape[0]
        tgt_len = tgt.shape[1]
        if(src is not None):
            src_len = src.shape[1]
        else:
            src_len = 0

        l_memory = paddle.full(shape=[batch_size, self.num_layers, self.segment_len, self.d_model], 
                fill_value=0, dtype=tgt.dtype)

        tgt_splits = (tgt_len - 1) // self.segment_len + 1
        src_splits = (src_len - 1) // self.segment_len + 1

        if(src is not None):
            src_inputs = paddle.split(src, src_splits, axis=1)
            for src_seg in src_inputs:
                l_memory, _ = self.encoder(src_seg, l_memory)

        src_mask = _convert_attention_mask(src_mask, src.dtype)
        tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)
        tgt_inputs = paddle.split(tgt, tgt_splits, axis=1)
        outputs = []
        for tgt_seg in tgt_inputs:
            outputs.append(self.decoder(tgt_seg, l_memory, tgt_mask = self.generate_sequare_subsequent_mask(tgt_inputs.shape[1])))
            l_memory, _ = self.encoder(tgt_seg, l_memory)

        return output

    def generate_square_subsequent_mask(self, length):
        return paddle.tensor.triu(
            (paddle.ones(
                (length, length), dtype=paddle.get_default_dtype()) * -np.inf),
            1)


if __name__=="__main__":
    pass
