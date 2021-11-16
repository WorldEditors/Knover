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
from paddle.nn import Linear, Dropout
from paddle.nn import LayerNorm
import paddle.nn.functional as F
from paddle import tensor
from paddle.fluid import layers
from paddle.nn import Layer, LayerList
from paddle.framework import ParamAttr
from paddle.fluid.data_feeder import convert_dtype
from paddle.nn import TransformerDecoderLayer, MultiHeadAttention
from paddle.nn.layer.transformer import _convert_attention_mask

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

        super(RecFormerEncoderLayer, self).__init__()

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

    def forward(self, src, memories):
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

    def forward(self, tgt, memories, tgt_mask=None, caches=None):
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
                            cache=cache["cache"][i])
                new_caches.append(new_cache)
        if self.norm is not None:
            output = self.norm(output)

        return output if cache is None else (output, new_caches)

    def gen_cache(self, memory, do_zip=False):
        """
        Memory, tensor of [Batch, NumLayer, Sequence, Hidden]
        seg_idx, [0, seg_idx) represents the range of the memory
        """
        cache = [layer.gen_cache(memory[:,i,:,:]) for (layer, i) in enumerate(self.layers)]
        if do_zip:
            cache = list(zip(*cache))
        return cache
