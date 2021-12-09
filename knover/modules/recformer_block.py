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
from paddle.nn.layer.transformer import _convert_attention_mask, _convert_param_attr_to_list

__all__ = []


class RecFormerEncoderLayer(Layer):
    """
    RecFormer Encoder Layer
    Takes the External Memory M_{t}^{L}, X_{t}^{L}, yield the External Memory in Next Step M_{t+1}^{L}, X_{t}^{L+1}
    """

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

    def forward(self, src, memory, is_last_layer=False):
        """
        src: source to be encoded into the memory, should be tensor of size [Batch, SeqLen, Hidden]
        memory: tensor of size [Batch, SegmentLen, Hidden]
        is_last_layer: for the last layer we will not process the source token
        """
        l_m = memory.shape[1]
        l_s = src.shape[1]
        concat_src = paddle.concat(x=[src, memory], axis=1)
        if self.normalize_before:
            concat_src = self.norm1(concat_src)

        # Add cache for encoder for the usage like UniLM
        if(not is_last_layer):
            output = self.self_attn(concat_src, concat_src, concat_src, None)
            residual = concat_src
        else:
            output = self.self_attn(memory, concat_src, concat_src, None)
            residual = memory

        output = residual + self.dropout1(output)

        if not self.normalize_before:
            output = self.norm1(output)

        residual = output
        if self.normalize_before:
            output = self.norm2(output)

        output = self.linear2(self.dropout(self.activation(self.linear1(output))))
        output = residual + self.dropout2(output)
        if not self.normalize_before:
            output = self.norm2(output)
        #return M_{t+1}^{L}, X_{t}^{L+1} if L is not last layer else M_{t+1}^{L} only

        return paddle.split(output, [l_s, l_m], axis=1) if(not is_last_layer) else output

class RecFormerEncoder(Layer):
    """
    Define Recursive Encoder
    Takes the External Memory M_{t}, X_{t}, yield the External Memory in Next Step M_{t+1}
    M_{t}: A length "NumLayer" list of Tensor of Size [BatchSize, SegmentLength, Hidden]
    X_{t}: input of size [Batch, SeqLength, Hidden]
    M_{t+1}: The same shaped memory as M_{t} containing the information of X_t and M_t
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
        memories:  A NumLayers list of tensor of shape [Batch, Sequence, Hidden]
        """
        assert(len(memories) == self.num_layers), "The memories have different layers"

        new_memories = []
        output = src
        for i, mod in enumerate(self.layers[:-1]):
            output, mem = mod(output, memories[i])
            new_memories.append(mem)
        mem = self.layers[-1](output, memories[self.num_layers-1], is_last_layer=True)
        new_memories.append(mem)

        return new_memories

class RecFormerDecoder(Layer):

    def __init__(self, decoder_layer, d_model, num_layers, normalize_before=False):
        super(RecFormerDecoder, self).__init__()
        self.layers = LayerList([(decoder_layer if i == 0 else
                                  type(decoder_layer)(**decoder_layer._config))
                                 for i in range(num_layers)])
        self.num_layers = num_layers
        self.normalize_before = normalize_before
        if(self.normalize_before):
            self.norm1 = [LayerNorm(d_model) for i in range(num_layers)]
            self.norm2 = LayerNorm(d_model) 

    def forward(self, tgt, memories, tgt_mask=None, caches=None):
        """
        tgt:  A tensor of shape [Batch, Sequence, Hidden]
        memories:  A NumLayers list of tensor of shape [Batch, Sequence, Hidden]
        """
        tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)

        output = tgt
        new_caches = []
        for i, mod in enumerate(self.layers):
            mem = self.norm1[i](memories[i]) if self.normalize_before else memories[i]
            if caches is None:
                output = mod(output,
                        mem,
                        tgt_mask=tgt_mask,
                        cache=None)
            else:
                output, new_cache = mod(output,
                        mem,
                        tgt_mask=tgt_mask,
                        cache=caches["cache"][i])
                new_caches.append(new_cache)
        if self.norm2 is not None:
            output = self.norm2(output)

        return output if caches is None else (output, new_caches)

    def gen_cache(self, memory, do_zip=False):
        """
        Memory, tensor of [Batch, NumLayer, Sequence, Hidden]
        seg_idx, [0, seg_idx) represents the range of the memory
        """
        cache = [layer.gen_cache(memory[i]) for (i, layer) in enumerate(self.layers)]
        if do_zip:
            cache = list(zip(*cache))
        return cache
