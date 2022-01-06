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
from paddle.nn.layer.transformer import _convert_attention_mask, _convert_param_attr_to_list, MultiHeadAttention
from knover.modules.rel_pos_attention import MultiHeadRelPosAttention

__all__ = []


class LongTermMemEncoder(Layer):
    """
    Long Term Memory Encoder
    Takes an external long term memory LT, and short term memory ST, and produces the next long term memory LT
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 rel_pos_layer=None,
                 dropout=0.1,
                 activation="relu",
                 position_emb=None,
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 weight_attr=None,
                 bias_attr=None):

        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(LongTermMemEncoder, self).__init__()

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

        #maximum position id in a segment
        self.max_length = max_length
        self.k = k

        weight_attrs = _convert_param_attr_to_list(weight_attr, 2)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 2)

        # Depending on whether to use relative positions, use MHDA or MHA
        if(rel_pos_layer is None):
            self.self_attn = MultiHeadAttention(
                d_model,
                nhead,
                dropout=attn_dropout,
                weight_attr=weight_attrs[0],
                bias_attr=bias_attrs[0])
        else:
            self.self_attn = MultiHeadRelPosAttention(
                d_model,
                nhead,
                rel_pos_layer=rel_pos_layer,
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

    def forward(self, ltm, stm):
        """
        stm: short term memory to be encoded into the long-term memory, should be a tensor of size [Batch, SeqLen, Hidden]
        ltm: long term memory to be updated, should be a tensor of size [Batch, SeqLen, Hidden]
        """
        ltm_len = ltm[0].shape[1]
        stm_len = stm[0].shape[1]
        query = ltm
        key = paddle.concat(x=[ltm, stm], axis=1)

        residual = ltm

        if self.normalize_before:
            key = self.norm1(key)
            query = self.norm1(ltm)

        # If relative position is used, the relative start position of key is the same as the relative start position of query
        output = self.self_attn(query, key, key)

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

        return output

class MemAugDecoderLayer(Layer):
    """
    RecFormer Encoder Layer
    Takes the External Memory M_{t}^{L}, X_{t}^{L}, yield the External Memory in Next Step M_{t+1}^{L}, X_{t}^{L+1}
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 rel_pos_layer=None,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 independent_attn=False,
                 weight_attr=None,
                 bias_attr=None):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(MemAugDecoderLayer, self).__init__()

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

        if(rel_pos_layer is None):
            self.use_rel_pos = False
            self.self_attn = MultiHeadAttention(
                d_model,
                nhead,
                dropout=attn_dropout,
                weight_attr=weight_attrs[0],
                bias_attr=bias_attrs[0])
        else:
            self.use_rel_pos = True
            self.self_attn = MultiHeadRelPosAttention(
                d_model,
                nhead,
                rel_pos_layer=rel_pos_layer,
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

    def forward(self, memory, tgt, tgt_mask=None, cache=None):
        """
        src: source to be encoded into the memory, should be tensor of size [Batch, SeqLen, Hidden]
        memory: tensor of size [Batch, SegmentLen, Hidden]
        is_last_layer: for the last layer we will not process the source token
        """
        l_s = tgt.shape[1]
        residual = tgt

        if(memory is None):
            concat_src = tgt
            l_m = 0
        else:
            concat_src = paddle.concat(x=[memory, tgt], axis=1)
            l_m = memory.shape[1]

        if self.normalize_before:
            tgt = self.norm1(tgt)
            concat_src = self.norm1(concat_src)

        # Notice that the key position starts from -l_m here 
        if(self.use_rel_pos):
            if(cache is None):
                output = self.self_attn(tgt, concat_src, concat_src, attn_mask=tgt_mask, rel_pos_start_key=-l_m)
            else:
                output, new_cache = self.self_attn(tgt, concat_src, concat_src, attn_mask=tgt_mask, rel_pos_start_key=-l_m, cache=cache)
        else:
            if(cache is None):
                output = self.self_attn(tgt, concat_src, concat_src, attn_mask=tgt_mask)
            else:
                output, new_cache = self.self_attn(tgt, concat_src, concat_src, attn_mask=tgt_mask, cache=cache)

        residual = tgt

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

        return output if cache is None else (output, new_cache)

    def gen_cache(self, tgt):
        incremental_cache = self.self_attn.gen_cache(
                src, type=self.self_attn.Cache)
        return incremental_cache

class MemAugDecoder(Layer):
    def __init__(self, decoder_layer, d_model, num_layers, normalize_before=False):
        super(MemAugDecoder, self).__init__()
        self.layers = LayerList([(decoder_layer if i == 0 else
                type(decoder_layer)(**decoder_layer._config))
                for i in range(num_layers)])

        self.num_layers = num_layers
        self.normalize_before = normalize_before
        if(self.normalize_before):
            self.norm1 = [LayerNorm(d_model) for i in range(num_layers)]
            self.norm2 = LayerNorm(d_model) 

    def forward(self, memories, tgt, tgt_mask=None, caches=None):
        """
        tgt:  A tensor of shape [Batch, Sequence, Hidden]
        memories:  A NumLayers list of tensor of shape [Batch, Sequence, Hidden]
        """
        seq_len = tgt.shape[1]
        if(memories is None):
            mem_len = 0
        else:
            mem_len = memories[0].shape[1]
            if(tgt_mask is not None):
                tgt_mask = paddle.concat([paddle.zeros((seq_len, mem_len)), tgt_mask], axis=1)

        output = tgt
        new_caches = []
        new_memories = []
        new_memories.append(output.detach())
        for i, mod in enumerate(self.layers):
            if(memories is None):
                mem = None
            else:
                mem = self.norm1[i](memories[i]) if self.normalize_before else memories[i]
            if caches is None:
                output = mod(mem, output,
                        tgt_mask=tgt_mask,
                        cache=None)
            else:
                output, new_cache = mod(mem, output,
                        tgt_mask=tgt_mask,
                        cache=caches["cache"][i])
                new_caches.append(new_cache)
            new_memories.append(output.detach())
        del new_memories[-1]

        if self.normalize_before:
            output = self.norm2(output)

        return new_memories, output if caches is None else (new_memories, output, new_caches)

    def gen_cache(self, tgt, do_zip=False):
        """
        Memory, tensor of [Batch, NumLayer, Sequence, Hidden]
        seg_idx, [0, seg_idx) represents the range of the memory
        """
        cache = [layer.gen_cache(tgt) for layer in self.layers]
        if do_zip:
            cache = list(zip(*cache))
        return cache
