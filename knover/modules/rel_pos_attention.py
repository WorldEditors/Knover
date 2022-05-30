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
from paddle.nn import functional as F
from paddle import tensor
from paddle.nn import Layer, LayerList
from paddle.framework import ParamAttr
from paddle.nn.layer.transformer import _convert_attention_mask, _convert_param_attr_to_list
from paddle.fluid.data_feeder import convert_dtype

__all__ = []

class RelPosLayer(Layer):
    """
    Layer For Relative Position Embeddings
    """
    def __init__(self,
            rel_min = - 128,
            rel_max = 128,
            emb_size = 512,
            weight_attr=None,
            ):
        """
        relative position is valid between [rel_min, rel_max)
        """
        super(RelPosLayer, self).__init__()
        assert rel_max > rel_min, "relative position max must be larger than min"

        # generate the position mapping matrix
        self.rel_min = rel_min
        self.rel_max = rel_max
        self.emb_size = emb_size
        self.emb_num = rel_max - rel_min + 1

        # create a position embedding parameter
        self.position_emb = self.create_parameter(
            shape=[self.emb_num, self.emb_size],
            attr=weight_attr,
            dtype=self._dtype,
            is_bias=False)

    def relative_pos_2_emb(self, rel_pos_k_q):
        return rel_pos_k_q - self.rel_min

    def emb_2_relative_pos(self, idx):
        return idx + self.rel_min

    def position_embedding(self, parameter_no_grad=False):
        if(not parameter_no_grad):
            return self.position_emb
        else:
            return self.position_emb.detach()

class LinearWrapper(Linear):    
    """
    A Linear Layer with detached parameters
    """
    def forward(self, input, parameter_no_grad=False):
        if(not parameter_no_grad):
            out = F.linear(
                x=input, weight=self.weight, bias=self.bias, name=self.name)
        else:
            out = F.linear(
                x=input, weight=self.weight.detach(), bias=self.bias.detach(), name=self.name)
        return out

class MultiHeadRelPosAttention(Layer):
    """
    Relative Position Attention Layer (DeBerta Type Relative Positions) 
    Relative Position Types:
        1. "cc"
        2. "pc"
        3. "cp"
    """

    Cache = collections.namedtuple("Cache", ["k", "v"])
    StaticCache = collections.namedtuple("StaticCache", ["k", "v"])

    def __init__(self,
                 embed_dim,
                 num_heads,
                 rel_pos_layer,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 pdim=None,
                 need_weights=False,
                 weight_attr=None,
                 bias_attr=None):
        super(MultiHeadRelPosAttention, self).__init__()

        assert embed_dim > 0, ("Expected embed_dim to be greater than 0, "
                               "but recieved {}".format(embed_dim))
        assert num_heads > 0, ("Expected num_heads to be greater than 0, "
                               "but recieved {}".format(num_heads))
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.pdim = pdim if pdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = LinearWrapper(
            embed_dim, 2 * embed_dim, weight_attr, bias_attr=bias_attr)
        self.k_proj = LinearWrapper(
            self.kdim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.v_proj = LinearWrapper(
            self.vdim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.out_proj = LinearWrapper(
            embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.kr_proj = LinearWrapper(
            self.pdim, embed_dim, weight_attr, bias_attr=bias_attr)

        # Use relative position or not
        assert isinstance(rel_pos_layer, RelPosLayer), "Argument `rel_pos_layer` in MHDA must be Type of RelPosLayer"
        self.rel_pos_layer = rel_pos_layer

    def _prepare_qkv(self, query, key, value, cache=None, parameter_no_grad=False):
        r"""
        """
        q = self.q_proj(query, parameter_no_grad=parameter_no_grad)
        qw, qr = paddle.chunk(q, chunks=2, axis=-1)
        qw = tensor.reshape(x=qw, shape=[0, 0, self.num_heads, self.head_dim])
        qw = tensor.transpose(x=qw, perm=[0, 2, 1, 3])
        qr = tensor.reshape(x=qr, shape=[0, 0, self.num_heads, self.head_dim])
        qr = tensor.transpose(x=qr, perm=[0, 2, 1, 3])
        kr = self.kr_proj(self.rel_pos_layer.position_embedding(parameter_no_grad=parameter_no_grad), parameter_no_grad=parameter_no_grad)
        kr = tensor.reshape(x=kr, shape=[0, -1, self.num_heads, self.head_dim])
        kr = tensor.transpose(x=kr, perm=[1, 2, 0, 3])

        if isinstance(cache, self.StaticCache):
            # for encoder-decoder attention in inference and has cached
            k, v = cache.k, cache.v
        else:
            k, v = self.compute_kv(key, value, parameter_no_grad=parameter_no_grad)

        if isinstance(cache, self.Cache):
            # for decoder self-attention in inference
            k = tensor.concat([cache.k, k], axis=2)
            v = tensor.concat([cache.v, v], axis=2)
            cache = self.Cache(k, v)

        return (qw, qr, k, v, kr) if cache is None else (qw, qr, k, v, kr, cache)

    def compute_kv(self, key, value, parameter_no_grad=False):
        k = self.k_proj(key, parameter_no_grad=parameter_no_grad)
        v = self.v_proj(value, parameter_no_grad=parameter_no_grad)
        k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = tensor.transpose(x=v, perm=[0, 2, 1, 3])
        return k, v

    def gen_cache(self, key, value=None, type=Cache, parameter_no_grad=False):
        if type == MultiHeadAttention.StaticCache:  # static_kv
            k, v = self.compute_kv(key, value, parameter_no_grad=parameter_no_grad)
            return self.StaticCache(k, v)
        elif value is None:  # incremental_state
            k = layers.fill_constant_batch_size_like(
                input=key,
                shape=[-1, self.num_heads, 0, self.head_dim],
                dtype=key.dtype,
                value=0)
            v = layers.fill_constant_batch_size_like(
                input=key,
                shape=[-1, self.num_heads, 0, self.head_dim],
                dtype=key.dtype,
                value=0)
            return self.Cache(k, v)
        else:
            # incremental_state with initial value, mainly for usage like UniLM
            return self.Cache(key, value)

    def forward(self, query, key=None, value=None, rel_pos_start_key=0, attn_mask=None, cache=None, parameter_no_grad=False):
        """
        rel_pos_start_key:  the relative position of the start position of key with respect to the start of query
                e.g.,  for query = key, rel_pos_start_key = 0
                       for transformer xl, query preceeds key, rel_pos_start_key = - memory_length
        """
        key = query if key is None else key
        value = query if value is None else value

        # calculate the query / key absolute position id
        # rel_min: -9
        # rel_max: +9
        # emb_num: +30
        # rel_pos_start_key: -10
        nb = query.shape[0]
        l_q = query.shape[1] # 10
        l_k = key.shape[1] # 20
        rel_s = rel_pos_start_key - (l_q - 1) # -9
        rel_e = rel_pos_start_key + (l_k - 1) # +19
        
        pos_emb_s = self.rel_pos_layer.relative_pos_2_emb(rel_s) # 0
        pos_emb_e = self.rel_pos_layer.relative_pos_2_emb(rel_e) # 28
        max_pos_emb = self.rel_pos_layer.emb_num # Max Number of Relative Embeddings being set 19
        max_pos_span = l_q + l_k - 1 # Actual Required Number of Relative Embeddings 29
        
        s_fill = 1 - pos_emb_s # number of how much must be filled in the start, we add extra 1 to make the length l_q + l_k
        e_fill = pos_emb_e + 1 - max_pos_emb # number of the positions to be filled at the end, we add 1 to make the length l_q + l_k
                                             # notice that s_fill + e_fill + max_pos_emb = l_q + l_k must be satisfied

        # compute q ,k ,v
        if cache is None:
            qw, qr, k, v, kr = self._prepare_qkv(query, key, value, cache, parameter_no_grad=parameter_no_grad)
        else:
            qw, qr, k, v, kr, cache = self._prepare_qkv(query, key, value, cache, parameter_no_grad=parameter_no_grad)

        alpha = self.head_dim**-0.5
        # scale dot product attention
        # prod_cc: `[batch_size, num_heads, q_seq_length, k_seq_length]`
        prod_cc = alpha * paddle.matmul(x=qw, y=k, transpose_y=True)
        # prod_cp
        # multiply `[batch_size, num_heads, q_seq_length, d_head]` and `[num_heads, nk, d_head]`
        # acquiring `[batch_size, num_heads, q_seq_length, max_pos_emb]`
        prod_cp = alpha * paddle.matmul(x=qr, y=kr, transpose_y=True)

        # acquiring `[batch_size, num_heads, q_seq_length, max_pos_span + 1]` 10, 30
        # by adding / slicing the rows out of range
        INT_MAX = 100000 # s_fill = 1, e_fill = 10
        if(s_fill < 0):
            prod_cp = paddle.slice(prod_cp, [0, 1, 2, 3], [0, 0, 0, -s_fill], [INT_MAX, INT_MAX, INT_MAX, INT_MAX])
        elif(s_fill > 0):
            fill_vec_s = paddle.slice(prod_cp, [0, 1, 2, 3], [0, 0, 0, 0], [INT_MAX, INT_MAX, INT_MAX, 1])
            fill_vec_s = paddle.tile(fill_vec_s, repeat_times=[1, 1, 1, s_fill])
            prod_cp = paddle.concat([fill_vec_s, prod_cp], axis=-1)
        if(e_fill < 0):
            prod_cp = paddle.slice(prod_cp, [0, 1, 2, 3], [0, 0, 0, 0], [INT_MAX, INT_MAX, INT_MAX, e_fill])
        elif(e_fill > 0):
            fill_vec_e = paddle.slice(prod_cp, [0, 1, 2, 3], [0, 0, 0, -1], [INT_MAX, INT_MAX, INT_MAX, INT_MAX])
            fill_vec_e = paddle.tile(fill_vec_e, repeat_times=[1, 1, 1, e_fill])
            prod_cp = paddle.concat([prod_cp, fill_vec_e], axis=-1)

        # Doing Relative shift
        prod_cp = paddle.reshape(prod_cp, shape=(nb, self.num_heads, max_pos_span + 1, l_q))
        prod_cp = paddle.slice(prod_cp, [0, 1, 2, 3], [0, 0, 1, 0], [INT_MAX, INT_MAX, INT_MAX, INT_MAX])
        prod_cp = paddle.reshape(prod_cp, shape=(nb, self.num_heads, l_q, max_pos_span))
        prod_cp = paddle.slice(prod_cp, [0, 1, 2, 3], [0, 0, 0, 0], [INT_MAX, INT_MAX, INT_MAX, l_k])

        # final attention before attn_mask
        # finally add to shape `[batch_size, num_heads, q_seq_length, k_seq_length]`
        product = prod_cc + prod_cp

        if attn_mask is not None:
            # Support bool or int mask
            attn_mask = _convert_attention_mask(attn_mask, product.dtype)
            product = product + attn_mask
        weights = F.softmax(product)
        if self.dropout:
            weights = F.dropout(
                weights,
                self.dropout,
                training=self.training,
                mode="upscale_in_train")

        out = tensor.matmul(weights, v)

        # combine heads
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out, parameter_no_grad=parameter_no_grad)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        if cache is not None:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)
