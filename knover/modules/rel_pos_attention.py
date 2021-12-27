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
from paddle.fluid import layers
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
            max_positions=256,
            rel_k=64,
            hidden_size=512,
            weight_attr=None,
            ):

        super(RelPosLayer, self).__init__()
        assert rel_k > 0, ("Expected relative_k to be greater than 0, "
                               "but recieved {}".format(rel_k))
        assert max_positions > 0, ("Expected max_positions to be greater than 0, "
                               "but recieved {}".format(max_positions))
        # generate the position mapping matrix
        self.max_pos = max_positions
        self.rel_k = rel_k
        self.position_map = paddle.zeros(shape=(self.max_pos, self.max_pos), dtype="int64")
        for delta in range(1 - self.max_pos, self.max_pos):
            sigma_ij = max(min(delta + self.rel_k, 2 * self.rel_k - 1), 0)
            for i in range(max(-delta, 0), min(self.max_pos - delta, self.max_pos)):
                self.position_map[i, i + delta] = sigma_ij

        # create a position embedding parameter
        self.position_emb = self.create_parameter(
            shape=[2 * self.rel_k, hidden_size],
            attr=weight_attr,
            dtype=self._dtype,
            is_bias=False)
        print("Relative Position Parameters successfully initialized")


class MultiHeadRelPosAttention(Layer):
    """
    Relative Position Attention Layer 

    Parameters:
        embed_dim (int): The expected feature size in the input and output.
        num_heads (int): The number of heads in multi-head attention.
        rel_pos_layer (optional): If None, no relative position is used, 
            else, use relative position, must be type of (RelPosLayer)
        dropout (float, optional): The dropout probability used on attention
            weights to drop some attention targets. 0 for no dropout. Default 0
        kdim (int, optional): The feature size in key. If None, assumed equal to
            `embed_dim`. Default None.
        vdim (int, optional): The feature size in value. If None, assumed equal to
            `embed_dim`. Default None.
        need_weights (bool, optional): Indicate whether to return the attention
            weights. Default False.
        weight_attr(ParamAttr, optional):  To specify the weight parameter property.
            Default: None, which means the default weight parameter property is used.
            See usage for details in :code:`ParamAttr` .
        bias_attr (ParamAttr|bool, optional): To specify the bias parameter property.
            Default: None, which means the default bias parameter property is used.
            If it is set to False, this layer will not have trainable bias parameter.
            See usage for details in :code:`ParamAttr` .
         
    Examples:

        .. code-block:: python

            import paddle

            # encoder input: [batch_size, sequence_length, d_model]
            query = paddle.rand((2, 4, 128))
            # self attention mask: [batch_size, num_heads, query_len, query_len]
            attn_mask = paddle.rand((2, 2, 4, 4))
            multi_head_attn = paddle.nn.MultiHeadAttention(128, 2)
            output = multi_head_attn(query, None, None, attn_mask=attn_mask)  # [2, 4, 128]
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

        self.q_proj = Linear(
            embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.k_proj = Linear(
            self.kdim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.v_proj = Linear(
            self.vdim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.out_proj = Linear(
            embed_dim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.qr_proj = Linear(
            self.pdim, embed_dim, weight_attr, bias_attr=bias_attr)
        self.kr_proj = Linear(
            self.pdim, embed_dim, weight_attr, bias_attr=bias_attr)

        # Use relative position or not
        assert isinstance(rel_pos_layer, RelPosLayer), "Argument `rel_pos_layer` in MHDA must be Type of RelPosLayer"
        self.rel_pos_layer = rel_pos_layer

    def _prepare_qkv(self, query, key, value, cache=None):
        r"""
        Prapares linear projected queries, keys and values for usage of subsequnt
        multiple parallel attention. If `cache` is not None, using cached results
        to reduce redundant calculations.

        Parameters:
            query (Tensor): The queries for multi-head attention. It is a
                tensor with shape `[batch_size, query_length, embed_dim]`. The
                data type should be float32 or float64.
            key (Tensor): The keys for multi-head attention. It is
                a tensor with shape `[batch_size, key_length, kdim]`. The
                data type should be float32 or float64. If None, use `query` as
                `key`.
            value (Tensor): The values for multi-head attention. It
                is a tensor with shape `[batch_size, value_length, vdim]`.
                The data type should be float32 or float64. If None, use `query` as
                `value`.
            cache (MultiHeadAttention.Cache|MultiHeadAttention.StaticCache, optional):
                It is a namedtuple with `k` and `v` as fields, and stores tensors
                shaped `[batch_size, num_heads, length, embed_dim]` which are results
                of linear projection, reshape and transpose calculations in
                MultiHeadAttention. If is an instance of `Cache`, `k` and `v`
                fields reserve intermediate results of previous positions, which
                mostly used for decoder self attention. If it is an instance of
                `StaticCache`, `key` and `value` args would be ignored, `k` and
                `v` fields would be used as calculated results on `key` and
                `value`, which mostly used for decoder-encoder cross attention.
                It is only used for inference and should be None for training.
                Default None.

        Returns:
            tuple: A tuple including linear projected keys and values. These two \
                tensors have shapes `[batch_size, n_head, sequence_length, d_key]` \
                and `[batch_size, n_head, sequence_length, d_value]` separately, \
                and their data types are same as inputs.
            the shapes of q_r, k_r are `[n_head, 2k, d_value]`
        """
        q = self.q_proj(query)
        q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
        q = tensor.transpose(x=q, perm=[0, 2, 1, 3])
        qr = self.qr_proj(self.rel_pos_layer.position_emb)
        qr = tensor.reshape(x=qr, shape=[0, self.num_heads, self.head_dim])
        qr = tensor.transpose(x=qr, perm=[1, 0, 2])

        if isinstance(cache, self.StaticCache):
            # for encoder-decoder attention in inference and has cached
            k, v, kr = cache.k, cache.v, cache.kr
        else:
            k, v, kr = self.compute_kvr(key, value)

        if isinstance(cache, self.Cache):
            # for decoder self-attention in inference
            k = tensor.concat([cache.k, k], axis=2)
            v = tensor.concat([cache.v, v], axis=2)
            kr = tensor.concat([cache.kr, kr], axis=2)
            cache = self.Cache(k, v, kr)

        return (q, k, v, qr, kr) if cache is None else (q, k, v, qr, kr, cache)

    def compute_kvr(self, key, value):
        r"""
        Applies linear projection on input keys and values, then splits heads
        (reshape and transpose) to get keys and values from different representation
        subspaces. The results are used as key-values pairs for subsequent multiple
        parallel attention.
        
        It is part of calculations in multi-head attention, and is provided as
        a method to pre-compute and prefetch these results, thus we can use them
        to construct cache for inference.

        Parameters:
            key (Tensor): The keys for multi-head attention. It is a tensor
                with shape `[batch_size, sequence_length, kdim]`. The data type
                should be float32 or float64.
            value (Tensor): The values for multi-head attention. It is a tensor
                with shape `[batch_size, sequence_length, vdim]`. The data type
                should be float32 or float64.

        Returns:
            tuple: A tuple including transformed keys and values. Their shapes \
                both are `[batch_size, num_heads, sequence_length, embed_dim // num_heads]`, \
                and their data types are same as inputs.
            shape of kr is `[num_heads, 2k, embed_dim // num_heads]`
        """
        k = self.k_proj(key)
        v = self.v_proj(value)
        kr = self.kr_proj(self.rel_pos_layer.position_emb)
        k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = tensor.transpose(x=v, perm=[0, 2, 1, 3])
        kr = tensor.reshape(x=kr, shape=[0, self.num_heads, self.head_dim])
        kr = tensor.transpose(x=kr, perm=[1, 0, 2])
        return k, v, kr

    def gen_cache(self, key, value=None, type=Cache):
        """
        Generates cache for `forward` usage in inference accroding to arguments.
        The generated cache is an instance of `MultiHeadAttention.Cache` or an
        instance of `MultiHeadAttention.StaticCache`.

        `Cache` or `StaticCache` is namedtuple with `k` and `v` as fields,
        and it stores tensors shaped `[batch_size, num_heads, length, embed_dim]`
        which are results of linear projection, reshape and transpose calculations
        in MultiHeadAttention.
        
        If the generated cache is an instance of `Cache`, `k` and `v` fields
        reserve intermediate result tensors of previous positions, and the tensors
        are incremental among decoding steps, which mostly are used for decoder
        decoder self attention.
        
        If the generated cache is an instance of `StaticCache`, `k` and `v` fields
        would be used as calculated result tensors on keys an values in `forward`,
        and the tensors keep unchanged among decoding steps, which are mostly used
        for decoder-encoder cross attention.

        The cache is generated as follows:

        1. If `type` is `StaticCache`, apply `compute_kv(key, value)` and use the
        results to create an instance of `StaticCache`.
        
        2. If `type` is `Cache` and `value` is None, generate empty tensors shaped
        `[batch_size, num_heads, 0, embed_dim // num_heads]` and use the results
        to create an instance of `Cache`, where `batch_size` is from the first
        dimension of `key`.

        3. If `type` is `Cache` and `value` is not None, use `key`, `value` to create
        an instance of `Cache`.

        Parameters:
            key (Tensor): The keys for multi-head attention. It is
                a tensor with shape `[batch_size, key_length, kdim]`. The
                data type should be float32 or float64. If `value` is None,
                it is only for batch size and data type reference.
            value (Tensor, optional): The values for multi-head attention. It
                is a tensor with shape `[batch_size, value_length, vdim]`.
                The data type should be float32 or float64. If None, `key` is only
                for batch size reference. Default None.
            type (type): It should be `MultiHeadAttention.StaticCache` or
                `MultiHeadAttention.Cache` to indicate the cache type to generate.
        
        Returns:
            namedtuple: an instance of `Cache` or `StaticCache` accordingly.
        """
        if type == MultiHeadAttention.StaticCache:  # static_kv
            k, v, kr = self.compute_kvr(key, value)
            return self.StaticCache(k, v, kr)
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
            kr = layers.fill_constant_batch_size_like(
                input=key[0],
                shape=[-1, 0, self.head_dim],
                dtype=key.dtype,
                value=0)
            return self.Cache(k, v, kr)
        else:
            # incremental_state with initial value, mainly for usage like UniLM
            return self.Cache(key, value, kr)

    def forward(self, query, key=None, value=None, key_pos_start_idx=0, attn_mask=None, cache=None):
        r"""
        Applies multi-head attention to map queries and a set of key-value pairs
        to outputs.

        Parameters:
            query (Tensor): The queries for multi-head attention. It is a
                tensor with shape `[batch_size, query_length, embed_dim]`. The
                data type should be float32 or float64.
            key (Tensor, optional): The keys for multi-head attention. It is
                a tensor with shape `[batch_size, key_length, kdim]`. The
                data type should be float32 or float64. If None, use `query` as
                `key`. Default None.
            value (Tensor, optional): The values for multi-head attention. It
                is a tensor with shape `[batch_size, value_length, vdim]`.
                The data type should be float32 or float64. If None, use `query` as
                `value`. Default None.
            key_pos_start_idx (int, optional): The relative start position of key relative to
                the start position of the query, 0 in default
            attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False` 
                values and the others have `True` values. When the data type is 
                int, the unwanted positions have 0 values and the others have 1 
                values. When the data type is float, the unwanted positions have 
                `-INF` values and the others have 0 values. It can be None when 
                nothing wanted or needed to be prevented attention to. Default None.
            cache (MultiHeadAttention.Cache|MultiHeadAttention.StaticCache, optional):
                It is a namedtuple with `k` and `v` as fields, and stores tensors
                shaped `[batch_size, num_heads, length, embed_dim]` which are results
                of linear projection, reshape and transpose calculations in
                MultiHeadAttention. If it is an instance of `Cache`, `k` and `v`
                fields reserve intermediate results of previous positions, which
                mostly used for decoder self attention. If it is an instance of
                `StaticCache`, `key` and `value` args would be ignored, `k` and
                `v` fields would be used as calculated results on `key` and
                `value`, which mostly used for decoder-encoder cross attention.
                It is only used for inference and should be None for training.
                Default None.

        Returns:
            Tensor|tuple: It is a tensor that has the same shape and data type \
                as `query`, representing attention output. Or a tuple if \
                `need_weights` is True or `cache` is not None. If `need_weights` \
                is True, except for attention output, the tuple also includes \
                the attention weights tensor shaped `[batch_size, num_heads, query_length, key_length]`. \
                If `cache` is not None, the tuple then includes the new cache \
                having the same type as `cache`, and if it is `StaticCache`, it \
                is same as the input `cache`, if it is `Cache`, the new cache \
                reserves tensors concatanating raw tensors with intermediate \
                results of current query.
        """
        key = query if key is None else key
        value = query if value is None else value

        # calculate the query / key absolute position id
        q_s_i = - min(key_pos_start_idx, 0)
        q_e_i = q_s_i + query.shape[1]
        k_s_i = q_s_i + key_pos_start_idx
        k_e_i = k_s_i + key.shape[1]
        assert (max(q_s_i, q_e_i, k_s_i, k_e_i) < self.rel_pos_layer.max_pos), \
                "position out of range, inproper max_pos setting"

        # extract the position map correpsonding to the current query and key position
        # shape: `[q_seq_length, k_seq_length, 2k]`
        pos_mat = self.rel_pos_layer.position_map[q_s_i:q_e_i, k_s_i:k_e_i]

        # compute q ,k ,v
        if cache is None:
            q, k, v, qr, kr = self._prepare_qkv(query, key, value, cache)
        else:
            q, k, v, qr, kr, cache = self._prepare_qkv(query, key, value, cache)

        # Turn qr, kr to `[batch_size, num_heads, 2k, d_heads]`
        nb = q.shape[0]
        nh, nk, dh = qr.shape
        lq = q_e_i - q_s_i
        lk = k_e_i - k_s_i
        qr = paddle.expand(qr, shape=(nb, nh, nk, dh))
        kr = paddle.expand(kr, shape=(nb, nh, nk, dh))

        # prepare position mapping indexes `[batch_size, num_heads, q_length, k_length]`
        exp_pos_mat = paddle.expand(pos_mat, shape=(nb, nh, lq, lk))
        pkr = paddle.reshape(exp_pos_mat, shape=(-1, lk))
        pqr = paddle.reshape(paddle.transpose(exp_pos_mat, perm=[0,1,3,2]), shape=(-1, lq))

        # scale dot product attention
        # TODO(guosheng): use tensor.matmul, however it doesn't support `alpha`
        # prod_cc: `[batch_size, num_heads, q_seq_length, k_seq_length]`
        prod_cc = layers.matmul(
            x=q, y=k, transpose_y=True, alpha=self.head_dim**-0.5)
        # prod_cp: `[batch_size * num_heads * q_seq_length, k_seq_length]`
        prod_cp = paddle.reshape(layers.matmul(
            x=q, y=kr, transpose_y=True, alpha=self.head_dim**-0.5), shape=(-1, nk))
        prod_cp = paddle.index_sample(prod_cp, pkr)
        prod_cp = paddle.reshape(prod_cp, [nb, nh, lq, lk])
        # prod_pc: `[batch_size, num_heads, q_seq_length, k_seq_length]`
        prod_pc = paddle.reshape(layers.matmul(
            x=k, y=qr, transpose_y=True, alpha=self.head_dim**-0.5), shape=(-1, nk))
        prod_pc = paddle.index_sample(prod_pc, pqr)
        prod_pc = paddle.transpose(paddle.reshape(prod_pc, [nb, nh, lk, lq]), perm=[0,1,3,2])

        # final attention before attn_mask
        # finally add to shape `[batch_size, num_heads, q_seq_length, k_seq_length]`
        product = prod_cc + prod_cp + prod_pc

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
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        if cache is not None:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)