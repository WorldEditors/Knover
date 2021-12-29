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
        self.emb_num = rel_max - rel_min

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
        kr = self.kr_proj(self.rel_pos_layer.position_emb)
        kr = tensor.reshape(x=kr, shape=[0, -1, self.num_heads, self.head_dim])
        kr = tensor.transpose(x=kr, perm=[1, 2, 0, 3])

        if isinstance(cache, self.StaticCache):
            # for encoder-decoder attention in inference and has cached
            k, v = cache.k, cache.v
        else:
            k, v = self.compute_kv(key, value)

        if isinstance(cache, self.Cache):
            # for decoder self-attention in inference
            k = tensor.concat([cache.k, k], axis=2)
            v = tensor.concat([cache.v, v], axis=2)
            cache = self.Cache(k, v)

        return (q, k, v, kr) if cache is None else (q, k, v, kr, cache)

    def compute_kv(self, key, value):
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
        k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = tensor.transpose(x=v, perm=[0, 2, 1, 3])
        return k, v

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
            k, v = self.compute_kv(key, value)
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

    def forward(self, query, key=None, value=None, rel_pos_start_key=0, attn_mask=None, cache=None):
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
        # rel_min: -20
        # rel_max: +10
        # emb_num: +30
        # rel_pos_start_key: -10
        nb = query.shape[0]
        l_q = query.shape[1] # 10
        l_k = key.shape[1] # 20
        rel_s = rel_pos_start_key - l_q # -20
        rel_e = rel_pos_start_key + l_k # +10
        
        pos_emb_s = self.rel_pos_layer.relative_pos_2_emb(rel_s) # 0
        pos_emb_e = self.rel_pos_layer.relative_pos_2_emb(rel_e) # 30
        max_pos_emb = self.rel_pos_layer.emb_num # 30
        max_pos_span = l_q + l_k # 30
        
        s_fill = pos_emb_s # 0
        e_fill = max_pos_emb - pos_emb_e # 0

        # compute q ,k ,v
        if cache is None:
            q, k, v, kr = self._prepare_qkv(query, key, value, cache)
        else:
            q, k, v, kr, cache = self._prepare_qkv(query, key, value, cache)

        alpha = self.head_dim**-0.5
        # scale dot product attention
        # TODO(guosheng): use tensor.matmul, however it doesn't support `alpha`
        # prod_cc: `[batch_size, num_heads, q_seq_length, k_seq_length]`
        prod_cc = alpha * paddle.matmul(x=q, y=k, transpose_y=True)
        # prod_cp
        # multiply `[batch_size, num_heads, q_seq_length, d_head]` and `[num_heads, nk, d_head]`
        # acquiring `[batch_size, num_heads, q_seq_length, max_pos_emb]`
        prod_cp = alpha * paddle.matmul(x=q, y=kr, transpose_y=True)

        # acquiring `[batch_size, num_heads, q_seq_length, max_pos_span]` 10, 30
        # by adding / slicing the rows out of range
        INT_MAX = 100000 # s_fill = -99, e_fill = 51
        if(s_fill > 0):
            prod_cp = paddle.slice(prod_cp, [0, 1, 2, 3], [0, 0, 0, s_fill], [INT_MAX, INT_MAX, INT_MAX, INT_MAX])
        elif(s_fill < 0):
            fill_vec_s = paddle.slice(prod_cp, [0, 1, 2, 3], [0, 0, 0, 0], [INT_MAX, INT_MAX, INT_MAX, 1])
            fill_vec_s = paddle.tile(fill_vec_s, repeat_times=[1, 1, 1, -s_fill])
            prod_cp = paddle.concat([fill_vec_s, prod_cp], axis=-1)
        if(e_fill > 0):
            prod_cp = paddle.slice(prod_cp, [0, 1, 2, 3], [0, 0, 0, 0], [INT_MAX, INT_MAX, INT_MAX, -e_fill])
        elif(e_fill < 0):
            fill_vec_e = paddle.slice(prod_cp, [0, 1, 2, 3], [0, 0, 0, -1], [INT_MAX, INT_MAX, INT_MAX, INT_MAX])
            fill_vec_e = paddle.tile(fill_vec_e, repeat_times=[1, 1, 1, -e_fill])
            prod_cp = paddle.concat([prod_cp, fill_vec_e], axis=-1)

        # Doing Relative shift
        prod_cp = paddle.reshape(prod_cp, shape=(nb, self.num_heads, max_pos_span, l_q))
        prod_cp = paddle.slice(prod_cp, [0, 1, 2, 3], [0, 0, 1, 0], [INT_MAX, INT_MAX, INT_MAX, INT_MAX])
        prod_cp = paddle.reshape(prod_cp, shape=(nb, self.num_heads, l_q, max_pos_span - 1))
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
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        if cache is not None:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)
