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

__all__ = []


class TransformerEncoderLayer(Layer):

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

    def forward(self, src, src_mask=None, cache=None):
        src_mask = _convert_attention_mask(src_mask, src.dtype)

        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        # Add cache for encoder for the usage like UniLM
        if cache is None:
            src = self.self_attn(src, src, src, src_mask)
        else:
            src, incremental_cache = self.self_attn(src, src, src, src_mask,
                                                    cache)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src if cache is None else (src, incremental_cache)

    def gen_cache(self, src):
        incremental_cache = self.self_attn.gen_cache(
            src, type=self.self_attn.Cache)
        return incremental_cache


class TransformerEncoder(Layer):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = LayerList([(encoder_layer if i == 0 else
                                  type(encoder_layer)(**encoder_layer._config))
                                 for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, cache=None):
        src_mask = _convert_attention_mask(src_mask, src.dtype)

        output = src
        new_caches = []
        for i, mod in enumerate(self.layers):
            if cache is None:
                output = mod(output, src_mask=src_mask)
            else:
                output, new_cache = mod(output,
                                        src_mask=src_mask,
                                        cache=cache[i])
                new_caches.append(new_cache)

        if self.norm is not None:
            output = self.norm(output)

        return output if cache is None else (output, new_caches)

    def gen_cache(self, src):
        cache = [layer.gen_cache(src) for layer in self.layers]
        return cache


class TransformerDecoderLayer(Layer):
    """
    TransformerDecoderLayer is composed of three sub-layers which are decoder
    self (multi-head) attention, decoder-encoder cross attention and feedforward
    network. Before and after each sub-layer, pre-process and post-precess would
    be applied on the input and output accordingly. If `normalize_before` is True,
    pre-process is layer normalization and post-precess includes dropout, residual
    connection. Otherwise, no pre-process and post-precess includes dropout, residual
    connection, layer normalization.

    Parameters:
        d_model (int): The expected feature size in the input and output.
        nhead (int): The number of heads in multi-head attention(MHA).
        dim_feedforward (int): The hidden layer size in the feedforward network(FFN).
        dropout (float, optional): The dropout probability used in pre-process
            and post-precess of MHA and FFN sub-layer. Default 0.1
        activation (str, optional): The activation function in the feedforward
            network. Default relu.
        attn_dropout (float, optional): The dropout probability used
            in MHA to drop some attention target. If None, use the value of
            `dropout`. Default None
        act_dropout (float, optional): The dropout probability used after FFN
            activition.  If None, use the value of `dropout`. Default None
        normalize_before (bool, optional): Indicate whether to put layer normalization
            into preprocessing of MHA and FFN sub-layers. If True, pre-process is layer
            normalization and post-precess includes dropout, residual connection.
            Otherwise, no pre-process and post-precess includes dropout, residual
            connection, layer normalization. Default False
        weight_attr(ParamAttr|list|tuple, optional): To specify the weight parameter property.
            If it is a list/tuple, `weight_attr[0]` would be used as `weight_attr` for
            self attention, `weight_attr[1]` would be used as `weight_attr` for
            cross attention, and `weight_attr[2]` would be used as `weight_attr`
            for linear in FFN. Otherwise, the three sub-layers all uses it as
            `weight_attr` to create parameters. Default: None, which means the
            default weight parameter property is used. See usage for details
            in :ref:`api_paddle_fluid_param_attr_ParamAttr` . 
        bias_attr (ParamAttr|list|tuple|bool, optional): To specify the bias parameter property.
            If it is a list/tuple, `bias_attr[0]` would be used as `bias_attr` for
            self attention, `bias_attr[1]` would be used as `bias_attr` for
            cross attention, and `bias_attr[2]` would be used as `bias_attr`
            for linear in FFN. Otherwise, the three sub-layers all uses it as
            `bias_attr` to create parameters. The `False` value means the
            corresponding layer would not have trainable bias parameter. See
            usage for details in :code:`ParamAttr` . Default: None,which means
            the default bias parameter property is used.

    Examples:

        .. code-block:: python

            import paddle
            from paddle.nn import TransformerDecoderLayer

            # decoder input: [batch_size, tgt_len, d_model]
            dec_input = paddle.rand((2, 4, 128))
            # encoder output: [batch_size, src_len, d_model]
            enc_output = paddle.rand((2, 6, 128))
            # self attention mask: [batch_size, n_head, tgt_len, tgt_len]
            self_attn_mask = paddle.rand((2, 2, 4, 4))
            # cross attention mask: [batch_size, n_head, tgt_len, src_len]
            cross_attn_mask = paddle.rand((2, 2, 4, 6))
            decoder_layer = TransformerDecoderLayer(128, 2, 512)
            output = decoder_layer(dec_input,
                                   enc_output,
                                   self_attn_mask,
                                   cross_attn_mask)  # [2, 4, 128]
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

        super(TransformerDecoderLayer, self).__init__()

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

        weight_attrs = _convert_param_attr_to_list(weight_attr, 3)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 3)

        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0])
        self.cross_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[1],
            bias_attr=bias_attrs[1])
        self.linear1 = Linear(
            d_model, dim_feedforward, weight_attrs[2], bias_attr=bias_attrs[2])
        self.dropout = Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = Linear(
            dim_feedforward, d_model, weight_attrs[2], bias_attr=bias_attrs[2])
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = Dropout(dropout, mode="upscale_in_train")
        self.dropout3 = Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, cache=None):
        r"""
        Applies a Transformer decoder layer on the input.

        Parameters:
            tgt (Tensor): The input of Transformer decoder layer. It is a tensor
                with shape `[batch_size, target_length, d_model]`. The data type
                should be float32 or float64.
            memory (Tensor): The output of Transformer encoder. It is a tensor
                with shape `[batch_size, source_length, d_model]`. The data type
                should be float32 or float64.
            tgt_mask (Tensor, optional): A tensor used in self attention
                to prevents attention to some unwanted positions, usually the
                the subsequent positions. It is a tensor with shape broadcasted
                to `[batch_size, n_head, target_length, target_length]`.
                When the data type is bool, the unwanted positions have `False` 
                values and the others have `True` values. When the data type is 
                int, the unwanted positions have 0 values and the others have 1 
                values. When the data type is float, the unwanted positions have 
                `-INF` values and the others have 0 values. It can be None when 
                nothing wanted or needed to be prevented attention to. Default None.
            memory_mask (Tensor, optional): A tensor used in decoder-encoder
                cross attention to prevents attention to some unwanted positions,
                usually the paddings. It is a tensor with shape broadcasted to 
                `[batch_size, n_head, target_length, source_length]`. When the 
                data type is bool, the unwanted positions have `False` values 
                and the others have `True` values. When the data type is int, 
                the unwanted positions have 0 values and the others have 1 
                values. When the data type is float, the unwanted positions have 
                `-INF` values and the others have 0 values. It can be None when 
                nothing wanted or needed to be prevented attention to. Default None.
            cache (tuple, optional): It is a tuple( :code:`(incremental_cache, static_cache)` ),
                `incremental_cache` is an instance of `MultiHeadAttention.Cache`,
                `static_cache` is an instance of `MultiHeadAttention.StaticCache.
                See `TransformerDecoderLayer.gen_cache` for more details. It is
                only used for inference and should be None for training. Default
                None.

        Returns:
            Tensor|tuple: It is a tensor that has the same shape and data type \
                as `tgt`, representing the output of Transformer decoder layer. \
                Or a tuple if `cache` is not None, except for decoder layer output, \
                the tuple includes the new cache which is same as input `cache` \
                argument but `incremental_cache` in it has an incremental length. \
                See `MultiHeadAttention.gen_cache` and `MultiHeadAttention.forward` \
                for more details.
        """
        tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)
        memory_mask = _convert_attention_mask(memory_mask, memory.dtype)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        if cache is None:
            tgt = self.self_attn(tgt, tgt, tgt, tgt_mask, None)
        else:
            tgt, incremental_cache = self.self_attn(tgt, tgt, tgt, tgt_mask,
                                                    cache[0])
        tgt = residual + self.dropout1(tgt)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        if cache is None:
            tgt = self.cross_attn(tgt, memory, memory, memory_mask, None)
        else:
            tgt, static_cache = self.cross_attn(tgt, memory, memory,
                                                memory_mask, cache[1])
        tgt = residual + self.dropout2(tgt)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm3(tgt)
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = residual + self.dropout3(tgt)
        if not self.normalize_before:
            tgt = self.norm3(tgt)
        return tgt if cache is None else (tgt, (incremental_cache,
                                                static_cache))

    def gen_cache(self, memory):
        r"""
        Generates cache for `forward` usage. The generated cache is a tuple
        composed of an instance of `MultiHeadAttention.Cache` and an instance
        of `MultiHeadAttention.StaticCache`.

        Parameters:
            memory (Tensor): The output of Transformer encoder. It is a tensor
                with shape `[batch_size, source_length, d_model]`. The data type
                should be float32 or float64.

        Returns:
            tuple: It is a tuple( :code:`(incremental_cache, static_cache)` ). \
                `incremental_cache` is an instance of `MultiHeadAttention.Cache` \
                produced by `self_attn.gen_cache(memory, MultiHeadAttention.Cache)`, \
                it reserves two tensors shaped `[batch_size, nhead, 0, d_model // nhead]`. \
                `static_cache` is an instance of `MultiHeadAttention.StaticCache` \
                produced by `cross_attn.gen_cache(memory, MultiHeadAttention.StaticCache)`, \
                it reserves two tensors shaped `[batch_size, nhead, source_length, d_model // nhead]`.
                See `MultiHeadAttention.gen_cache` and `MultiHeadAttention.forward` \
                for more details.
        """
        incremental_cache = self.self_attn.gen_cache(
            memory, type=self.self_attn.Cache)
        static_cache = self.cross_attn.gen_cache(
            memory, memory, type=self.cross_attn.StaticCache)
        return incremental_cache, static_cache


class TransformerDecoder(Layer):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = LayerList([(decoder_layer if i == 0 else
                                  type(decoder_layer)(**decoder_layer._config))
                                 for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, cache=None):
        tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)
        memory_mask = _convert_attention_mask(memory_mask, memory.dtype)

        output = tgt
        new_caches = []
        for i, mod in enumerate(self.layers):
            if cache is None:
                output = mod(output,
                             memory,
                             tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             cache=None)
            else:
                output, new_cache = mod(output,
                                        memory,
                                        tgt_mask=tgt_mask,
                                        memory_mask=memory_mask,
                                        cache=cache[i])
                new_caches.append(new_cache)

        if self.norm is not None:
            output = self.norm(output)

        return output if cache is None else (output, new_caches)

    def gen_cache(self, memory, do_zip=False):
        cache = [layer.gen_cache(memory) for layer in self.layers]
        if do_zip:
            cache = list(zip(*cache))
        return cache


class Transformer(Layer):
    """
    A Transformer model composed of an instance of `TransformerEncoder` and an
    instance of `TransformerDecoder`. While the embedding layer and output layer
    are not included.

    Please refer to `Attention is all you need <http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf>`_ ,
    and see `TransformerEncoder` and `TransformerDecoder` for more details.
    
    Users can configurate the model architecture with corresponding parameters.
    Note the usage of `normalize_before` representing where to apply layer
    normalization (in pre-process or post-precess of multi-head attention or FFN),
    and some transformer like models are different on this, such as
    `BERT <https://arxiv.org/abs/1810.04805>`_ and `GPT2 <https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf>`_ . 
    The default architecture here places layer normalization in post-process and
    applies another layer normalization on the output of last encoder/decoder layer.

    Parameters:
        d_model (int, optional): The expected feature size in the encoder/decoder input
            and output. Default 512
        nhead (int, optional): The number of heads in multi-head attention(MHA). Default 8
        num_encoder_layers (int, optional): The number of layers in encoder. Default 6
        num_decoder_layers (int, optional): The number of layers in decoder. Default 6
        dim_feedforward (int, optional): The hidden layer size in the feedforward network(FFN). Default 2048
        dropout (float, optional): The dropout probability used in pre-process
            and post-precess of MHA and FFN sub-layer. Default 0.1
        activation (str, optional): The activation function in the feedforward
            network. Default relu.
        attn_dropout (float, optional): The dropout probability used
            in MHA to drop some attention target. If None, use the value of
            `dropout`. Default None
        act_dropout (float, optional): The dropout probability used after FFN
            activition.  If None, use the value of `dropout`. Default None
        normalize_before (bool, optional): Indicate whether to put layer normalization
            into preprocessing of MHA and FFN sub-layers. If True, pre-process is layer
            normalization and post-precess includes dropout, residual connection.
            Otherwise, no pre-process and post-precess includes dropout, residual
            connection, layer normalization. Default False
        weight_attr(ParamAttr|list|tuple, optional): To specify the weight parameter property.
            If it is a list/tuple, the length of `weight_attr` could be 1, 2 or 3. If it is 3, 
            `weight_attr[0]` would be used as `weight_attr` for self attention, `weight_attr[1]` 
            would be used as `weight_attr` for cross attention of `TransformerDecoder`, 
            and `weight_attr[2]` would be used as `weight_attr` for linear in FFN. 
            If it is 2, `weight_attr[0]` would be used as `weight_attr` both for self attention 
            and cross attntion and `weight_attr[1]` would be used as `weight_attr` for 
            linear in FFN. If it is 1, `weight_attr[0]` would be used as `weight_attr` 
            for self attention, cross attention and linear in FFN. Otherwise, 
            the three sub-layers all uses it as `weight_attr` to create parameters. 
            Default: None, which means the default weight parameter property is used. 
            See usage for details
            in :code:`ParamAttr` . 
        bias_attr (ParamAttr|list|tuple|bool, optional): To specify the bias parameter property.
            If it is a list/tuple, the length of `bias_attr` could be 1, 2 or 3. If it is 3, 
            `bias_attr[0]` would be used as `bias_attr` for self attention, `bias_attr[1]` 
            would be used as `bias_attr` for cross attention of `TransformerDecoder`, 
            and `bias_attr[2]` would be used as `bias_attr` for linear in FFN. 
            If it is 2, `bias_attr[0]` would be used as `bias_attr` both for self attention 
            and cross attntion and `bias_attr[1]` would be used as `bias_attr` for 
            linear in FFN. If it is 1, `bias_attr[0]` would be used as `bias_attr` 
            for self attention, cross attention and linear in FFN. Otherwise, 
            the three sub-layers all uses it as `bias_attr` to create parameters. 
            The `False` value means the corresponding layer would not have trainable 
            bias parameter. See usage for details in :code:`ParamAttr` . 
            Default: None,which means the default bias parameter property is used.
        custom_encoder (Layer, optional): If custom encoder is provided, use it as the encoder.
            Default None
        custom_decoder (Layer, optional): If custom decoder is provided, use it as the decoder.
            Default None

    Examples:

        .. code-block:: python

            import paddle
            from paddle.nn import Transformer

            # src: [batch_size, tgt_len, d_model]
            enc_input = paddle.rand((2, 4, 128))
            # tgt: [batch_size, src_len, d_model]
            dec_input = paddle.rand((2, 6, 128))
            # src_mask: [batch_size, n_head, src_len, src_len]
            enc_self_attn_mask = paddle.rand((2, 2, 4, 4))
            # tgt_mask: [batch_size, n_head, tgt_len, tgt_len]
            dec_self_attn_mask = paddle.rand((2, 2, 6, 6))
            # memory_mask: [batch_size, n_head, tgt_len, src_len]
            cross_attn_mask = paddle.rand((2, 2, 6, 4))
            transformer = Transformer(128, 2, 4, 4, 512)
            output = transformer(enc_input,
                                 dec_input,
                                 enc_self_attn_mask,
                                 dec_self_attn_mask,
                                 cross_attn_mask)  # [2, 6, 128]
    """

    def __init__(self,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 weight_attr=None,
                 bias_attr=None,
                 custom_encoder=None,
                 custom_decoder=None):
        super(Transformer, self).__init__()

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

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation,
                attn_dropout, act_dropout, normalize_before,
                encoder_weight_attr, encoder_bias_attr)
            encoder_norm = LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers,
                                              encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation,
                attn_dropout, act_dropout, normalize_before,
                decoder_weight_attr, decoder_bias_attr)
            decoder_norm = LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers,
                                              decoder_norm)

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        r"""
        Applies a Transformer model on the inputs.

        Parameters:
            src (Tensor): The input of Transformer encoder. It is a tensor
                with shape `[batch_size, source_length, d_model]`. The data type
                should be float32 or float64.
            tgt (Tensor): The input of Transformer decoder. It is a tensor
                with shape `[batch_size, target_length, d_model]`. The data type
                should be float32 or float64.
            memory (Tensor): The output of Transformer encoder. It is a tensor
                with shape `[batch_size, source_length, d_model]`. The data type
                should be float32 or float64.
            src_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False` 
                values and the others have `True` values. When the data type is 
                int, the unwanted positions have 0 values and the others have 1 
                values. When the data type is float, the unwanted positions have 
                `-INF` values and the others have 0 values. It can be None when 
                nothing wanted or needed to be prevented attention to. Default None.
            tgt_mask (Tensor, optional): A tensor used in self attention
                to prevents attention to some unwanted positions, usually the
                the subsequent positions. It is a tensor with shape broadcasted
                to `[batch_size, n_head, target_length, target_length]`. When 
                the data type is bool, the unwanted positions have `False` 
                values and the others have `True` values. When the data type is 
                int, the unwanted positions have 0 values and the others have 1 
                values. When the data type is float, the unwanted positions have 
                `-INF` values and the others have 0 values. It can be None when 
                nothing wanted or needed to be prevented attention to. Default None.
            memory_mask (Tensor, optional): A tensor used in decoder-encoder
                cross attention to prevents attention to some unwanted positions,
                usually the paddings. It is a tensor with shape broadcasted to
                `[batch_size, n_head, target_length, source_length]`. When the 
                data type is bool, the unwanted positions have `False` values 
                and the others have `True` values. When the data type is int, 
                the unwanted positions have 0 values and the others have 1 
                values. When the data type is float, the unwanted positions have 
                `-INF` values and the others have 0 values. It can be None when 
                nothing wanted or needed to be prevented attention to. Default None.

        Returns:
            Tensor: It is a tensor that has the same shape and data type \
                as `tgt`, representing the output of Transformer decoder.
        """
        src_mask = _convert_attention_mask(src_mask, src.dtype)
        memory = self.encoder(src, src_mask=src_mask)

        tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)
        memory_mask = _convert_attention_mask(memory_mask, memory.dtype)
        output = self.decoder(
            tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return output

    def generate_square_subsequent_mask(self, length):
        """
        Generate a square mask for the sequence. The mask ensures that the
        predictions for position i can depend only on the known outputs at
        positions less than i.

        Parameters:
            length (int|Tensor): The length of sequence.

        Returns:
            Tensor: Generated square mask according to the given length.

        Examples:
            .. code-block:: python

                import paddle
                from paddle.nn.layer.transformer import Transformer
                length = 5
                d_model, n_head, dim_feedforward = 8, 4, 64
                transformer_paddle = Transformer(
                    d_model, n_head, dim_feedforward=dim_feedforward)
                mask = transformer_paddle.generate_square_subsequent_mask(length)
                print(mask)

                # [[  0. -inf -inf -inf -inf]
                # [  0.   0. -inf -inf -inf]
                # [  0.   0.   0. -inf -inf]
                # [  0.   0.   0.   0. -inf]
                # [  0.   0.   0.   0.   0.]]

        """
        return paddle.tensor.triu(
            (paddle.ones(
                (length, length), dtype=paddle.get_default_dtype()) * -np.inf),
            1)
