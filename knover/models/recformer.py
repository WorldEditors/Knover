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
"""Unified Transformer model."""

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Layer, LayerList

from knover.modules.rel_pos_attention import RelPosLayer
from knover.modules.recformer_block import LongTermMemEncoder, MemAugDecoderLayer, MemAugDecoder
from knover.models import register_model
from knover.core.model import Model
from knover.modules.generator import Generator
from knover.utils import gather, str2bool

INT_MAX=999999

@register_model("RecFormer")
class RecFormer(Model):
    """RecFormer"""

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline arguments."""
        group = Model.add_cmdline_args(parser)
        group.add_argument("--weight_sharing", type=str2bool, default=True,
                           help="Whether to share the token embedding with the output FC.")
        group.add_argument("--use_role", type=str2bool, default=False,
                           help="Whether use role embeddings.")

        Generator.add_cmdline_args(parser)
        return group

    def _build_model(self, args):
        self.max_seq_len = args.max_seq_len

        # initializer
        initializer = paddle.nn.initializer.TruncatedNormal(std=args.initializer_range)
        param_attr = paddle.ParamAttr(initializer=initializer)

        self.emb_size = args.get("emb_size", args.hidden_size)
        self.hidden_size = args.hidden_size
        self.dropout = args.hidden_dropout_prob

        self.n_layer = args.num_hidden_layers
        self.n_head = args.num_attention_heads
        self.d_key = args.get("key_size", self.hidden_size // self.n_head)
        self.d_value = args.get("value_size", self.hidden_size // self.n_head)
        self.inner_hidden_size = args.get("inner_hidden_size", self.hidden_size * 4)

        self.memory_length = args.get("memory_length", 256)
        self.max_segment_length = args.get("max_segment_length", 256)
        self.aux_loss_weight = args.get("auxiliary_loss_weight", 0.1)
        self.use_absolute_position = args.get("use_absolute_position", True)

        max_mem_len = self.memory_length +  self.max_segment_length

        self.dec_relative_position_min = args.get("dec_relative_position_min", - (max_mem_len + self.max_segment_length))
        self.dec_relative_position_max = args.get("dec_relative_position_max", 0)

        self.enc_relative_position_min = args.get("enc_relative_position_min", -self.memory_length)
        self.enc_relative_position_max = args.get("enc_relative_position_max", max_mem_len)

        # embeddings
        self.vocab_size = args.vocab_size
        self.type_size = args.type_vocab_size
        if(self.type_size < 1):
            self.type_embedding = None
        else:
            self.type_embedding = nn.Embedding(self.type_size, self.emb_size, weight_attr=param_attr)

        self.token_embedding = nn.Embedding(self.vocab_size, self.emb_size, weight_attr=param_attr)
        if(self.use_absolute_position):
            self.pos_embedding = nn.Embedding(self.max_segment_length, self.emb_size, weight_attr=param_attr)
            print("use both relative position and absolute position")
        else:
            self.pos_embedding = None

        # Use relative position layer or position embedding depending on use relative position or not
        self.rel_pos_layer_enc = RelPosLayer(rel_min=self.enc_relative_position_min, 
                rel_max = self.enc_relative_position_max, 
                emb_size = self.hidden_size, weight_attr=param_attr)

        # Same for decoder
        self.rel_pos_layer_dec = RelPosLayer(rel_min=self.dec_relative_position_min, 
                rel_max = self.dec_relative_position_max, 
                emb_size = self.hidden_size, weight_attr=param_attr)

        # embeddings
        self.vocab_size = args.vocab_size
        self.type_size = args.type_vocab_size
        if(self.type_size < 1):
            self.type_embedding = None
        else:
            self.type_embedding = nn.Embedding(self.type_size, self.emb_size, weight_attr=param_attr)

        self.token_embedding = nn.Embedding(self.vocab_size, self.emb_size, weight_attr=param_attr)

        # role embeddings
        self.use_role = args.use_role
        if self.use_role:
            self.role_embedding = nn.Embedding(args.role_type_size, self.emb_size, weight_attr=param_attr)

        # embeding mapping
        if self.hidden_size != self.emb_size:
            self.emb_mapping_in = True
        else:
            self.emb_mapping_in = args.get("emb_mapping_in", False)
        if self.emb_mapping_in:
            self.emb_mapping_fc = nn.Linear(self.emb_size, self.hidden_size, weight_attr=param_attr)

        # LTM encoder
        self.normalize_before = args.get("normalize_before", True)
        self.hidden_act = args.hidden_act
        self.encoders = LayerList([LongTermMemEncoder(
                self.hidden_size, self.n_head, self.inner_hidden_size, rel_pos_layer=self.rel_pos_layer_enc, dropout=self.dropout, activation=self.hidden_act,
                act_dropout=0, normalize_before=self.normalize_before, weight_attr=param_attr)
                for i in range(self.n_layer)])

        # Memory Augmented Decoder
        self.decoder_layer = MemAugDecoderLayer(
            self.hidden_size, self.n_head, self.inner_hidden_size, rel_pos_layer=self.rel_pos_layer_dec, dropout=self.dropout, activation=self.hidden_act,
            act_dropout=0, normalize_before=self.normalize_before, weight_attr=param_attr)
        self.decoder = MemAugDecoder(self.decoder_layer, self.hidden_size, self.n_layer, normalize_before=self.normalize_before)

        # lm head
        self.lm_trans_fc = nn.Linear(self.hidden_size, self.hidden_size, weight_attr=param_attr)
        self.activation = getattr(F, self.hidden_act)
        self.lm_trans_norm = nn.LayerNorm(self.hidden_size)
        self.weight_sharing = args.weight_sharing
        if self.weight_sharing:
            self.lm_logits_bias = paddle.create_parameter(
                [self.vocab_size], "float32", name="lm_out_fc.b_0", is_bias=True)
        else:
            self.lm_out_fc = nn.Linear(self.emb_size, self.emb_size, weight_attr=param_attr)

        # task-related
        from knover.modules.generator import GENERATOR_REGISTRY
        generator_cls = GENERATOR_REGISTRY[args.decoding_strategy]
        self.generator = generator_cls(args)
        self.do_generation = args.get("do_generation", False)

    def _gen_input(self,
                   token_ids,
                   type_ids,
                   role_ids,
                   aux_emb=None):
        """Generate input embeddings of Transformer

        Args:
            tokens_ids: represents the token id of each token, shape is [batch_size, max_seq_len, 1]
            type_ids: represents the type of each token, shape is [batch_size, max_seq_len, 1]
            aux_emb: represents the auxiliary input embeddings of Transformer.

        Returns:
            A Tuple contains the input embeddings and the attention masking matrix of Transformer.
        """
        batch_size = token_ids.shape[0]
        segment_length = token_ids.shape[1]
        if(segment_length > self.max_segment_length):
            raise Exception("segment length %f, max segment length: %f"%(segment_length, self.max_segment_length))
        emb_out = self.token_embedding(token_ids)

        if(self.pos_embedding is not None):
            pos_ids = paddle.tile(paddle.arange(segment_length), [batch_size, 1])
            pos_emb_out = self.pos_embedding(pos_ids)
            emb_out = emb_out + pos_emb_out

        if(self.type_embedding is not None):
            type_emb_out = self.type_embedding(type_ids)
            emb_out = emb_out + type_emb_out

        if self.use_role:
            role_emb_out = self.role_embedding(role_ids)
            emb_out = emb_out + role_emb_out

        # concat auxiliary memory embeddings
        if aux_emb is not None:
            emb_out = paddle.concat([aux_emb, emb_out], axis=1)

        if self.emb_mapping_in:
            emb_out = self.emb_mapping_fc(emb_out)

        return emb_out

    def _generation_network(self,
                            token_ids,
                            type_ids,
                            role_ids,
                            aux_emb=None):
        """Run Transformer generation network.

        Args:
            tokens_ids: represents the token id of each token, shape is [batch_size, max_seq_len, 1]
            type_ids: represents the type of each token, shape is [batch_size, max_seq_len, 1]
            aux_emb: represents the auxiliary input embeddings of Transformer.

        Returns:
            The output embeddings of Transformer.
        """
        emb_input = self._gen_input(
            token_ids, type_ids, role_ids, aux_emb=aux_emb)
        hid_pairs = (None, None)
        if self._generation_caches is None:
            enc_out, hid_pairs = self._encode(emb_input)
        else:
            enc_out, self._generation_caches = self._encode(
                emb_input, self._generation_caches)
        return enc_out, hid_pairs

    def _generation_step(self, state):
        # gather caches
        if "parent_idx" in state:
            raise Exception("RecFormer (Transformer-XL) can not support beam search currently")
        enc_out, hid_pairs = self._generation_network(
            state["token_ids"],
            state["type_ids"],
            state.get("role_ids", None),
        )
        logits = self._calc_logits(enc_out)
        return logits

    def _encode(self, emb_input, caches=None):
        """Run Transformer encode pass.

        Args:
            emb_input: represents the input embeddings fo Transformer, shape is [batch_size, max_seq_len, hidden_dim]
            caches: Dict of {"seg_id": n_seg_id,
                "memories": long_term_memories,
                "kv": key_value_cache_for_decoder
            }

        Returns:
            The output embeddings of Transformer.
        """
        seg_len = emb_input.shape[1]

        # Output size [Batch, SegLen, HiddenDim]
        mask = paddle.tensor.triu((paddle.ones(
            (seg_len, seg_len), dtype=paddle.get_default_dtype()) * -np.inf), 1)

        if(caches is not None):
            hids, output, caches = self.decoder(self.memories, emb_input,
                    additional_memories=self.st_memories[:-1],
                    detach_memory=True,
                    tgt_mask=mask, 
                    cache=caches)
        else:
            hids, output = self.decoder(self.memories, emb_input,
                    additional_memories=self.st_memories[:-1],
                    detach_memory=True,
                    tgt_mask=mask)

        self.update_memories(hids)

        # Use the updated memory to recalculate the attentions
        if(caches is None):
            hids_cont, output_cont = self.decoder(self.memories, emb_input, tgt_mask=mask, 
                    detach_memory=False,
                    detach_tgt=True,
                    parameter_no_grad=True)

        aux_mse_loss = paddle.mean(F.mse_loss(hids[1], hids_cont[1], reduction="none"), axis=-1)
        for i in range(1, self.n_layer):
            aux_mse_loss += paddle.mean(F.mse_loss(hids[i+1], hids_cont[i+1], reduction="none"), axis=-1)
        aux_mse_loss *= 1.0 / self.n_layer

        # Return outputs
        self.detach_memories()
        if caches is None:
            return output, aux_mse_loss
        else:
            return output, caches

    def _calc_logits(self, enc_out, tgt_idx=None):
        """Get the logits of generation task.

        The network may share weight with token embeddings.

        Args:
            enc_out: the output embeddings of Transformer, shape is [batch_size, max_seq_len, hidden_dim]
            tgt_idx (optional): the indices of prediction tokens, shape is [num_predictions, 2].

        Returns:
            logits: the logits of prediction task, shape is [num_predictions, vocab_size].
        """
        if tgt_idx is None:
            seq_feat = enc_out
        elif len(tgt_idx.shape) == 2 and tgt_idx.shape[1] == 2:
            seq_feat = paddle.gather_nd(enc_out, tgt_idx)
        else:
            raise ValueError(f"Invalid indices shape {tgt_idx.shape} is used")

        seq_trans_feat = self.lm_trans_fc(seq_feat)
        seq_trans_feat = self.activation(seq_trans_feat)
        seq_trans_feat = self.lm_trans_norm(seq_trans_feat)

        if self.weight_sharing:
            logits = paddle.matmul(
                seq_trans_feat, self.token_embedding.weight, transpose_y=True)
            logits += self.lm_logits_bias
        else:
            logits = self.lm_out_fc(seq_trans_feat)
        return logits

    def forward(self, inputs, is_infer=False):
        """Run model main forward."""
        outputs = {}
        if is_infer:
            self._generation_caches = self.decoder.gen_cache(self.memories)
        else:
            self._generation_caches = None

        outputs["enc_out"], outputs["aux_loss"] = self._generation_network(
            token_ids=inputs["token_ids"],
            type_ids=inputs["type_ids"],
            role_ids=inputs.get("role_ids", None),
        )
        return outputs

    def get_metrics(self, inputs, outputs):
        """Get metrics."""
        metrics = {}

        if "tgt_idx" in inputs:
            raise Exception("Target Idx can not be used for recformer")
        else:
            tgt_logits = self._calc_logits(outputs["enc_out"])
            tgt_lm_loss = F.cross_entropy(tgt_logits, inputs["tgt_label"], reduction="none")

            metrics["sum_tokens_logp"] = paddle.sum(tgt_lm_loss * inputs["loss_mask"])
            metrics["tokens_num"] = paddle.sum(inputs["loss_mask"])
            metrics["sum_tokens_aux_mse"] = paddle.sum(outputs["aux_loss"] * inputs["loss_mask"])

            metrics["loss"] = metrics["sum_tokens_logp"] / metrics["tokens_num"] + self.aux_loss_weight * metrics["sum_tokens_aux_mse"] / metrics["tokens_num"]

        return metrics

    def get_statistics(self, inputs, outputs):
        """Get statistics."""
        statistics = {}
        #if "tgt_label" in inputs:
        #    statistics["tokens_num"] = outputs["tokens_num"]
        statistics["batch_size"] = inputs["token_ids"].shape[0]
        return statistics

    def infer(self, inputs, outputs):
        """Run model inference.

        Only support generation now.
        """
        raise NotImplementedError

    def reset_memories(self, batch_size):
        self.memories = [paddle.full(shape=[int(batch_size), self.memory_length, self.hidden_size], fill_value=0.0)] * self.n_layer
        self.st_memories = [paddle.full(shape=[int(batch_size), self.memory_length, self.hidden_size], fill_value=0.0)] * (self.n_layer + 1)
        self.valid_st = 0

    def update_memories(self, hids):
        #update short term memory
        obs_memories = []
        new_st_memories = []
        batch_size, seq_len, _ = hids[0].shape
        self.valid_st += seq_len
        if(self.valid_st > self.memory_length):
            obs_st_len = self.valid_st - self.memory_length
            self.valid_st = self.memory_length
        else:
            obs_st_len = 0
        for i,hid in enumerate(hids):
            obs_mem, new_st_mem = paddle.chunk(paddle.concat([self.st_memories[i], hid], axis=1), chunks=[seq_len, self.memory_length], axis=1)
            new_st_memories.append(new_st_mem)
            if(obs_st_len > 0):
                obs_memories.append(paddle.slice(obs_mem, axes=[1], starts=[-obs_st_len], ends=[seq_len]))
        self.st_memories = new_st_memories

        #update long term memory
        if(obs_st_len > 0):
            new_lt_mems = []
            for i, layer in enumerate(self.encoders):
                new_lt_mems.append(layer(self.memories[i], obs_memories[i]))
            self.memories = new_lt_mems

    def detach_memories(self):
        self.memories = [mem.detach() for mem in self.memories]
