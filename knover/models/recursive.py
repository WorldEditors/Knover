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
"""Recursive Model."""

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from knover.models import register_model
from knover.core.model import Model
from knover.modules.generator import Generator
from knover.modules.recursive_cells import PlasticRNNCell, PlasticLSTMCell, ModulateRNNCell
from knover.utils import gather, str2bool


@register_model("RecursiveModels")
class RecursiveModels(Model):
    """Recursive Models"""

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
        self.hidden_act = args.hidden_act

        # embeddings
        self.vocab_size = args.vocab_size
        self.type_size = args.type_vocab_size
        self.token_embedding = nn.Embedding(self.vocab_size, self.emb_size, weight_attr=param_attr)
        self.type_embedding = nn.Embedding(self.type_size, self.emb_size, weight_attr=param_attr)

        # role embeddings
        self.use_role = args.use_role
        if self.use_role:
            self.role_embedding = nn.Embedding(args.role_type_size, self.emb_size, weight_attr=param_attr)

        print("hidden_size:", self.hidden_size, "emb_size:", self.emb_size)
        # embeding mapping
        if self.hidden_size != self.emb_size:
            self.emb_mapping_in = True
        else:
            self.emb_mapping_in = args.get("emb_mapping_in", False)
        if self.emb_mapping_in:
            self.emb_mapping_fc = nn.Linear(self.emb_size, self.hidden_size, weight_attr=param_attr)

        # encoder
        if(args.encoder_type == "PlasticRNN"):
            self._rnn_cell = PlasticRNNCell(self.hidden_size, self.hidden_size)
        elif(args.encoder_type == "NaiveRNN"):
            self._rnn_cell = nn.SimpleRNNCell(self.hidden_size, self.hidden_size)
        elif(args.encoder_type == "LSTM"):
            self._rnn_cell = nn.LSTMCell(self.hidden_size, self.hidden_size)
        elif(args.encoder_type == "PlasticLSTM"):
            self._rnn_cell = PlasticLSTMCell(self.hidden_size, self.hidden_size)
        elif(args.encoder_type == "ModulateRNN"):
            self._rnn_cell = ModulateRNNCell(self.hidden_size, self.hidden_size)
        else:
            raise Exception("No such encoder type or encoder type not defined: %s" % args.encoder_type)
        self.encoder = nn.RNN(self._rnn_cell)

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
        """Generate input embeddings of PlasticRNN

        Args:
            tokens_ids: represents the token id of each token, shape is [batch_size, max_seq_len, 1]
            type_ids: represents the type of each token, shape is [batch_size, max_seq_len, 1]
            aux_emb: represents the auxiliary input embeddings of PlasticRNN.

        Returns:
            A Tuple contains the input embeddings and the attention masking matrix of PlasticRNN.
        """
        token_emb_out = self.token_embedding(token_ids)
        type_emb_out = self.type_embedding(type_ids)
        emb_out = token_emb_out + type_emb_out

        if self.use_role:
            role_emb_out = self.role_embedding(role_ids)
            emb_out = emb_out + role_emb_out

        # concat auxiliary memory embeddings
        if aux_emb is not None:
            print(aux_emb.shape, emb_out.shape)
            emb_out = paddle.concat([aux_emb, emb_out], axis=1)

        if self.emb_mapping_in:
            emb_out = self.emb_mapping_fc(emb_out)

        return emb_out

    def _generation_network(self,
                            token_ids,
                            type_ids,
                            role_ids,
                            aux_emb=None):
        """Run PlasticRNN generation network.

        Args:
            tokens_ids: represents the token id of each token, shape is [batch_size, max_seq_len, 1]
            type_ids: represents the type of each token, shape is [batch_size, max_seq_len, 1]
            aux_emb: represents the auxiliary input embeddings of PlasticRNN.

        Returns:
            The output embeddings of PlasticRNN.
        """
        emb_input = self._gen_input(
            token_ids, type_ids, role_ids, aux_emb=aux_emb)
        enc_out, final_states = self._encode(emb_input)

        return enc_out

    def _generation_step(self, state):
        # gather caches
        enc_out = self._generation_network(
            state["token_ids"],
            state["type_ids"],
            state.get("role_ids", None),
        )
        logits = self._calc_logits(enc_out)
        return logits

    def _encode(self, emb_input):
        """Run PlasticRNN encode pass.

        Args:
            emb_input: represents the input embeddings fo PlasticRNN, shape is [batch_size, max_seq_len, hidden_dim]

        Returns:
            The output embeddings of PlasticRNN.
        """
        ret = self.encoder(emb_input)
        return ret

    def _calc_logits(self, enc_out, tgt_idx=None):
        """Get the logits of generation task.

        The network may share weight with token embeddings.

        Args:
            enc_out: the output embeddings of PlasticRNN, shape is [batch_size, max_seq_len, hidden_dim]
            tgt_idx (optional): the indices of prediction tokens, shape is [num_predictions, 2].

        Returns:
            logits: the logits of prediction task, shape is [num_predictions, vocab_size].
        """
        if tgt_idx is None:
            seq_feat = paddle.reshape(x=enc_out, shape=[-1, self.hidden_size])
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
        outputs["enc_out"] = self._generation_network(
            token_ids=inputs["token_ids"],
            type_ids=inputs["type_ids"],
            role_ids=inputs.get("role_ids", None)
        )
        return outputs

    def get_metrics(self, inputs, outputs):
        """Get metrics."""
        metrics = {}

        tgt_logits = self._calc_logits(outputs["enc_out"], inputs["tgt_idx"])
        mean_tgt_lm_loss = F.cross_entropy(tgt_logits, inputs["tgt_label"])
        metrics["token_lm_loss"] = mean_tgt_lm_loss

        loss = mean_tgt_lm_loss
        metrics["loss"] = loss
        return metrics

    def get_statistics(self, inputs, outputs):
        """Get statistics."""
        statistics = {}
        if "tgt_label" in inputs:
            statistics["tokens_num"] = inputs["tgt_label"].shape[0]
        statistics["batch_size"] = inputs["token_ids"].shape[0]
        return statistics

    def infer(self, inputs, outputs):
        """Run model inference.

        Only support generation now.
        """
        if self.do_generation:
            outputs = self.generator(self, inputs, outputs)
            data_id_list = outputs["data_id"].numpy()
            token_ids_list = outputs["token_ids"].numpy()
            seq_ids_list = outputs["finished_ids"].numpy()
            score_list = outputs["finished_score"].numpy()
            predictions = []
            for data_id, token_ids, seq_ids, score in zip(data_id_list, token_ids_list, seq_ids_list, score_list):
                if len(seq_ids.shape) == 1:
                    pred = {}
                    pred["data_id"] = int(data_id)
                    pred["decode_score"] = float(score)
                    pred["context_token_ids"] = token_ids
                    pred["response_token_ids"] = seq_ids
                    predictions.append(pred)
                else:
                    for candidate_seq_ids, candidate_score in zip(seq_ids, score):
                        pred = {}
                        pred["data_id"] = int(data_id)
                        pred["decode_score"] = float(candidate_score)
                        pred["context_token_ids"] = token_ids
                        pred["response_token_ids"] = candidate_seq_ids
                        predictions.append(pred)
            return predictions
        else:
            raise NotImplementedError