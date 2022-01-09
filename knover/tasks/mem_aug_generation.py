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
"""Dialogue generation task."""

from collections import defaultdict
import math
import paddle
import numpy as np

from knover.core.task import Task
from knover.tasks.dialog_generation import DialogGeneration
from knover.data.dialog_reader import DialogReader
from knover.data.plato_reader import PlatoReader
from knover.data.longtext_reader import LongTextReader
from knover.tasks import register_task
from knover.core.model import ModelInterface


@register_task("MemAugGeneration")
class MemAugGeneration(DialogGeneration):
    """Define XL dialogue response generation task."""
    def __init__(self, args):
        super(MemAugGeneration, self).__init__(args)
        self.reader = LongTextReader(args)
        self.validation_step = args.validation_step
        self.step = 0
        if(self.reader.use_role):
            self.train_keys = ["token_ids", "type_ids", "role_ids", "tgt_label", "loss_mask"]
        else:
            self.train_keys = ["token_ids", "type_ids", "tgt_label", "loss_mask"]

        return

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline arguments."""
        group = parser.add_argument_group("Task")
        group.add_argument("--segment_length", nargs='+', type=int, default=256,
                help="length of each segments")
        group.add_argument("--validation_step", type=int, default=1000,
                help="use inner validation steps")
        DialogGeneration.add_cmdline_args(parser)

    def train_step(self, model: ModelInterface, inputs):
        """Run one training step."""
        inputs = dict(zip(model.model.feed_names, inputs))
        
        model.model.reset_memories(inputs["batch_size"])
        outputs = {"valid_sum_logp": 0.0, "valid_tokens":0.0, "loss":0.0}
        seg_num = inputs["seg_num"]
        sum_lm_mask = 0.0

        for i in range(inputs["seg_num"]):
            #avoiding large memories, do some detach
            seg_len = inputs["segment_lengths"][i]
            seg_input = {k:inputs[k][i, :, :seg_len] for k in self.train_keys}
            tmp_outputs = model.train_step(seg_input)
            self.step += 1

            outputs["valid_sum_logp"] += tmp_outputs["valid_sum_logp"]
            if("valid_sum_aux_logp" in tmp_outputs):
                if("valid_sum_aux_logp" in outputs):
                    outputs["valid_sum_aux_logp"] += tmp_outputs["valid_sum_aux_logp"]
                else:
                    outputs["valid_sum_aux_logp"] = tmp_outputs["valid_sum_aux_logp"]
            outputs["loss"] += tmp_outputs["loss"]
            outputs["valid_tokens"] += tmp_outputs["valid_tokens"]

            if(self.step >= self.validation_step):
                self.step = 0
                outputs["require_validation"] = True
                break

        outputs["scheduled_lr"] = tmp_outputs["scheduled_lr"]
        outputs["token_average_ppl"] = outputs["valid_sum_logp"] / outputs["valid_tokens"]
        if("valid_sum_aux_logp" in outputs):
            outputs["auxiliary_token_average_ppl"] = outputs["valid_sum_aux_logp"] / outputs["valid_tokens"]

        outputs = {k: v.tolist()[0] if isinstance(v, np.ndarray) else v
                   for k, v in outputs.items()}

        return outputs

    def eval_step(self, model: ModelInterface, inputs):
        """Run one evaluation step"""
        inputs = dict(zip(model.model.feed_names, inputs))

        model.model.reset_memories(inputs["batch_size"])
        outputs = {"valid_sum_logp": 0.0, "valid_tokens":0.0, "loss":0.0}
        seg_num = inputs["seg_num"]
        sum_lm_mask = 0.0

        for i in range(inputs["seg_num"]):
            #avoiding large memories, do some detach
            seg_len = inputs["segment_lengths"][i]
            seg_input = {k:inputs[k][i, :, :seg_len] for k in self.train_keys}
            tmp_outputs = model.eval_step(seg_input)

            outputs["valid_sum_logp"] += tmp_outputs["valid_sum_logp"]
            if("valid_sum_aux_logp" in tmp_outputs):
                if("valid_sum_aux_logp" in outputs):
                    outputs["valid_sum_aux_logp"] += tmp_outputs["valid_sum_aux_logp"]
                else:
                    outputs["valid_sum_aux_logp"] = tmp_outputs["valid_sum_aux_logp"]
            outputs["valid_tokens"] += tmp_outputs["valid_tokens"]
            outputs["loss"] += tmp_outputs["loss"]

        outputs["batch_size"] = tmp_outputs["batch_size"]
        outputs["tokens_num"] = tmp_outputs["tokens_num"]
        outputs["token_average_ppl"] = outputs["valid_sum_logp"] / outputs["valid_tokens"]
        if("valid_sum_aux_logp" in outputs):
            outputs["auxiliary_token_average_ppl"] = outputs["valid_sum_aux_logp"] / outputs["valid_tokens"]

        outputs = {k: v.tolist()[0] if isinstance(v, np.ndarray) else v
                   for k, v in outputs.items()}
        return outputs

    def infer_step(self, model: ModelInterface, inputs):
        """Run one inference step."""
        raise NotImplementedError("Inference for mem augmented transformer is not yet implemented")
