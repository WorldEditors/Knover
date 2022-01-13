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
        self.validation_words = args.validation_words
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
        group.add_argument("--validation_words", type=int, default=-1,
                help="used for calculate ppl_per_word")
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
            outputs["loss"] += tmp_outputs["loss"] * tmp_outputs["valid_tokens"]
            outputs["valid_tokens"] += tmp_outputs["valid_tokens"]
            if("auxiliary_loss" in tmp_outputs):
                if("auxiliary_loss" in outputs):
                    outputs["auxiliary_loss"] +=  tmp_outputs["auxiliary_loss"] * tmp_outputs["valid_tokens"]
                else:
                    outputs["auxiliary_loss"] = tmp_outputs["auxiliary_loss"] * tmp_outputs["valid_tokens"]

            if(self.step >= self.validation_step):
                self.step = 0
                outputs["require_validation"] = True
                break

        outputs["scheduled_lr"] = tmp_outputs["scheduled_lr"]
        outputs["loss"] = outputs["loss"] / outputs["valid_tokens"]
        if("auxiliary_loss" in outputs):
                outputs["auxiliary_loss"] = outputs["auxiliary_loss"] / outputs["valid_tokens"]
        outputs["tokens_ppl"] = math.exp(outputs["valid_sum_logp"] / outputs["valid_tokens"])

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
            outputs["loss"] += tmp_outputs["loss"] * tmp_outputs["valid_tokens"]
            if("auxiliary_loss" in tmp_outputs):
                if("auxiliary_loss" in outputs):
                    outputs["auxiliary_loss"] +=  tmp_outputs["auxiliary_loss"] * tmp_outputs["valid_tokens"]
                else:
                    outputs["auxiliary_loss"] = tmp_outputs["auxiliary_loss"] * tmp_outputs["valid_tokens"]
            outputs["valid_tokens"] += tmp_outputs["valid_tokens"]

        outputs["batch_size"] = tmp_outputs["batch_size"]
        outputs["tokens_num"] = tmp_outputs["tokens_num"]
        outputs["loss"] = outputs["loss"] / outputs["valid_tokens"]
        if("token_auxiliary_loss" in outputs):
                outputs["auxiliary_loss"] = outputs["auxiliary_loss"] / outputs["valid_tokens"]
        outputs["token_token_ppl"] = math.exp(outputs["valid_sum_logp"] / outputs["valid_tokens"])

        outputs = {k: v.tolist()[0] if isinstance(v, np.ndarray) else v
                   for k, v in outputs.items()}
        return outputs

    def infer_step(self, model: ModelInterface, inputs):
        """Run one inference step."""
        raise NotImplementedError("Inference for mem augmented transformer is not yet implemented")

    def get_metrics(self, outputs):
        """Get metrics."""
        if outputs is None:
            raise ValueError("metrics is None")
        outputs = dict(outputs)
        metrics = {}
        batch_size = outputs.pop("batch_size", None)
        tokens_num = outputs.pop("tokens_num", None)
        for k in outputs:
            if k.startswith("token_"):
                metrics[k[6:]] = outputs[k]
            else:
                metrics[k] = outputs[k]
        return metrics

    def merge_metrics_and_statistics(self, outputs, part_outputs):
        """Merge two evaulation output.

        Args:
            outputs: Original outputs which contains metrics and statistics.
            part_outputs: New outputs which contains metrics and statistics.

        Returns:
            Return merged output which contains metrics and statistics.
        """
        if outputs is None:
            return part_outputs

        if part_outputs is None:
            return outputs

        batch_size = outputs.pop("batch_size")
        tokens_num = outputs.pop("tokens_num")
        valid_tokens = outputs.pop("valid_tokens")
        valid_sum_logp = outputs.pop("valid_sum_logp")
        part_batch_size = part_outputs.pop("batch_size")
        part_tokens_num = part_outputs.pop("tokens_num")
        part_valid_tokens = part_outputs.pop("valid_tokens")
        part_valid_sum_logp = part_outputs.pop("valid_sum_logp")

        new_outputs = {
            "batch_size": batch_size + part_batch_size,
            "tokens_num": tokens_num + part_tokens_num,
            "valid_tokens": valid_tokens + part_valid_tokens,
            "valid_sum_logp": valid_sum_logp + part_valid_sum_logp,
        }

        for k in outputs:
            if k.startswith("token_"):
                new_outputs[k] = (
                    outputs[k] * tokens_num + part_outputs[k] * part_tokens_num
                ) / new_outputs["tokens_num"]
            elif(k != "word_normalized_ppl"):
                new_outputs[k] = (
                    outputs[k] * batch_size + part_outputs[k] * part_batch_size
                ) / new_outputs["batch_size"]

        if(self.validation_words > 0):
            new_outputs["word_normalized_ppl"] = math.exp(new_outputs["valid_sum_logp"] / self.validation_words)

        return new_outputs
