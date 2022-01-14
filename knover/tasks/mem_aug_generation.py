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

    def merge_outputs_inbatch(self, outputs, part_outputs):
        new_outputs = dict(outputs)
        for key in part_outputs:
            if((key.startswith("sum_tokens_") or key == "tokens_num") and key in outputs):
                new_outputs[key] += part_outputs[key]
            else:
                new_outputs[key] = part_outputs[key]
        return new_outputs

    def post_process(self, outputs, info):
        """
        outputs: dicts of statistics
        additional_info: dicts of other cnt number
        """
        new_outputs = {k: v.tolist()[0] if isinstance(v, np.ndarray) else v
                   for k, v in outputs.items()}
        for key in info:
            if(key.startswith("sta_")):
                new_outputs[key] = info[key]
        return new_outputs

    def train_step(self, model: ModelInterface, inputs):
        """Run one training step."""
        inputs = dict(zip(model.model.feed_names, inputs))
        
        model.model.reset_memories(inputs["batch_size"])
        outputs = dict()
        seg_num = inputs["seg_num"]

        for i in range(inputs["seg_num"]):
            #avoiding large memories, do some detach
            seg_len = inputs["segment_lengths"][i]
            seg_input = {k:inputs[k][i, :, :seg_len] for k in self.train_keys}
            tmp_outputs = model.train_step(seg_input)
            self.step += 1

            outputs = self.merge_outputs_inbatch(outputs, tmp_outputs)

            if(self.step >= self.validation_step):
                self.step = 0
                outputs["require_validation"] = True
                break

        outputs = self.post_process(outputs, inputs)
        return outputs

    def eval_step(self, model: ModelInterface, inputs):
        """Run one evaluation step"""
        inputs = dict(zip(model.model.feed_names, inputs))

        model.model.reset_memories(inputs["batch_size"])
        outputs = dict()
        seg_num = inputs["seg_num"]

        for i in range(inputs["seg_num"]):
            #avoiding large memories, do some detach
            seg_len = inputs["segment_lengths"][i]
            seg_input = {k:inputs[k][i, :, :seg_len] for k in self.train_keys}
            tmp_outputs = model.eval_step(seg_input)
            outputs = self.merge_outputs_inbatch(outputs, tmp_outputs)

        outputs = self.post_process(outputs, inputs)
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
        sta_keys = {}
        for key in outputs:
            if(key.startswith("sta_")):
                sta_keys[key] = outputs[key]

        for key in outputs:
            if(key.startswith("sum_tokens_")):
                metrics["avg_tokens_" + key[11:]] = outputs[key] / outputs["tokens_num"]
                for sta_key in sta_keys:
                    metrics["avg_" + sta_key[4:] + key[10:]] = outputs[key] / outputs[sta_key]
            else:
                metrics[key] = outputs[key]
        avg_metrics = dict()
        for key in metrics:
            if(key.endswith("_logp") and key.startswith("avg_")):
                avg_metrics[key.replace("_logp", "_ppl")] = math.exp(metrics[key])
        metrics.update(avg_metrics)
        metrics = {key: float(metrics[key].numpy()) if paddle.is_tensor(metrics[key]) else float(metrics[key]) for key in metrics}

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

        new_outputs = dict(outputs)
        for key in part_outputs:
            if((key.startswith("sum_tokens_") or key == "tokens_num" or key.startswith("sta_") or key=="batch_size") and key in outputs):
                new_outputs[key] += part_outputs[key]
            else:
                new_outputs[key] = part_outputs[key]
        new_outputs = {key: float(new_outputs[key].numpy()) if paddle.is_tensor(new_outputs[key]) else float(new_outputs[key]) for key in new_outputs}
        return new_outputs
