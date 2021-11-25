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

from knover.core.task import Task
from knover.tasks.dialog_generation import DialogGeneration


@register_task("MemAugGeneration")
class MemAugGeneration(DialogGeneration):
    """Define XL dialogue response generation task."""
    def __init__(self, args):
        super(RecursiveGeneration, self).__init__(args)
        self.detach_interval = args.detach_interval
        self.segment_length = args.segment_length
        self.memories = None
        return

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline arguments."""
        group = parser.add_argument_group("Task")
        group.add_arguement("--segment_length", type=int, default=128,
                help="length of each segments")
        group.add_arguement("--detach_interval", type=int, default=1,
                help="every that much segments detach the memory")
        DialogGeneration.add_cmdline_args(parser)

    def train_step(self, model: ModelInterface, inputs):
        """Run one training step."""

        batch_size = inputs.shape[0]
        all_seq_len = inputs.shape[1]

        #Split Inputs into segments
        seg_num = (all_seq_len - 1) // self.segment_len + 1
        seg_len_list = [self.segment_len] * seg_num
        seg_len_list[-1] -= self.segment_len * seg_num - all_seq_len
        seg_inputs = paddle.split(inputs, seg_len_list, axis=1)

        all_token_num = sum(seg_len_list)
        
        model.reset_memories(batch_size, self.segment_length, self.detach_interval)
        fin_outputs = {"token_lm_loss": 0.0, "loss": 0.0}
        for idx, seg_input in enumerate(seg_inputs):
            #avoiding large memories, do some detach
            outputs = model.train_step(seg_input)
            token_num = seg_len_list[idx]
            fin_outputs["token_lm_loss"] += token_num / all_token_num * outputs["token_lm_loss"]
            fin_outputs["loss"] += token_num / all_token_num * outputs["loss"]

        outputs = {k: v.tolist()[0] if isinstance(v, np.ndarray) else v
                   for k, v in fin_outputs.items()}
        return outputs

    def eval_step(self, model: ModelInterface, inputs):
        """Run one evaluation step"""
        batch_size = inputs.shape[0]
        all_seq_len = inputs.shape[1]

        #Split Inputs into segments
        seg_num = (all_seq_len - 1) // self.segment_len + 1
        seg_len_list = [self.segment_len] * seg_num
        seg_len_list[-1] -= self.segment_len * seg_num - all_seq_len
        seg_inputs = paddle.split(inputs, seg_len_list, axis=1)
        
        all_token_num = sum(seg_len_list)

        model.reset_memories(batch_size, self.segment_length, self.detach_interval)
        fin_outputs = {"token_lm_loss": 0.0, "loss": 0.0, "batch_size": 0, "tokens_num": 0}
        for idx, seg_input in enumerate(seg_inputs):
            #avoiding large memories, do some detach
            outputs = model.eval_step(seg_input)
            token_num = seg_len_list[idx]
            fin_outputs["token_lm_loss"] += token_num / all_token_num * outputs["token_lm_loss"]
            fin_outputs["loss"] += token_num / all_token_num * outputs["loss"]
            fin_outputs["batch_size"] += outputs["batch_size"]
            fin_outputs["tokens_num"] += outputs["tokens_num"]

        outputs = {k: v.tolist()[0] if isinstance(v, np.ndarray) else v
                   for k, v in fin_outputs.items()}
        return outputs

    def infer_step(self, model: ModelInterface, inputs):
        """Run one inference step."""
        batch_size = inputs.shape[0]
        all_seq_len = inputs.shape[1]

        #Split Inputs into segments
        seg_num = (all_seq_len - 1) // self.segment_len + 1
        seg_len_list = [self.segment_len] * seg_num
        seg_len_list[-1] -= self.segment_len * seg_num - all_seq_len
        seg_inputs = paddle.split(inputs, seg_len_list, axis=1)
        
        model.reset_memories(batch_size, self.segment_length, self.detach_interval)
        for idx, seg_input in enumerate(seg_inputs):
            #avoiding large memories, do some detach
            predictions = model.infer_step(seg_input)

        outputs = self._post_process_infer_output(predictions)
        return outputs
