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
"""Long Text Readers."""

from collections import namedtuple
import os

import numpy as np 
import paddle
import paddle.fluid as fluid
from paddle.fluid.incubate.fleet.collective import fleet
from knover.data.dialog_reader import DialogReader


class LongTextReader(DialogReader):
    """The implement of LongTextReader."""

    def __init__(self, args):
        super(LongTextReader, self).__init__(args)
        self.segment_length = args.segment_length
        return

    @classmethod
    def add_cmdline_args(cls, parser):
        """Add cmdline arguments."""
        group = parser.add_argument_group("Task")
        group.add_argument("--segment_length", type=int, default=128,
                help="length of each segments")
        DialogReader.add_cmdline_args(parser)

    def _pad_batch_records(self, batch_records, is_infer, phase=None):
        """Padding a batch of records and construct model's inputs.

        This function can be override by its subclass if necessary.
        """
        assert self.is_autoregressive, "Currently, LongText Reader Only supports autoregressive mode"
        batch = super(LongTextReader, self)._pad_batch_records(batch_records, is_infer, phase)
        batch_size = len(batch_records)
        all_seq_len = batch["token_ids"].shape[1] - 1

        seg_num = (all_seq_len - 1) // self.segment_length + 1
        seg_len_list = [self.segment_length] * seg_num
        seg_len_list[-1] -= self.segment_length * seg_num - all_seq_len
        res_seg = self.segment_length - seg_len_list[-1]


        batch_token_ids = paddle.concat([paddle.to_tensor(batch["token_ids"][:, :-1]), paddle.full((batch_size, res_seg), self.pad_id, dtype='int64')], axis=1)
        batch_type_ids = paddle.concat([paddle.to_tensor(batch["type_ids"][:, :-1]), paddle.full((batch_size, res_seg), 0, dtype='int64')], axis=1)
        batch["token_ids"] = paddle.split(batch_token_ids, seg_num, axis=1)
        batch["type_ids"] = paddle.split(batch_type_ids, seg_num, axis=1)

        if self.use_role:
            batch_role_ids = paddle.concat([paddle.to_tensor(batch["role_ids"][:, :-1]), paddle.full((batch_size, res_seg), 0, dtype='int64')], axis=1)
            batch["role_ids"] = paddle.split(batch_role_ids, seg_num, axis=1)
        if not is_infer:
            batch_tgt_label = paddle.concat([paddle.to_tensor(batch["tgt_label"]), paddle.full((batch_size, res_seg), self.pad_id, dtype='int64')], axis=1)
            batch_loss_mask = paddle.concat([paddle.to_tensor(batch["loss_mask"]), paddle.full((batch_size, res_seg), 0, dtype='float32')], axis=1)
            batch["tgt_label"] = paddle.split(batch_tgt_label, seg_num, axis=1)
            batch["loss_mask"] = paddle.split(batch_loss_mask, seg_num, axis=1)

        # add pos id
        batch["batch_size"] = batch_size
        batch["seg_num"] = seg_num
        batch["seg_len"] = self.segment_length
        batch["segment_lengths"] = seg_len_list

        return batch
