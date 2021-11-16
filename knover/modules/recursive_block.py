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
"""Transformer block."""

import math
import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn import initializer as I
from paddle.nn.layer.rnn import RNNCellBase


class RecFormerLayer(RNNCellBase):
    def __init__(self, 
            hidden_size):
        super(RecFormerLayer, self).__init__()
        self.hidden_size = hidden_size
        std = 1.0 / math.sqrt(hidden_size)
        pl_std = 1.0e-6
        self.RNN = nn.RNN(nn.SimpleRNNCell(hidden_size, hidden_size))
        self.fc_w = self.create_parameter((hidden_size, hidden_size),
            default_initializer=I.Uniform(-std, std))
        self.fc_b = self.create_parameter((hidden_size,),
            is_bias=True,
            default_initializer=I.Uniform(-std, std))
        self.modulator_w = self.create_parameter((hidden_size, hidden_size),
            default_initializer=I.Uniform(-std, std))
        self.modulator_b = self.create_parameter((hidden_size,),
            is_bias=True,
            default_initializer=I.Uniform(-std, std))
        self.plastic_A = self.create_parameter((hidden_size, hidden_size),
            default_initializer=I.Uniform(-pl_std, pl_std))
        self.plastic_B = self.create_parameter((hidden_size, hidden_size),
            default_initializer=I.Uniform(-pl_std, pl_std))
        self.plastic_C = self.create_parameter((hidden_size, hidden_size),
            default_initializer=I.Uniform(-pl_std, pl_std))
        self.plastic_D = self.create_parameter((hidden_size, hidden_size),
            default_initializer=I.Uniform(-pl_std, pl_std))

        self._unit_vec = paddle.full((1, self.hidden_size), 1.0)

        self._gate_activation = F.sigmoid
        self._activation = F.relu

    def plasticity_rule(self, input, output, m):
        """
        calculate plasticity update
        """
        pi = paddle.reshape(input, [0, 0, 1, -1])
        po = paddle.reshape(output, [0, 0, 1, -1])
        pi.stop_gradient = True
        po.stop_gradient = True
        d_A = paddle.matmul(po, pi, transpose_x=True)
        d_B = paddle.matmul(po, self._unit_vec, transpose_x=True)
        d_C = paddle.matmul(self._unit_vec, pi, transpose_x=True)
        dw = paddle.reshape(m, [0, 0, 1, -1]) * (
                d_A * self.plastic_A + 
                d_B * self.plastic_B + 
                d_C * self.plastic_C + 
                paddle.expand_as(self.plastic_D, d_A))
        return paddle.sum(dw, axis=[1])

    def forward(self, input_bulk, memory):
        h, pl_w = memory
        rnn_out, rnn_h = self.RNN(input_bulk, h)
        output = self._activation(paddle.matmul(rnn_out, pl_w + self.fc_w, transpose_y=True) + self.fc_b)
        mod = self._gate_activation(paddle.matmul(output, self.modulator_w, transpose_y=True) + self.modulator_b)

        n_pl_w = pl_w + self.plasticity_rule(rnn_out, output, mod)
        return output, (rnn_h, n_pl_w)

if __name__=="__main__":
    B = 16
    S = 8
    H = 32

    layer = RecFormerLayer(H)
    print(layer.forward(paddle.randn((B, S, H)), (paddle.randn((B, H)), paddle.randn((B, H, H)))))
