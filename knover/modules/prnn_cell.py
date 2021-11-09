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

import paddle
import math
from paddle.nn import functional as F
from paddle.nn import initializer as I
from paddle.nn.layer.rnn import RNNCellBase

class PlasticRNNCell(RNNCellBase):
    """
    Plastic RNN
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 weight_ih_attr=None,
                 weight_hh_attr=None,
                 bias_ih_attr=None,
                 plastic_A=None,
                 plastic_B=None,
                 plastic_C=None,
                 plastic_D=None,
                 name=None):
        super(PlasticRNNCell, self).__init__()
        std = 1.0 / math.sqrt(hidden_size)
        pl_std = 1.0e-6
        self.weight_ih = self.create_parameter(
                (2 * hidden_size, input_size),
                weight_ih_attr,
                default_initializer=I.Uniform(-std, std))
        self.weight_hh = self.create_parameter(
                (2 * hidden_size, hidden_size),
                weight_hh_attr,
                default_initializer=I.Uniform(-std, std))
        self.bias_ih = self.create_parameter(
                (2 * hidden_size, ),
                bias_ih_attr,
                is_bias=True,
                default_initializer=I.Uniform(-std, std))

        self.plastic_A = self.create_parameter((hidden_size, hidden_size),
                plastic_A,
                default_initializer=I.Uniform(-pl_std, pl_std))
        self.plastic_B = self.create_parameter((hidden_size, hidden_size),
                plastic_B,
                default_initializer=I.Uniform(-pl_std, pl_std))
        self.plastic_C = self.create_parameter((hidden_size, hidden_size),
                plastic_C,
                default_initializer=I.Uniform(-pl_std, pl_std))
        self.plastic_D = self.create_parameter((hidden_size, hidden_size),
                plastic_D,
                default_initializer=I.Uniform(-pl_std, pl_std))

        self.hidden_size = hidden_size
        self.input_size = input_size
        self._unit_vec = paddle.full((1, hidden_size), 1.0)
        self._gate_activation = F.sigmoid
        self._activation = paddle.tanh

    def forward(self, inputs, states=None):
        if states is None:
            states = self.get_initial_states(inputs, self.state_shape)
        w_hh, h_pre = states

        gates = paddle.matmul(inputs, self.weight_ih, transpose_y=True)
        gates = paddle.reshape(gates, [0,1,-1])
        if self.bias_ih is not None:
            gates = gates + self.bias_ih
        gates += paddle.matmul(h_pre, self.weight_hh, transpose_y=True)
        chunked_gates = paddle.split(gates, num_or_sections=2, axis=-1)
        add_plastic = paddle.matmul(h_pre, w_hh, transpose_y=True)

        out = self._activation(chunked_gates[0] + add_plastic)
        s = self._gate_activation(chunked_gates[1])

        unit_vec = paddle.expand_as(self._unit_vec, h_pre)
        d_A = paddle.matmul(h_post, h_pre, transpose_x=True)
        d_B = paddle.matmul(h_post, unit_vec, transpose_x=True)
        d_C = paddle.matmul(unit_vec, h_pre, transpose_x=True)

        w_hh_n = w_hh + s * (d_A * self.plastic_A + d_B * self.plastic_B + d_C * self.plastic_C + paddle.expand_as(self.plastic_D, d_A))

        return paddle.squeeze(out), (w_hh_n, out)

    @property
    def state_shape(self):
        #Hidden State include w_hh and h
        return ((self.hidden_size, self.hidden_size), (1, self.hidden_size))

    def extra_repr(self):
        return '{input_size}, {hidden_size}'.format(**self.__dict__)


class PlasticLSTMCell(RNNCellBase):
    """
    Plastic LSTM
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 weight_ih_attr=None,
                 weight_hh_attr=None,
                 bias_ih_attr=None,
                 bias_hh_attr=None,
                 plastic_A=None,
                 plastic_B=None,
                 plastic_C=None,
                 plastic_D=None,
                 name=None):
        super(PlasticLSTMCell, self).__init__()
        std = 1.0 / math.sqrt(hidden_size)
        pl_std = 1.0e-6

        self.weight_ih = self.create_parameter(
                    (5 * hidden_size, input_size),
                    weight_ih_attr,
                    default_initializer=I.Uniform(-std, std))
        self.weight_hh = self.create_parameter(
                    (5 * hidden_size, hidden_size),
                    weight_hh_attr,
                    default_initializer=I.Uniform(-std, std))
        self.bias_ih = self.create_parameter(
                    (5 * hidden_size, ),
                    bias_ih_attr,
                    is_bias=True,
                    default_initializer=I.Uniform(-std, std))

        self.plastic_A = self.create_parameter((hidden_size, hidden_size),
                plastic_A,
                default_initializer=I.Uniform(-pl_std, pl_std))
        self.plastic_B = self.create_parameter((hidden_size, hidden_size),
                plastic_B,
                default_initializer=I.Uniform(-pl_std, pl_std))
        self.plastic_C = self.create_parameter((hidden_size, hidden_size),
                plastic_C,
                default_initializer=I.Uniform(-pl_std, pl_std))
        self.plastic_D = self.create_parameter((hidden_size, hidden_size),
                plastic_D,
                default_initializer=I.Uniform(-pl_std, pl_std))

        self.hidden_size = hidden_size
        self.input_size = input_size
        self._unit_vec = paddle.full((1, hidden_size), 1.0)
        self._gate_activation = F.sigmoid
        self._activation = paddle.tanh

    def forward(self, inputs, states=None):
        if states is None:
            states = self.get_initial_states(inputs, self.state_shape)
        w_hh, pre_hidden, pre_cell = states
        gates = paddle.matmul(inputs, self.weight_ih, transpose_y=True)
        gates = paddle.reshape(gates, [0,1,-1])
        if self.bias_ih is not None:
            gates = gates + self.bias_ih
        gates += paddle.matmul(pre_hidden, self.weight_hh, transpose_y=True)

        add_plastic = paddle.matmul(pre_hidden, w_hh, transpose_y=True)
        chunked_gates = paddle.split(gates, num_or_sections=5, axis=-1)

        i = self._gate_activation(chunked_gates[0])
        f = self._gate_activation(chunked_gates[1])
        o = self._gate_activation(chunked_gates[3] + add_plastic)
        s = self._gate_activation(chunked_gates[4])
        c = f * pre_cell + i * self._activation(chunked_gates[2])
        h = o * self._activation(c)

        h_post = paddle.reshape(h, [0,1,-1])
        unit_vec = paddle.expand_as(self._unit_vec, pre_hidden)
        d_A = paddle.matmul(h_post, pre_hidden, transpose_x=True)
        d_B = paddle.matmul(h_post, unit_vec, transpose_x=True)
        d_C = paddle.matmul(unit_vec, pre_hidden, transpose_x=True)

        w_hh_n = w_hh + s * (d_A * self.plastic_A + d_B * self.plastic_B 
                + d_C * self.plastic_C + paddle.expand_as(self.plastic_D, d_A))

        return paddle.squeeze(h), (w_hh_n, h, c)

    @property
    def state_shape(self):
        #Hidden State include w_hh and h
        return ((self.hidden_size, self.hidden_size), (1, self.hidden_size), (1, self.hidden_size))

    def extra_repr(self):
        return '{input_size}, {hidden_size}'.format(**self.__dict__)
