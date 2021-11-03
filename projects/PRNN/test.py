import paddle
from knover.module.prnn_cell import PlasticRNNCell

import paddle

batch = 16
hidden = 32
input = 16
time_seq = 5
x = paddle.randn((batch, time_seq, input))
prev_w_hh = paddle.randn((batch, hidden, hidden))
prev_h = paddle.randn((batch, 1, hidden))

rnn_cell = PlasticRNNCell(input, hidden)
states = (prev_w_hh, prev_h)
rnn_encoder = paddle.nn.RNN(rnn_cell)
outputs, final_states = rnn_encoder(x, states)
print(outputs, final_states)
