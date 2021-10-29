import paddle
from plastic_rnn import PlasticRNNCell


import paddle

batch = 16
hidden = 32
input = 16
time_seq = 5
x = paddle.randn((batch, time_seq, input))
prev_w_hh = paddle.randn((batch, hidden, hidden))
prev_h = paddle.randn((batch, 1, hidden))

cell = PlasticRNNCell(input, hidden)
states = (prev_w_hh, prev_h)
for i in range(time_seq):
    h, states = cell(x[:, i, :], states)
    print(h.shape, states[0].shape, states[1].shape)
