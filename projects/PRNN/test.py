import paddle
from knover.modules.prnn_cell import PlasticRNNCell, PlasticLSTMCell, ModulateRNNCell

batch = 8
input = 16
hidden = 32
output = 64
time_seq = 5
x = paddle.randn((batch, time_seq, input))

#rnn_cell = PlasticLSTMCell(input, hidden)
#rnn_cell = PlasticRNNCell(input, hidden)
rnn_cell = ModulateRNNCell(input, hidden, output)
states = rnn_cell.get_initial_states(x, rnn_cell.state_shape)
rnn_encoder = paddle.nn.RNN(rnn_cell)
outputs, final_states = rnn_encoder(x, states)
print(outputs, final_states)
