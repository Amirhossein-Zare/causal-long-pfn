import torch
from torch import nn


class VariationalLSTM(nn.Module):
    """
    Variational LSTM layer in PyTorch.
    """

    def __init__(self, input_size, hidden_size, num_layer=1, dropout_rate=0.0):
        super().__init__()

        self.lstm_layers = [nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)]
        if num_layer > 1:
            self.lstm_layers += [
                nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
                for _ in range(num_layer - 1)
            ]

        self.lstm_layers = nn.ModuleList(self.lstm_layers)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

    def forward(self, x, init_states=None):
        for lstm_cell in self.lstm_layers:
            if init_states is None:
                hx = torch.zeros((x.shape[0], self.hidden_size)).type_as(x)
                cx = torch.zeros((x.shape[0], self.hidden_size)).type_as(x)
            else:
                hx, cx = init_states, init_states

            keep_prob = 1.0 - float(self.dropout_rate)
            if keep_prob <= 0:
                raise ValueError("dropout_rate must be < 1.0")

            out_dropout = torch.bernoulli(
                hx.data.new(hx.data.size()).fill_(keep_prob)
            ) / keep_prob
            h_dropout = torch.bernoulli(
                hx.data.new(hx.data.size()).fill_(keep_prob)
            ) / keep_prob
            c_dropout = torch.bernoulli(
                cx.data.new(cx.data.size()).fill_(keep_prob)
            ) / keep_prob

            output = []
            for t in range(x.shape[1]):
                hx, cx = lstm_cell(x[:, t, :], (hx, cx))

                if lstm_cell.training:
                    out = hx * out_dropout
                    hx, cx = hx * h_dropout, cx * c_dropout
                else:
                    out = hx

                output.append(out)

            x = torch.stack(output, dim=1)

        return x

__all__ = ["VariationalLSTM"]
