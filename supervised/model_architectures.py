import torch
from torch import nn
import numpy as np
import math


def TGLSTM(input_size, hidden_size, num_layers, bias=True,
           batch_first=False, dropout=False, bidirectional=False):
    '''Returns a ScriptModule that mimics a PyTorch native LSTM.'''

    # The following are not implemented.
    assert bias
    assert not batch_first
    assert not dropout

    if bidirectional:
        stack_type = StackedLSTM2
        layer_type = BidirLSTMLayer
        dirs = 2
    else:
        stack_type = StackedLSTM
        layer_type = LSTMLayer
        dirs = 1

    return stack_type(num_layers, layer_type,
                      first_layer_args=[LSTMCell, input_size, hidden_size],
                      other_layer_args=[LSTMCell, hidden_size * dirs,
                                        hidden_size])


def reverse(lst):
    # type: (List[Tensor]) -> List[Tensor]
    return lst[::-1]


class LSTMCell(torch.nn.Module):
    def __init__(self, input_features, state_size):
        super(LSTMCell, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        # 4 * state_size for input gate, output gate, forget gate and candidate cell gate.
        # input_features + state_size because we will multiply with [input, h].
        self.weights = torch.nn.Parameter(
            torch.Tensor(4 * state_size, input_features + state_size))
        self.bias = torch.nn.Parameter(torch.Tensor(1, 4 * state_size))
        self.weight_t = torch.nn.Parameter(torch.Tensor(3 * state_size, 1))
        self.bias_t = torch.nn.Parameter(torch.Tensor(1, 3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, time, state):
        if state is None:
            old_h = input.new_zeros(input.size(0), self.state_size, requires_grad=False)
            old_cell = input.new_zeros(input.size(0), self.state_size, requires_grad=False)
        else:
            old_h, old_cell = state

        X = torch.cat([old_h, input], dim=1)
        # Compute the input, output and candidate cell gates with one MM.
        gate_weights = F.linear(X, self.weights, self.bias)
        time_weights = F.linear(time, self.weight_t, self.bias_t)
        # Split the combined gate weight matrix into its components.
        ingate, forgetgate, cellgate, outgate = gate_weights.chunk(4, dim=1)
        ingate_t, forgetgate_t, outgate_t = time_weights.chunk(3, dim=1)

        input_gate = torch.sigmoid(ingate)
        output_gate = torch.sigmoid(outgate)
        forget_gate = torch.sigmoid(forgetgate)
        candidate_cell = torch.tanh(cellgate)

        input_gate_t = torch.sigmoid(ingate_t)
        output_gate_t = torch.sigmoid(outgate_t)
        forget_gate_t = torch.sigmoid(forgetgate_t)

        # ******************* TEST PURPOSE ONLY
        # set time gates to ones to get an equivalent classic LSTM
        #        input_gate_t = input_gate_t.new_ones(input_gate_t.size(0), input_gate_t.size(1))
        #        forget_gate_t = forget_gate_t.new_ones(forget_gate_t.size(0), forget_gate_t.size(1))
        #        output_gate_t = output_gate_t.new_ones(output_gate_t.size(0), output_gate_t.size(1))
        # *******************

        # Compute the new cell state.
        new_cell = old_cell * forget_gate * forget_gate_t + candidate_cell * input_gate * input_gate_t
        # Compute the new hidden state and output.
        new_h = torch.tanh(new_cell) * output_gate * output_gate_t

        return new_h, (new_h, new_cell)


class LSTMLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(self, input, time, state=None):
        inputs = input.unbind(0)
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], time[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class ReverseLSTMLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(ReverseLSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(self, input, time, state=None):
        inputs = reverse(input.unbind(0))
        times = reverse(time.unbind(0))
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], times[i], state)
            outputs += [out]
        return torch.stack(reverse(outputs)), state


class BidirLSTMLayer(nn.Module):
    __constants__ = ['directions']

    def __init__(self, cell, *cell_args):
        super(BidirLSTMLayer, self).__init__()
        self.directions = nn.ModuleList([
            LSTMLayer(cell, *cell_args),
            ReverseLSTMLayer(cell, *cell_args),
        ])

    def forward(self, input, time, states=None):
        outputs = []
        output_states = []
        for i, direction in enumerate(self.directions):
            if states is None:
                state = None
            else:
                state = states[i]
            out, out_state = direction(input, time, state)
            outputs += [out]
            output_states += [out_state]
        return torch.cat(outputs, -1), output_states


def init_stacked_lstm(num_layers, layer, first_layer_args, other_layer_args):
    layers = [layer(*first_layer_args)] + [layer(*other_layer_args)
                                           for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)


class StackedLSTM(nn.Module):
    __constants__ = ['layers']  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedLSTM, self).__init__()
        self.layers = init_stacked_lstm(num_layers, layer, first_layer_args,
                                        other_layer_args)

    def forward(self, input, time, states=None):
        output_states = []
        output = input
        i = 0
        for rnn_layer in self.layers:
            if states is None:
                state = None
            else:
                state = states[i]
            output, out_state = rnn_layer(output, time, state)
            output_states += [out_state]
            i += 1
        return output, output_states


# Differs from StackedLSTM in that its forward method takes
# List[List[Tuple[Tensor,Tensor]]]. It would be nice to subclass StackedLSTM
# except we don't support overriding script methods.
# https://github.com/pytorch/pytorch/issues/10733
class StackedLSTM2(nn.Module):
    __constants__ = ['layers']  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedLSTM2, self).__init__()
        self.layers = init_stacked_lstm(num_layers, layer, first_layer_args,
                                        other_layer_args)

    def forward(self, input, time, states=None):
        # type: (Tensor, List[List[Tuple[Tensor, Tensor]]]) -> Tuple[Tensor, List[List[Tuple[Tensor, Tensor]]]]
        # List[List[LSTMState]]: The outer list is for layers,
        #                        inner list is for directions.
        output_states = []
        output = input
        i = 0
        for rnn_layer in self.layers:
            if states is None:
                state = None
            else:
                state = states[i]
            output, out_state = rnn_layer(output, time, state)
            output_states += [out_state]
            i += 1
        return output, output_states


def flatten_states(states):
    states = list(zip(*states))
    assert len(states) == 2
    return [torch.stack(state) for state in states]


def double_flatten_states(states):
    states = flatten_states([flatten_states(inner) for inner in states])
    return [hidden.view([-1] + list(hidden.shape[2:])) for hidden in states]


# ----------------------
# ------ Models --------
# ----------------------


class Model_TimeGatedLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, extra_to_model):
        super(Model_TimeGatedLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.extra_to_model = extra_to_model

        tglstm_input_size = np.prod(input_size)

        self.tglstm = TGLSTM(tglstm_input_size, hidden_dim, n_layers, bias=True, batch_first=False, dropout=False, bidirectional=False)

        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_size)

    def forward(self, X, hidden):
        # X.shape: [batch_size, seq_len, features]
        # time.shape: [batch_size, seq_len, features]
        # swap axis to get batch_first=False
        x, time = X, X  # TODO cut the input into X (features) and time.
        x = x.permute(1, 0, 2)
        time = time.permute(1, 0, 2)
        output_rnn, hidden = self.tglstm(x, time, hidden)
        fc_output = self.fc(output_rnn.permute(1, 0, 2))
        # fc_output will be batch_size*seq_len*num_classes
        return fc_output, hidden

    def init_hidden(self, batch_size, device):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        if self.extra_to_model['hidden'] == 'zeros':
            h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)  # hidden state
            c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)  # internal state
            hidden = (h_0, c_0)
        elif self.extra_to_model['hidden'] == 'randn':
            h_0 = torch.randn(self.n_layers, batch_size, self.hidden_dim).to(device)
            c_0 = torch.randn(self.n_layers, batch_size, self.hidden_dim).to(device)
            hidden = (h_0, c_0)

        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden


class Model_RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, extra_to_model):
        super(Model_RNN, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.extra_to_model = extra_to_model

        rnn_input_size = int(np.prod(input_size))

        dropout_prob = 0
        if self.extra_to_model.get('dropout') != 0:
            self.dropout = nn.Dropout(self.extra_to_model['dropout'])
            dropout_prob = self.extra_to_model['dropout']

        # Adding encoder and decoder convolutional networks
        if self.extra_to_model.get('encoder_channels') and self.extra_to_model.get('decoder_channels'):
            self.encoder = nn.Sequential(
                nn.Conv2d(1, self.extra_to_model['encoder_channels'], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.extra_to_model['encoder_channels'], self.extra_to_model['encoder_channels'],
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.extra_to_model['encoder_channels'], self.extra_to_model['decoder_channels'],
                          kernel_size=3, padding=1),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Conv2d(self.extra_to_model['decoder_channels'], self.extra_to_model['decoder_channels'],
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.extra_to_model['decoder_channels'], self.extra_to_model['decoder_channels'],
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.extra_to_model['decoder_channels'], 1, kernel_size=3, padding=1),
                nn.ReLU()
            )

        # Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(rnn_input_size, hidden_dim, n_layers, batch_first=True, dropout=dropout_prob)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, state, hidden):

        # Applying the encoder and decoder layers
        if self.extra_to_model.get('encoder_channels') and self.extra_to_model.get('decoder_channels'):
            state1 = state.unsqueeze(1)  # adding a channel dimension for the convolutional layers
            state1 = self.encoder(state1)
            state1 = self.decoder(state1)
            state1 = state1.squeeze(1)  # removing the channel dimension
        else:
            state1 = state


        # Passing in the input and hidden state into the model and obtaining outputs
        state2, hidden_nest_state = self.rnn(state1, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        state3 = state2.contiguous().view(-1, self.hidden_dim)

        if self.extra_to_model.get('dropout') != 0:
            out = self.dropout(state3)
        else:
            out = state3

        out2 = self.fc(out)

        return out2, hidden_nest_state

    def init_hidden(self, batch_size, device):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        if self.extra_to_model.get('hidden') == 'rand_gauss':
            hidden = torch.randn(self.n_layers, batch_size, self.hidden_dim).to(device)
            hidden = torch.normal(mean=torch.zeros_like(hidden), std=torch.ones_like(hidden)*0.5).to(device)
        elif self.extra_to_model.get('hidden') == 'rand_unif':
            hidden = torch.Tensor(self.n_layers, batch_size, self.hidden_dim).uniform_(-0.1, 0.1).to(device)
        else:  # self.extra_to_model('hidden') == 'zeros'      default
            hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)

        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden


class Model_GRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, extra_to_model):
        super(Model_GRU, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.extra_to_model = extra_to_model

        gru_input_size = int(np.prod(input_size))

        dropout_prob = 0
        if self.extra_to_model.get('dropout') != 0:
            self.dropout = nn.Dropout(self.extra_to_model['dropout'])
            dropout_prob = self.extra_to_model['dropout']

        # Adding encoder and decoder convolutional networks
        if self.extra_to_model.get('encoder_channels') and self.extra_to_model.get('decoder_channels'):
            self.encoder = nn.Sequential(
                nn.Conv2d(1, self.extra_to_model['encoder_channels'], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.extra_to_model['encoder_channels'], self.extra_to_model['encoder_channels'],
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.extra_to_model['encoder_channels'], self.extra_to_model['decoder_channels'],
                          kernel_size=3, padding=1),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Conv2d(self.extra_to_model['decoder_channels'], self.extra_to_model['decoder_channels'],
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.extra_to_model['decoder_channels'], self.extra_to_model['decoder_channels'],
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.extra_to_model['decoder_channels'], 1, kernel_size=3, padding=1),
                nn.ReLU()
            )

        # Defining the layers
        # GRU Layer
        self.gru = nn.GRU(gru_input_size, hidden_dim, n_layers, batch_first=True, dropout=dropout_prob)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, state, hidden):

        # Applying the encoder and decoder layers
        if self.extra_to_model.get('encoder_channels') and self.extra_to_model.get('decoder_channels'):
            state1 = state.unsqueeze(1)  # adding a channel dimension for the convolutional layers
            state1 = self.encoder(state1)
            state1 = self.decoder(state1)
            state1 = state1.squeeze(1)  # removing the channel dimension
        else:
            state1 = state

        # Passing in the input and hidden state into the model and obtaining outputs
        state2, hidden_next_state = self.gru(state1, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        state3 = state2.contiguous().view(-1, self.hidden_dim)

        if self.extra_to_model.get('dropout') != 0:
            out = self.dropout(state3)
        else:
            out = state3

        out2 = self.fc(out)

        return out2, hidden_next_state

    def init_hidden(self, batch_size, device):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        if self.extra_to_model.get('hidden') == 'rand_gauss':
            hidden = torch.randn(self.n_layers, batch_size, self.hidden_dim).to(device)
            hidden = torch.normal(mean=torch.zeros_like(hidden), std=torch.ones_like(hidden)*0.5).to(device)
        elif self.extra_to_model.get('hidden') == 'rand_unif':
            hidden = torch.Tensor(self.n_layers, batch_size, self.hidden_dim).uniform_(-0.1, 0.1).to(device)
        else:  # self.extra_to_model('hidden') == 'zeros'      default
            hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)

        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden


class Model_LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, extra_to_model):
        super(Model_LSTM, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.extra_to_model = extra_to_model

        lstm_input_size = int(np.prod(input_size))

        dropout_prob = 0
        if self.extra_to_model.get('dropout') != 0:
            self.dropout = nn.Dropout(self.extra_to_model['dropout'])
            dropout_prob = self.extra_to_model['dropout']

        # Adding encoder and decoder convolutional networks
        if self.extra_to_model.get('encoder_channels') and self.extra_to_model.get('decoder_channels'):
            self.encoder = nn.Sequential(
                nn.Conv2d(1, self.extra_to_model['encoder_channels'], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.extra_to_model['encoder_channels'], self.extra_to_model['encoder_channels'],
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.extra_to_model['encoder_channels'], self.extra_to_model['decoder_channels'],
                          kernel_size=3, padding=1),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Conv2d(self.extra_to_model['decoder_channels'], self.extra_to_model['decoder_channels'],
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.extra_to_model['decoder_channels'], self.extra_to_model['decoder_channels'],
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.extra_to_model['decoder_channels'], 1, kernel_size=3, padding=1),
                nn.ReLU()
            )

        # Defining the layers
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout_prob)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, state, hidden):

        # Applying the encoder and decoder layers
        if self.extra_to_model.get('encoder_channels') and self.extra_to_model.get('decoder_channels'):
            state1 = state.unsqueeze(1)  # adding a channel dimension for the convolutional layers
            state1 = self.encoder(state1)
            state1 = self.decoder(state1)
            state1 = state1.squeeze(1)  # removing the channel dimension
        else:
            state1 = state

        # Passing in the input and hidden state into the model and obtaining outputs
        # print(state1.shape)
        state2, hidden_nest_state = self.lstm(state1, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        state3 = state2.contiguous().view(-1, self.hidden_dim)

        if self.extra_to_model.get('dropout') != 0:
            out = self.dropout(state3)
        else:
            out = state3

        out2 = self.fc(out)

        return out2, hidden_nest_state

    def init_hidden(self, batch_size, device):

        # This method generates the first hidden state of zeros which we'll use in the forward pass
        if self.extra_to_model.get('hidden') == 'rand_gauss':
            hidden = torch.randn(self.n_layers, batch_size, self.hidden_dim).to(device)
            h_0 = torch.normal(mean=torch.zeros_like(hidden), std=torch.ones_like(hidden)*0.5).to(device)
            c_0 = torch.normal(mean=torch.zeros_like(hidden), std=torch.ones_like(hidden)*0.5).to(device)
            hidden = (h_0, c_0)
        elif self.extra_to_model.get('hidden') == 'rand_unif':
            h_0 = torch.Tensor(self.n_layers, batch_size, self.hidden_dim).uniform_(-0.1, 0.1).to(device)
            c_0 = torch.Tensor(self.n_layers, batch_size, self.hidden_dim).uniform_(-0.1, 0.1).to(device)
            hidden = (h_0, c_0)
        else:  # self.extra_to_model('hidden') == 'zeros'      default
            h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
            c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
            hidden = (h_0, c_0)

        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden


def get_model(parameters):
    features = parameters['features']
    state_sample_length = parameters['state_sample_length']

    output_size = parameters['output_size']
    input_size = (features * state_sample_length)

    # Instantiate the model with hyperparameters
    hidden_dim = parameters['hidden_dim']
    n_layers = parameters['n_layers']

    architecture = parameters['architecture']
    extra_to_model = parameters['extra_to_model']

    chosen_architecture = 'RNN'
    for key, value in architecture.items():
        if value == 1:
            chosen_architecture = key
            break

    if chosen_architecture == 'RNN':
        model = Model_RNN(input_size=input_size, output_size=output_size, hidden_dim=hidden_dim, n_layers=n_layers, extra_to_model=extra_to_model)
        print("The architecture is RNN.")
    elif chosen_architecture == 'GRU':
        model = Model_GRU(input_size=input_size, output_size=output_size, hidden_dim=hidden_dim, n_layers=n_layers, extra_to_model=extra_to_model)
        print("The architecture is GRU.")
    elif chosen_architecture == 'LSTM':
        model = Model_LSTM(input_size=input_size, output_size=output_size, hidden_dim=hidden_dim, n_layers=n_layers, extra_to_model=extra_to_model)
        print("The architecture is LSTM.")
    elif chosen_architecture == 'TGLSTM':
        model = Model_TimeGatedLSTM(input_size=input_size, output_size=output_size, hidden_dim=hidden_dim, n_layers=n_layers, extra_to_model=extra_to_model)
        print("The architecture is TGLSTM.")
    else:
        model = Model_RNN(input_size=input_size, output_size=output_size, hidden_dim=hidden_dim, n_layers=n_layers, extra_to_model=extra_to_model)
        print("The architecture is RNN.")

    return model
