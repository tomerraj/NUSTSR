from torch import nn
import torch.nn.functional as F
import os
import sys

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from supervised.make_data_supervized import *
from Data_Base.Best_Slots import *


def get_device(to_print=True):
    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()
    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        cuda_num = np.random.choice(range(4), p=[0.19, 0.27, 0.27, 0.27])
        # cuda_num = 0
        device = torch.device(f"cuda:{str(cuda_num)}")
        if to_print:
            print(f"GPU:{str(cuda_num)} is available!\n")
        torch.cuda.init()

    else:
        device = torch.device("cpu")
        if to_print:
            print("GPU not available, CPU used")

    # return torch.device("cpu")
    return device


def randomize_batch(matrix,device):
    """
    Randomly swaps the rows of a matrix along the first axis (batch dimension).

    Parameters:
    - matrix: Input matrix of shape (batch_size, ...)

    Returns:
    - randomized_matrix: Matrix with the rows along the first axis randomly swapped.
    """
    batch_size = matrix.size(0)
    permutation = torch.randperm(batch_size, device=device)
    randomized_matrix = torch.index_select(matrix, 0, permutation)
    return randomized_matrix

def create_loss_fn(config, number_of_slots):
    criterion_array = []
    if config['CrossEntropyLoss']:
        criterion_array.append(nn.CrossEntropyLoss())
    if config['L1Loss']:
        criterion_array.append(nn.L1Loss())
    if config['MSELoss']:
        criterion_array.append(nn.MSELoss())

    def loss_fn(inputs, targets):
        loss = 0.0

        for criterion in criterion_array:
            factor = config['CrossEntropyLoss']
            # nn.CrossEntropyLoss() doesn't need check the values are correct
            if isinstance(criterion, nn.L1Loss):
                inputs = F.softmax(inputs, dim=1)
                targets = F.one_hot(targets, num_classes=number_of_slots)
                factor = config['L1Loss']

            if isinstance(criterion, nn.MSELoss):
                inputs = F.softmax(inputs, dim=1).float()
                targets = F.one_hot(targets, num_classes=number_of_slots).float()
                factor = config['MSELoss']

            if config.get('label_smoothing_loss') != None:
                epsilon = config.get('label_smoothing_loss')
                n_classes = inputs.size(1)
                targets = (1 - epsilon) * targets + epsilon / n_classes

            loss += criterion(inputs, targets)* factor
        return loss.to(torch.float64)
        # return loss.double()

    return loss_fn


def train(parameters, train_mat, model, update_loss_n=2, start_epoch=0, device=get_device(to_print=False)):
    if start_epoch == 0:
        print(f'Start train of {parameters["MODEL_NAME"]}')

    simulation_time = parameters['simulation_time']
    fps = parameters['fps']
    number_of_slots = parameters['number_of_slots']
    features = parameters['features']
    feature_dict = parameters['feature_dict']
    state_sample_length = parameters['state_sample_length']

    batch_size = parameters['batch_size']
    seq_length = parameters['seq_length']  # seq_length = 1 doesnt work well because there is a squeeze somewhere
    n_layers = parameters['n_layers']
    hidden_dim = parameters['hidden_dim']

    train_mat = torch.from_numpy(train_mat)
    train_mat = train_mat.to(device)

    # We'll also set the model to the device that we defined earlier (default is CPU)
    model = model.to(device)

    # Define hyperparameters
    n_epochs = parameters['n_epochs']
    evaluation_interval = parameters['evaluation_interval']
    lr_start = parameters['lr_start']
    gamma = parameters['gamma']


    # Define Loss, Optimizer
    loss_fn = create_loss_fn(parameters['loss_funcs'], number_of_slots)

    step_decay = lambda epoch: 0.1 * np.power(0.5, np.floor((1 + epoch) / gamma))
    cosine_annealing = lambda epoch: 0.001 + 0.5 * (0.1 - 0.001) * (1 + np.cos(np.pi * epoch / gamma))
    poly_decay = lambda epoch: 0.1 * (1 - epoch / gamma) ** 1.0
    exp_decay = lambda epoch: 0.1 * np.exp(-0.1 * epoch/gamma)
    exp_log_decay = lambda epoch: 0.05 * np.exp(-np.log(2) * epoch / gamma)
    linear_decay = lambda epoch: 1 / (2 ** (epoch // (gamma/2)))

    # Define Lambda Function
    def lr_lambda(epoch):
        lr = 1.0
        for key, value in parameters['lr_func'].items():
            if value:
                if key == 'step_decay':
                    lr *= step_decay(epoch)
                elif key == 'cosine_annealing':
                    lr *= cosine_annealing(epoch)
                elif key == 'poly_decay':
                    lr *= poly_decay(epoch)
                elif key == 'exp_decay':
                    lr *= exp_decay(epoch)
                elif key == 'exp_log_decay':
                    lr *= exp_log_decay(epoch)
                elif key == 'linear_decay':
                    lr *= linear_decay(epoch)
        return lr

    # Define Optimizer and Scheduler
    optimizer = torch.optim.RAdam(model.parameters(), lr=lr_start)
    optimizer.param_groups[0]['initial_lr'] = lr_start  # set initial_lr for first parameter group
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=start_epoch)

    accumulated_losses = []
    ac_loss = 0
    norm_count = 0
    param_count = sum(p.numel() for p in model.parameters())
    required_params = {
        'L1_loss_on_weghits': parameters['loss_funcs'].get('L1_loss_on_weghits'),
        'L2_loss_on_weghits': parameters['loss_funcs'].get('L2_loss_on_weghits'),
        'param_count':  param_count
    }
    required_params = {k: torch.tensor(v).to(device) for k, v in required_params.items()}


    # Training Run
    for epoch in range(1, evaluation_interval + 1):
        # runs over batch index to go over all funcs
        randomized_train_mat = randomize_batch(train_mat,device)
        assert train_mat.shape == randomized_train_mat.shape
        for batch_index in range(0, randomized_train_mat.shape[0] - batch_size + 1, batch_size):
            hidden = model.init_hidden(batch_size, device)

            # runs over time index, jumps over seq_length
            for frame_index in range(0, fps * simulation_time - state_sample_length - seq_length, seq_length):
                optimizer.zero_grad()  # Clears existing gradients from previous epoch

                # run only on the first batch. continue with the same functions
                input_matrix, target_vector = create_seq_of_slice(randomized_train_mat, seq_length, batch_size, batch_index,
                                                                  frame_index, state_sample_length=state_sample_length,
                                                                  fps=fps, number_of_slots=number_of_slots,
                                                                  features=features,
                                                                  feature_dict=feature_dict)

                output, hidden = model(input_matrix, hidden)
                output = output.to(device)

                loss = loss_fn(output, target_vector.view(-1).long())


                # TODO : add the extra data needed, coppy train mat, create func that replace the 1 vale for all the bathc

                if required_params['L1_loss_on_weghits'] != 0:
                    l1_loss = 0.0
                    for param in model.parameters():
                        l1_loss += torch.norm(param, 1)
                    loss += required_params['L1_loss_on_weghits'] * l1_loss *0.1 / required_params['param_count']
                if required_params['L2_loss_on_weghits'] != 0:
                    l2_loss = 0.0
                    for param in model.parameters():
                        l2_loss += torch.norm(param, 1)**2
                    loss += required_params['L2_loss_on_weghits'] * l2_loss *0.1 / required_params['param_count']

                ac_loss += loss
                norm_count += 1

                loss.backward()  # Does backpropagation and calculates gradients
                optimizer.step()  # Updates the weights accordingly
                if isinstance(hidden, tuple):
                    assert len(hidden) == 2
                    hidden_state, cell_state = hidden
                    hidden_state = hidden_state.detach()
                    cell_state = cell_state.detach()
                    hidden = (hidden_state, cell_state)
                else:
                    assert hidden.shape == torch.Size([n_layers, batch_size, hidden_dim])
                    hidden = hidden.detach()

        scheduler.step()  # Update learning rate

        if epoch % update_loss_n == 0 and norm_count != 0:
            accumulated_losses.append(ac_loss / norm_count)
            ac_loss = 0
            norm_count = 0

        if epoch % 10 == 0 or epoch + start_epoch == update_loss_n:
            print('Epoch: {}/{}.............'.format(epoch + start_epoch, n_epochs), end=' ')
            print("Loss: {:.4f}".format(accumulated_losses[-1]))
            print("=======================================================================")

    path = parameters["path_to_save"] + '/' + parameters["MODEL_NAME"] + '/'
    if not os.path.exists(path):
        os.makedirs(path)

    # plt.plot(accumulated_losses)
    # plt.xlabel(f'{update_loss_n} Epochs')
    # plt.ylabel('Accumulated average Loss')
    # plt.title(f'Accumulated average Loss per {update_loss_n} Epochs')
    # plt.savefig(path + f'Accumulated_Loss_per_Epoch_for_{parameters["MODEL_NAME"]}.png')
    # plt.show(block=False)

    torch.save(model.state_dict(), path + parameters["MODEL_NAME"])
    print('\nSaved Model', path + parameters["MODEL_NAME"], '\n')

    return model, accumulated_losses


def test(parameters, test_mat, model, device=get_device(to_print=False)):
    # print(f'rnn_test of {parameters["MODEL_NAME"]}')

    simulation_time = parameters['simulation_time']
    fps = parameters['fps']
    number_of_slots = parameters['number_of_slots']
    features = parameters['features']
    feature_dict = parameters['feature_dict']

    state_sample_length = parameters['state_sample_length']

    seq_length = simulation_time * fps - state_sample_length

    test_mat = torch.from_numpy(test_mat)
    test_mat = test_mat.to(device)

    # model to device
    model = model.to(device)

    # Define Loss, Optimizer
    criterion = create_loss_fn(parameters['loss_funcs'], number_of_slots)

    ac_loss = 0
    norm_count = 0
    model_predicted_good_count = 0
    assert seq_length == test_mat.shape[2] - state_sample_length   # expected length

    for batch_index in range(0, test_mat.shape[0]):
        hidden = model.init_hidden(1, device)  # 1 is batch size

        input_matrix, target_vector = create_seq_of_slice(test_mat, seq_length, 1, batch_index,
                                                          0, state_sample_length=state_sample_length,
                                                          fps=fps, number_of_slots=number_of_slots,
                                                          features=features,
                                                          feature_dict=feature_dict)
        # print(input_matrix.device)
        output, hidden = model(input_matrix, hidden)
        output = output.to(device)

        # print(input_matrix.shape)
        # print(output.shape)
        # print(target_vector.shape)
        # print("========================================================")

        loss = criterion(output, target_vector.view(-1).long())
        loss = loss.detach()

        ac_loss += loss
        norm_count += 1

        # Get the predicted class for each sample in the batch
        pred = output.argmax(dim=1)
        pred = pred.detach()
        # print(f'target_vector',target_vector)
        # print(f'pred',pred)
        # Count the number of correct predictions

        model_predicted_good_count += torch.eq(pred, target_vector).sum()

        del input_matrix, target_vector, output, hidden

    average_loss = (ac_loss / norm_count).detach()
    model_predicted_good_count = model_predicted_good_count.detach()
    # print(f"average loss per step is {average_loss}")
    # print(f"model was correct {model_predicted_good_count} out of {(norm_count*seq_length)} times")
    # print(f"correct  = {100*model_predicted_good_count/(norm_count*seq_length)}% ")

    return average_loss, model_predicted_good_count/(norm_count*seq_length)


def predict(model, state, hidden):

    output, hidden = model(state, hidden)

    # Taking the class with the highest probability score from the output
    predict = torch.argmax(output, dim=1)

    return predict, hidden


def sample_with_model(model, parameters, x_high_res, y_high_res, device=get_device(to_print=False)):

    simulation_time = parameters['simulation_time']
    fps = parameters['fps']
    number_of_slots = parameters['number_of_slots']
    feature_dict = parameters['feature_dict']

    state_sample_length = parameters['state_sample_length']

    model.to(device)

    # start with the state sample length and sample additional points from y_high_res using interpolation
    unif_slots = get_unif_vec(x_high_res, number_of_slots, simulation_time, fps)
    x_sampled = unif_slots[:state_sample_length]

    y_sampled = sample_function(x_high_res, y_high_res, x_sampled)  # lets say i have the samples at the x_sampled points

    hidden = model.init_hidden(1, device)
    for i in range(state_sample_length, simulation_time*fps):

        # use the 20 y values and x values to make an input to the model and get its prediction
        x_input = x_sampled[(i - state_sample_length):i]
        y_input = y_sampled[(i - state_sample_length):i]

        input_to_model = np.concatenate((x_input, y_input), axis=0)

        if feature_dict.get('derivative') == 1:
            derivative_at_best = derivative(x_input,y_input)
            input_to_model = np.concatenate((input_to_model, derivative_at_best), axis=0)

        if feature_dict.get('FFT') == 1:
            FFT = np.fft.fft(y_input)

            FFT_real = FFT.real
            FFT_imaginary = FFT.imag
            input_to_model = np.concatenate((input_to_model, FFT_real), axis=0)
            input_to_model = np.concatenate((input_to_model, FFT_imaginary), axis=0)

        input_to_model = input_to_model[np.newaxis, np.newaxis, :]  # hopefuly this will create a size 1 batch dimantion and a sequence dimantion

        input_to_model = torch.from_numpy(input_to_model).float().to(device)

        index_predicted, hidden = predict(model,input_to_model, hidden)
        # print(input_to_model)
        # print(index_predicted)
        # input()

        # calculate what time in seconds correlates to the chosen action
        x_new = get_real_time_from_index_and_frame(index_predicted, i, number_of_slots, x_high_res, simulation_time, fps)

        x_sampled = np.append(x_sampled, x_new)

        # sample the new y value
        y_sampled = sample_function(x_high_res, y_high_res, x_sampled)  # lets say i have the samples at the x_sampled points

    return x_sampled, y_sampled


def get_real_time_from_index_and_frame(slot_index, frame_index, number_of_slots, x_high_res, simulation_time, fps):

    number_of_frames = simulation_time * fps
    frame_length = len(x_high_res) / number_of_frames
    slot_length = frame_length / number_of_slots

    x_slot = frame_length * frame_index + slot_index * slot_length

    if isinstance(x_slot, np.ndarray):
        x_slot = np.round(x_slot)
        return x_high_res[x_slot]
    elif isinstance(x_slot, torch.Tensor):
        x_high_res = torch.from_numpy(x_high_res).to(x_slot.device)
        x_slot = torch.round(x_slot).long()
        return x_high_res[x_slot].detach().cpu().numpy()
    else:
        raise TypeError("x_slot must be a numpy array or a torch tensor")


# def get_real_time_from_index_and_frame(slot_index, frame_index, number_of_slots, x_high_res, simulation_time, fps):
#
#     number_of_frames = simulation_time * fps
#     frame_length = len(x_high_res) / number_of_frames
#     slot_length = frame_length / number_of_slots
#
#     x_slot = frame_length * frame_index + slot_index * slot_length
#     x_slot = int(x_slot)
#
#     return x_high_res[x_slot]
