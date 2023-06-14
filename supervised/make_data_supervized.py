import torch
from torch.fft import fft
import os
import sys

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from Data_Base.Best_Slots import *


# each point in training get a slice of  bachdim x (x,y,dx,...) x 20
# remember to cut the xd corectly
# the first 20 data points are uniform data points
# run over 50 slices befor gradient update


def get_training_data(path, start_from_func=0, end_in_func=-1, test_percent=0.2, state_sample_length=20,
                      fps=30, number_of_slots=10 , feature_dict = {}):
    """
    return a dict with 2 values: training_mat, test_mat
    both are a 3D array/

    batch_dim x (x,y,dx,...) x 210

    the x values are 20 are uniform  and the rest are best for the specific interp,
    the y is the sampled values at those x
    """

    load_data_dict = np.load(path, allow_pickle=True)

    training_data_list = []

    x_high_res_dict = load_data_dict[0]
    x_high_res = x_high_res_dict["x_high_res"]

    unif_slots = get_unif_vec(x_high_res, number_of_slots, x_high_res_dict["simulation_time"], fps)
    unif_slots = unif_slots[0:state_sample_length]


    dicts = load_data_dict[1:]
    dicts = dicts[start_from_func:] if end_in_func == -1 else dicts[start_from_func:end_in_func]

    for i, dict in enumerate(dicts):

        data_list = []

        best_slots_interp = dict["best_slots_interp"]
        best_slots_CubicSpline = dict["best_slots_CubicSpline"]
        best_slots_SecondOrder = dict["best_slots_SecondOrder"]

        if np.isnan(best_slots_interp).all() and np.isnan(best_slots_SecondOrder).all():
            best_x_time = np.concatenate((unif_slots, best_slots_CubicSpline[state_sample_length:]), axis=None)
            data_list.append(best_x_time)

        if np.isnan(best_slots_CubicSpline).all() and np.isnan(best_slots_SecondOrder).all():
            best_x_time = np.concatenate((unif_slots, best_slots_interp[state_sample_length:]), axis=None)
            data_list.append(best_x_time)

        if np.isnan(best_slots_CubicSpline).all() and np.isnan(best_slots_interp).all():
            best_x_time = np.concatenate((unif_slots, best_slots_SecondOrder[state_sample_length:]), axis=None)
            data_list.append(best_x_time)

        best_y_value = sample_function(x_high_res, dict["y_high_res"], best_x_time)
        data_list.append(best_y_value)

        if feature_dict.get('derivative') == 1:
            derivative_at_best = derivative(best_x_time,best_y_value)  # need to make it work with 2 full vectors of the sampled points/ best to do with convolve and find a vector that does it
            data_list.append(derivative_at_best)

        training_data_list.append(np.stack(data_list, axis=1))

    training_mat = np.stack(training_data_list, axis=2)
    training_mat = np.transpose(training_mat, (2, 1, 0))

    split_index = int(training_mat.shape[0] * (1 - test_percent))

    training_mat, test_mat = np.split(training_mat, [split_index], axis=0)

    data_dict = {'training_mat': training_mat, 'test_mat': test_mat}

    # if need more values from the data base dict we can add here

    return data_dict


def call_get_training_data(parameters, test_percent=0.2):
    data_dict = get_training_data(path=parameters['data_path'], test_percent=test_percent,
                                  start_from_func=parameters['start_from_func'], end_in_func=parameters['end_in_func'],
                                  state_sample_length=parameters['state_sample_length'], fps=parameters['fps'],
                                  number_of_slots=parameters['number_of_slots'],
                                  feature_dict=parameters['feature_dict'])

    return data_dict['training_mat'], data_dict['test_mat']

# input is matrix from the data,outputs are a matrix of a single batch of inputs and the expected outputs for this batch
# the input is flattened : batch x (data * type)
def slice_input(training_mat, batch_size, batch_index, time_index, state_sample_length=20, fps=30, number_of_slots=10 ,feature_dict = {} ):
    input_mat = training_mat[batch_index:batch_index + batch_size,
                :,
                time_index:time_index + state_sample_length]

    if feature_dict.get('derivative') == 1:  # there is a derivative dimention
        input_mat[:, 2, 0] = 0
        input_mat[:, 2, -1] = 0  # remove derivatives that are imposible to calc

    if feature_dict.get('FFT') == 1:
        FFT = fft(input_mat[:, 1, :], dim=1).unsqueeze(dim=1)
        FFT_real = torch.real(FFT)
        FFT_imag = torch.imag(FFT)
        input_mat = torch.cat((input_mat, FFT_real), dim=1)
        input_mat = torch.cat((input_mat, FFT_imag), dim=1)


    input_vector = input_mat.reshape(input_mat.shape[0], -1)

    output_index = training_mat[batch_index:batch_index + batch_size,
                   0,  # 0 is the x row
                   time_index + state_sample_length]

    assert output_index.shape == (batch_size, )

    output_index = get_slot_index_from_time_value(output_index, time_index + state_sample_length , fps,
                                                  number_of_slots)

    return input_vector, output_index


def create_seq_of_slice(training_mat, seq_length, batch_size, batch_index, time_index, state_sample_length=20, fps=30,
                        number_of_slots=10, features=3, feature_dict={}):
    input_list = []
    output_list = []
    for i in range(seq_length):
        input_vector, output = slice_input(training_mat, batch_size, batch_index, time_index + i,
                                           state_sample_length=state_sample_length, fps=fps,
                                           number_of_slots=number_of_slots, feature_dict=feature_dict)

        input_tensor = (input_vector.reshape((batch_size, 1, features * state_sample_length))).clone().detach()
        output_tensor = (output.reshape((batch_size, 1))).clone().detach()

        input_list.append(input_tensor)
        output_list.append(output_tensor)

    # input_list = [torch.tensor(tensor) for tensor in input_list]
    # output_list = [torch.tensor(tensor) for tensor in output_list]
    input_mat = torch.stack(input_list, dim=1).squeeze()
    output_mat = torch.stack(output_list, dim=1).squeeze()


    if len(input_mat.shape) == 2:
        # Add back the batch dimension if it's not present
        input_mat = input_mat.unsqueeze(0)
        output_mat = output_mat.unsqueeze(0)
    elif len(input_mat.shape) == 3:
        # The input already has a batch dimension
        pass
    else:
        raise ValueError(f"Unexpected input shape: {input_mat.shape}")

    # print(output_mat)

    return input_mat.float(), output_mat



def get_slot_index_from_time_value(time_vector, frame_index, fps=30, number_of_slots=10):
    # it is just for now, need to add or find function
    x_norm = time_vector - (frame_index / fps)
    best_slot_index_vector = torch.round(x_norm * fps * number_of_slots).int()

    return best_slot_index_vector

# def one_hot_encode(vector, num_classes):
#     # Create an empty matrix to hold the one-hot encoding
#     matrix = np.zeros((len(vector), num_classes), dtype=int)
#
#     # Set the appropriate element of each row to 1
#     for i, val in enumerate(vector):
#         matrix[i, val] = 1
#
#     return matrix


def derivative(x, y):
    len_x = len(x)
    assert len_x == len(y)
    dx_dy = np.zeros(len_x)
    for i in range(len_x - 2):
        dx_dy[i + 1] = derivative_step(y[i], y[i + 1], y[i + 2], x[i], x[i + 1], x[i + 2])

    return dx_dy


# calculate derivative for y1
def derivative_step(y0, y1, y2, x0, x1, x2):
    h0 = x1 - x0
    h1 = x2 - x1
    r = h1 / h0
    slope = (y2 * 1 - y0 * r * r - y1 * (1 - (r * r))) / h1 * (1 + r)
    return slope
