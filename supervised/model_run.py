import csv
import os
import sys

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from supervised.model_test import *


def save_result_matrix_csv(path_save, result_matrix, column_names, row_names):
    # Check if directory exists, create it if it doesn't
    if not os.path.exists(os.path.dirname(path_save)):
        os.makedirs(os.path.dirname(path_save))

    with open(path_save, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([""] + column_names)
        for row_name, row in zip(row_names, result_matrix):
            writer.writerow([str(row_name[
                                 20:-4])] + row.tolist())  # Convert row name to string and write row name followed by the data row as a list


parameters = {
    'PATH': 'models5',
    'MODEL_NAME': 'interp_sin',
    'path_to_save': '',
    'data_path': 'Data_Base_interp_both_freq_sin_with_phase_10-20.npy',
    'start_from_func': 0,
    'end_in_func': -1,  # -1 for all function in the data_path

    'fps': 30,
    'number_of_slots': 10,
    'simulation_time': 7,

    'state_sample_length': 20,
    'output_size': 10,  # same as number of slots
    'batch_size': 40,
    'seq_length': 7,
    'feature_dict': {'derivative': 0, 'FFT': 0},
    'features': -1,

    'hidden_dim': 70,
    'n_layers': 7,
    'architecture': {'RNN': 0, 'GRU': 0, 'LSTM': 1, 'TGLSTM': 0},
    # only one should be on, if not the first one would be choose
    'extra_to_model': {'dropout': 0.3, 'encoder_channels': 0, 'decoder_channels': 0, 'hidden': 'rand_gauss'},
    # the dropout value is the drop_prob, and the embedding value is the embedding_dim
    #  rand_gauss ,   rand_unif ,   zeros

    'n_epochs': 0,
    'lr_start': 0.0004,
    'evaluation_interval': 20,
    'gamma': 2000,
    'lr_func': {'step_decay': 0, 'cosine_annealing': 0, 'poly_decay': 0, 'exp_decay': 0, 'exp_log_decay': 0,
                'linear_decay': 1},  # the linear is more like a log acualy
    'loss_funcs': {'CrossEntropyLoss': 1, 'L1_loss_on_weghits': 0.8, 'L2_loss_on_weghits': 0, 'L1Loss': 0,
                   'MSELoss': 0}
}

column_names = ['train_mat avg_loss', 'train_mat correct prob', 'test_mat avg_loss', 'test_mat correct prob',
                'mse_avg from sample', 'mse_avg from unif', 'ratio_improve_avg',
                'mse_avg from sample on train', 'mse_avg from unif on train', 'ratio_improve_avg on train']

complex_test_counter = 0
resume_index = 0  # set this to the last print of complex_test_counter to start from that point

interpolation_method_array = ['interp']  # interp ,  CubicSpline ,  SecondOrder

architecture_array = [{'RNN': 0, 'GRU': 0, 'LSTM': 1, 'TGLSTM': 0}]
feature_dict_array = [{'derivative': 0, 'FFT': 0}]
extra_to_model_array = [{'dropout': 0.5, 'encoder_channels': 0, 'decoder_channels': 0, 'hidden': 'zeros'}]
n_epochs_array = [600]

total_complex_tests = len(interpolation_method_array) * len(architecture_array) * len(feature_dict_array) * \
                      len(extra_to_model_array) * len(n_epochs_array)
print(f'starting from complex test {resume_index} out of {total_complex_tests}')

for interpolation_method in interpolation_method_array:
    for architecture in architecture_array:
        for feature_dict in feature_dict_array:
            for extra_to_model in extra_to_model_array:
                for n_epochs in n_epochs_array:

                    if resume_index != 0:
                        if resume_index > complex_test_counter:
                            complex_test_counter += 1
                            continue

                    PATHs = ['Data_Base_' + interpolation_method + '_both_freq_sin_with_phase_10-20.npy',
                             'Data_Base_' + interpolation_method + '_both_freq_sin_with_phase_1-10.npy']

                    PATHs = [os.path.join(os.path.dirname(current_dir), 'Data_Base', path) for path in PATHs]
                    if any(not os.path.exists(path) for path in PATHs):
                        continue

                    extra_name = ''
                    extra_name += 'n_epochs-' + str(n_epochs) + '_'

                    for key, value in architecture.items():
                        if value != 0:
                            extra_name += key + '_'
                            break

                    for key, value in feature_dict.items():
                        if value != 0:
                            extra_name += key + '_'

                    for key, value in extra_to_model.items():
                        if value != 0:
                            extra_name += key + '-' + str(value) + '_'

                    extra_name = extra_name[:-1]

                    parameters_new = copy.deepcopy(parameters)
                    parameters_new["architecture"] = architecture
                    parameters_new["feature_dict"] = feature_dict
                    parameters_new["extra_to_model"] = extra_to_model
                    parameters_new["n_epochs"] = n_epochs

                    new_path = parameters_new["PATH"] + '/' + parameters_new[
                        "MODEL_NAME"] + '/' + interpolation_method + '/' + extra_name + '/'
                    parameters_new["path_to_save"] = new_path

                    result_matrix = complex_test(parameters_new, interpolation_method, PATHs)

                    path_save = new_path + 'result_matrix.csv'
                    save_result_matrix_csv(path_save, result_matrix, column_names, PATHs)
                    complex_test_counter += 1
                    print(f'(now saved complex test #{complex_test_counter} out of {total_complex_tests})')
