from supervised.model_train import sample_with_model
from supervised.model_load import load_model

from signal_class.signal_class import *
from signal_class.utils import *

import pandas as pd

import os
from shutil import rmtree

import numpy as np


def main():
    simulation_time = 7
    freq = (2, 12)
    fps = 30
    factor = 2

    if not os.path.exists("Data"):
        os.mkdir("Data")

    if os.path.exists("Data/signals"):
        rmtree("Data/signals")
    os.mkdir("Data/signals")

    if os.path.exists("Data/signals/signal_stats.txt"):
        os.remove("Data/signals/signal_stats.txt")

    all_type_list = ['rand_with_gaussian_sigma_150']

    sampling_methods = ["uniform", "random", "chebyshev"]
    interpolation_methods = ["CubicSpline"]  # interp, CubicSpline ,  SecondOrder

    mse_error_mat = np.zeros([len(sampling_methods), len(interpolation_methods)])
    mse_error_mat_addative = np.zeros([len(sampling_methods), len(interpolation_methods)])
    mse_no_edges_error_mat = np.zeros([len(sampling_methods), len(interpolation_methods)])
    mse_cut_edge_error_mat = np.zeros([len(sampling_methods), len(interpolation_methods)])

    DQN_parameters = {
        'PATH': 'DQN/model/',
        'MODEL_NAME': 'cubic_spline_withloss_26-03',
        'EXP_START_DECAY': 500,
        'EPS_START': 0.8,
        'EPS_END': 0.05,
        'upsampling_factor': 10,
        'BATCH_SIZE': 4,
        'GAMMA': 0.98,
        'state_sample_length': 50,
        'NUM_OF_FUNCTIONS': 60,
        'NUM_OF_EPISODES': 300,
        'TARGET_UPDATE': 1000,
        'initial_temperature': 0.8,
        'exploration_strategy': 'softmax'
    }

    supervised_parameters = {'model_path': 'supervised/models2/to_poster_base_long_18-5-23/CubicSpline/LSTM_derivative_FFT_dropout:0.4_hidden:rand_unif/to_poster_base_long_18-5-23_CubicSpline_low_freq_simple/',
                             'simulation_time': simulation_time,
                             'fps': fps,
                             'number_of_slots': 10,  # upsampling_factor from DQN dict
                             'feature_dict': {'derivative': 1, 'FFT': 1},
                             'state_sample_length': 20}

    signals_count = 10
    for k in range(signals_count):
        os.mkdir(f"Data/signals/signal{k + 1}")

        print("starting signal", k + 1, "out of", signals_count)
        type_list = random_pop(all_type_list)
        signal1 = SignalClass(simulation_time, freqs=freq, factor=factor)
        signal1.create_signal(type_list, op=lambda a, b: a + b)
        signal1.save_high_res(name=f"Data/signals/signal{k + 1}/signal{k + 1}")
        signal1.show_high_res()

        for i, sampling_method in enumerate(sampling_methods):
            if sampling_method == "supervised_model":
                x_high_res, y_high_res = signal1.get_high_res_vec()
                model = load_model(supervised_parameters['model_path'])
                x_sampled, y_sampled = sample_with_model(model, supervised_parameters, x_high_res, y_high_res)
                signal1.save_signal_with_interpolations_from_x_y_sampled(interpolation_methods, sampling_method, x_sampled, y_sampled,
                                                        name=f"Data/signals/signal{k + 1}/signal{k + 1}_{sampling_method}")
            else:
                x_sampled, y_sampled = signal1.get_sampled_vec(fps, sampling_method, op=lambda t: t * t,
                                                               NUS_parameters=DQN_parameters)
                signal1.save_signal_with_interpolations(interpolation_methods, fps, sampling_method, op=lambda t: t * t,
                                                        NUS_parameters=DQN_parameters,
                                                        name=f"Data/signals/signal{k + 1}/signal{k + 1}_{sampling_method}")

            for j, interpolation_method in enumerate(interpolation_methods):
                x_interp, y_interp = signal1.get_interpolation_vec(x_sampled, y_sampled, interpolation_method)
                x_high_res, y_high_res = signal1.get_high_res_vec()

                # print(interpolation_method,np.unique(x_interp-x_high_res))
                mse_error_mat[i, j] = get_mean_square_error(y_high_res, y_interp)
                mse_error_mat_addative[i, j] += get_mean_square_error(y_high_res, y_interp)
                mse_cut_edge_error_mat[i, j] += get_mse_remove_edges_by_frac(y_high_res, y_interp, factor=0.95)

        pd.DataFrame(mse_error_mat, index=sampling_methods).to_csv(f"Data/signals/signal{k + 1}/signal{k + 1}.csv",
                                                                   header=interpolation_methods, index=True)
        with open("Data/signals/signal_stats.txt", 'a') as fp:
            fp.write(f"For signal {k + 1}: {signal1.name}")
            fp.write("\n")
            fp.write(f"best preforming sampling methood: {sampling_methods[np.argmin(np.mean(mse_error_mat, axis=1))]}")
            fp.write("\n")
            fp.write(
                f"best preforming interp methood: {interpolation_methods[np.argmin(np.mean(mse_error_mat, axis=0))]}")
            fp.write("\n\n")


    # for the DQN
    # x_sampled_DQN, y_sampled_DQN = signal1.get_sampled_vec(fps, "DQN", op=lambda t: t * t,
    #                                                NUS_parameters=DQN_parameters)
    #
    # x_sampled_un, y_sampled_un = signal1.get_sampled_vec(fps, "uniform")
    #
    # print((x_sampled_DQN - x_sampled_un), '\n www \n')
    # print("the x_sampled_DQN:", x_sampled_DQN)

    with open("Data/signals/signal_stats.txt", 'a') as fp:
        fp.write(
            f"\nbest overall preforming sampling methood: {sampling_methods[np.argmin(np.mean(mse_error_mat_addative, axis=1))]}")
        fp.write("\n")
        fp.write(
            f"best overall preforming interp methood: {interpolation_methods[np.argmin(np.mean(mse_error_mat_addative, axis=0))]}")
        fp.write("\n\n")


if __name__ == '__main__':
    main()
