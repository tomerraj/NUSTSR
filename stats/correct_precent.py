from Data_Base.Best_Slots import translate_from_absolute_time_to_index_of_vector, get_mean_square_error
from supervised.model_train import sample_with_model, get_device
from supervised.model_load import load_model
from signal_class.utils import *


def run_func(model, data_path, interpolation_method="interp"):
    simulation_time = 7
    fps = 30
    number_of_slots = 10

    supervised_parameters = {'model_path': model, 'simulation_time': simulation_time, 'fps': fps,
                             'number_of_slots': number_of_slots, 'feature_dict': {'derivative': 0, 'FFT': 0},
                             'state_sample_length': 20}

    dicts = np.load(data_path, allow_pickle=True)
    model = load_model(supervised_parameters['model_path'])

    x_high_res_dict = dicts[0]
    x_high_res = x_high_res_dict["x_high_res"]

    device = get_device()

    mse_mat = []
    percentage_correct_mat = []
    dicts = dicts[1:]

    test_percent = 0.2
    split_index = int(len(dicts) * (1 - test_percent))
    training_funcs = dicts[:split_index]
    test_funcs = dicts[split_index:]  # only this one

    for func in test_funcs:
        y_high_res = func["y_high_res"]
        unif_mse = func["unif_mse"]
        x_sampled, y_sampled = sample_with_model(model, supervised_parameters, x_high_res, y_high_res, device)

        y_interp = interpolate_samples(x_high_res, x_sampled, y_sampled, interpolation_method)
        rnn_mse = get_mean_square_error(y_high_res, y_interp)

        if interpolation_method == 'interp':
            x_time_opt = func["best_slots_interp"]
        if interpolation_method == 'CubicSpline':
            x_time_opt = func["best_slots_CubicSpline"]

        count_correct = sum(0 == abs(np.round((x_sampled[20:] - x_time_opt[20:]) * fps * number_of_slots)) )

        percentage_correct = (count_correct / len(x_sampled)) * 100

        mse_mat.append([rnn_mse, unif_mse])
        percentage_correct_mat.append(percentage_correct)

    mse_mat = np.array(mse_mat)
    percentage_correct_mat = np.array(percentage_correct_mat)
    # mse_avg = np.mean(mse_mat, axis=0)
    ratio_improve = np.divide(mse_mat[:, 1], mse_mat[:, 0])
    ratio_improve_avg = np.mean(ratio_improve)
    percentage_correct_avg = np.mean(percentage_correct_mat)

    return ratio_improve_avg,percentage_correct_avg

# main start
model_path = ['supervised/models5/interp_sin_07-06-23/interp/n_epochs-600_LSTM_dropout-0.5_hidden-zeros/interp_sin_07-06-23_interp_both_freq_sin_with_phase_10-20/',
              'supervised/models5/interp_sin_sum_06-06-23/interp/n_epochs-600_LSTM_dropout-0.5_hidden-zeros/interp_sin_sum_06-06-23_interp_both_freq_1_sin_with_phase/',
              # 'supervised/models4/to_poster_sawtooth_22-5-23/interp/n_epochs-400_LSTM_dropout-0.5_hidden-zeros/to_poster_sawtooth_22-5-23_interp_both_freq_sawtooth_with_phase_1-10/',
              # 'supervised/models4/to_poster_sawtooth_22-5-23/interp/n_epochs-400_LSTM_dropout-0.5_hidden-zeros/to_poster_sawtooth_22-5-23_interp_both_freq_sawtooth_with_phase_10-20/',
              # 'supervised/models5/interp_oscillating_chirp_09-06-23/interp/n_epochs-600_LSTM_dropout-0.4_hidden-zeros/interp_oscillating_chirp_09-06-23_interp_both_freq_oscillating_chirp_with_phase_1-10/',
              # 'supervised/models5/interp_oscillating_chirp_09-06-23/interp/n_epochs-600_LSTM_dropout-0.4_hidden-zeros/interp_oscillating_chirp_09-06-23_interp_both_freq_oscillating_chirp_with_phase_10-20/',
              # 'supervised/models5/interp_pixel_up_07-06-23/interp/n_epochs-600_LSTM_dropout-0.4_hidden-zeros/interp_pixel_up_07-06-23_interp_pixel_up_GOT_60fps/',
              # 'supervised/models5/interp_rand_with_gaussian_08-06-23/interp/n_epochs-600_LSTM_dropout-0.5_hidden-zeros/interp_rand_with_gaussian_08-06-23_interp_rand_with_gaussian_sigma_001/',
              # 'supervised/models5/interp_rand_with_gaussian_08-06-23/interp/n_epochs-600_LSTM_dropout-0.5_hidden-zeros/interp_rand_with_gaussian_08-06-23_interp_rand_with_gaussian_sigma_035/',
              # 'supervised/models5/interp_rand_with_gaussian_08-06-23/interp/n_epochs-600_LSTM_dropout-0.5_hidden-zeros/interp_rand_with_gaussian_08-06-23_interp_rand_with_gaussian_sigma_070/',
              # 'supervised/models5/interp_rand_with_gaussian_08-06-23/interp/n_epochs-600_LSTM_dropout-0.5_hidden-zeros/interp_rand_with_gaussian_08-06-23_interp_rand_with_gaussian_sigma_105/',
              # 'supervised/models5/interp_rand_with_gaussian_08-06-23/interp/n_epochs-600_LSTM_dropout-0.5_hidden-zeros/interp_rand_with_gaussian_08-06-23_interp_rand_with_gaussian_sigma_140/',
              # 'supervised/models5/interp_sin_sum_06-06-23/interp/n_epochs-600_LSTM_dropout-0.5_hidden-zeros/interp_sin_sum_06-06-23_interp_both_freq_1_sin_with_phase/',
              # 'supervised/models5/interp_sin_sum_06-06-23/interp/n_epochs-600_LSTM_dropout-0.5_hidden-zeros/interp_sin_sum_06-06-23_interp_both_freq_2_sin_with_phase/',
              # 'supervised/models5/interp_sin_sum_06-06-23/interp/n_epochs-600_LSTM_dropout-0.5_hidden-zeros/interp_sin_sum_06-06-23_interp_both_freq_3_sin_with_phase/',
              # 'supervised/models5/interp_sin_sum_06-06-23/interp/n_epochs-600_LSTM_dropout-0.5_hidden-zeros/interp_sin_sum_06-06-23_interp_both_freq_4_sin_with_phase/',
              # 'supervised/models5/interp_sin_sum_06-06-23/interp/n_epochs-600_LSTM_dropout-0.5_hidden-zeros/interp_sin_sum_06-06-23_interp_both_freq_5_sin_with_phase/',
              ]
data_path = ['Data_Base/Data_Base_interp_both_freq_sin_with_phase_10-20.npy',
             'Data_Base/Data_Base_interp_both_freq_sin_with_phase_1-10.npy',
             # 'Data_Base/Data_Base_interp_both_freq_sawtooth_with_phase_1-10.npy',
             # 'Data_Base/Data_Base_interp_both_freq_sawtooth_with_phase_10-20.npy',
             # "/data/students/etaytomer/Data_Base/Data_Base_interp_both_freq_oscillating_chirp_with_phase_1-10.npy",
             # "/data/students/etaytomer/Data_Base/Data_Base_interp_both_freq_oscillating_chirp_with_phase_10-20.npy",
             # "/data/students/etaytomer/Data_Base/Data_Base_interp_pixel_up_GOT_60fps.npy",
             # "/data/students/etaytomer/Data_Base/Data_Base_interp_rand_with_gaussian_sigma_001.npy",
             # "/data/students/etaytomer/Data_Base/Data_Base_interp_rand_with_gaussian_sigma_035.npy",
             # "/data/students/etaytomer/Data_Base/Data_Base_interp_rand_with_gaussian_sigma_070.npy",
             # "/data/students/etaytomer/Data_Base/Data_Base_interp_rand_with_gaussian_sigma_105.npy",
             # "/data/students/etaytomer/Data_Base/Data_Base_interp_rand_with_gaussian_sigma_140.npy",
             # "/data/students/etaytomer/Data_Base/Data_Base_interp_both_freq_1_sin_with_phase.npy",
             # "/data/students/etaytomer/Data_Base/Data_Base_interp_both_freq_2_sin_with_phase.npy",
             # "/data/students/etaytomer/Data_Base/Data_Base_interp_both_freq_3_sin_with_phase.npy",
             # "/data/students/etaytomer/Data_Base/Data_Base_interp_both_freq_4_sin_with_phase.npy",
             # "/data/students/etaytomer/Data_Base/Data_Base_interp_both_freq_5_sin_with_phase.npy",
             ]
interp_of_model = ['interp',
                   'interp',
                   'interp',
                   'interp',
                   # 'interp',
                   # 'interp',
                   # 'interp',
                   # 'interp',
                   # 'interp',
                   # 'interp',
                   # 'interp',
                   # 'interp',
                   # 'interp',
                   # 'interp',
                   # 'interp',
                   # 'interp',
                   # 'interp',
                   ]       # interp, CubicSpline,  SecondOrder


# model_path = ['supervised/models5/cubic_oscillating_chirp_06-06-23/CubicSpline/n_epochs-600_LSTM_dropout-0.5_hidden-rand_gauss/cubic_oscillating_chirp_06-06-23_CubicSpline_both_freq_oscillating_chirp_with_phase_1-10/',
#               'supervised/models5/cubic_oscillating_chirp_06-06-23/CubicSpline/n_epochs-600_LSTM_dropout-0.5_hidden-rand_gauss/cubic_oscillating_chirp_06-06-23_CubicSpline_both_freq_oscillating_chirp_with_phase_10-20/',
#               'supervised/models5/cubic_sin_06-06-23/CubicSpline/n_epochs-600_LSTM_dropout-0.5_hidden-rand_gauss/cubic_sin_06-06-23_CubicSpline_both_freq_sin_with_phase_1-10/',
#               'supervised/models5/cubic_sin_06-06-23/CubicSpline/n_epochs-600_LSTM_dropout-0.5_hidden-rand_gauss/cubic_sin_06-06-23_CubicSpline_both_freq_sin_with_phase_10-20/',
#               ]
# data_path = ["/data/students/etaytomer/Data_Base/Data_Base_CubicSpline_both_freq_oscillating_chirp_with_phase_1-10.npy",
#              "/data/students/etaytomer/Data_Base/Data_Base_CubicSpline_both_freq_oscillating_chirp_with_phase_10-20.npy",
#              "/data/students/etaytomer/Data_Base/Data_Base_CubicSpline_both_freq_sin_with_phase_1-10.npy",
#              "/data/students/etaytomer/Data_Base/Data_Base_CubicSpline_both_freq_sin_with_phase_10-20.npy",
#              ]
# interp_of_model = ['CubicSpline',
#                    'CubicSpline',
#                    'CubicSpline',
#                    'CubicSpline',
#                    ]       # interp, CubicSpline,  SecondOrder

result_matrix = []

for i in range(len(model_path)):
    print("starting model:",i)


    ratio_improve_avg,percentage_correct_avg = run_func(model_path[i],data_path[i],interp_of_model[i])
    result_matrix.append([ratio_improve_avg, percentage_correct_avg])

# Convert the result matrix to a numpy array
result_matrix = np.array(result_matrix)

# Define the titles
titles = ['ratio_improve_avg', 'percentage_correct_avg']

# Insert the titles as the first row in the result matrix
result_matrix_with_titles = np.vstack([titles, result_matrix])

# Save the result matrix with titles as a CSV file
np.savetxt('result_matrix.csv', result_matrix_with_titles, delimiter=',', fmt='%s')

print("~finish")