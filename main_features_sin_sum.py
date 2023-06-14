import pandas as pd
from shutil import rmtree

from Data_Base.Best_Slots import get_unif_vec, sample_function
from supervised.model_train import sample_with_model
from supervised.model_load import load_model
from signal_class.signal_class import *
from signal_class.utils import *
from signal_class.signal_error import *


def get_error_func(error_name="MSE"):
    error_functions = {
        "MSE": get_mean_square_error,
        "MAE": get_onenorm_error,
        "Max_error": get_max_error,
    }

    if error_name in error_functions:
        return error_functions[error_name]
    else:
        raise NameError("Invalid error name")


def main():
    simulation_time = 7
    fps = 30
    factor = 3
    number_of_slots = 10
    state_sample_length = 20

    if not os.path.exists("Data"):
        os.mkdir("Data")

    directory = "Data/signals_features_sin_sum"

    # Check if the directory exists
    if os.path.exists(directory):
        # Get a list of all files in the directory
        file_list = os.listdir(directory)

        # Iterate over the files in the directory
        for filename in file_list:
            # Create the file path
            file_path = os.path.join(directory, filename)

            # Check if the file does not match the desired file names
            if not ((filename.startswith("signals_features_sin_sum_plot") and filename.endswith(".png")) or
                     (filename.startswith("signals_features_sin_sum") and filename.endswith(".csv"))):
                # Delete the file
                try:
                    os.remove(file_path)
                except IsADirectoryError:
                    rmtree(file_path)

        # Create the directory if it does not exist
        os.makedirs(directory, exist_ok=True)
    else:
        # Create the directory if it does not exist
        os.mkdir(directory)

    features_dict = {
        "1_sin": ['sin_with_phase'],
        "2_sin": ['sin_with_phase', 'sin_with_phase'],
        "3_sin": ['sin_with_phase', 'sin_with_phase', 'sin_with_phase'],
        "4_sin": ['sin_with_phase', 'sin_with_phase', 'sin_with_phase', 'sin_with_phase'],
        "5_sin": ['sin_with_phase', 'sin_with_phase', 'sin_with_phase', 'sin_with_phase', 'sin_with_phase'],
    }

    sampling_methods = ["supervised_model", "uniform", "random"]
    interpolation_method = "interp"  # interp, CubicSpline ,  SecondOrder

    supervised_parameters = {
        'model_path_1_sin': 'supervised/models/interp_sin_sum/interp/n_epochs-600_LSTM_dropout-0.5_hidden-zeros/interp_sin_sum_interp_both_freq_1_sin_with_phase/',
        'model_path_2_sin': 'supervised/models/interp_sin_sum/interp/n_epochs-600_LSTM_dropout-0.5_hidden-zeros/interp_sin_sum_interp_both_freq_2_sin_with_phase/',
        'model_path_3_sin': 'supervised/models/interp_sin_sum/interp/n_epochs-600_LSTM_dropout-0.5_hidden-zeros/interp_sin_sum_interp_both_freq_3_sin_with_phase/',
        'model_path_4_sin': 'supervised/models/interp_sin_sum/interp/n_epochs-600_LSTM_dropout-0.5_hidden-zeros/interp_sin_sum_interp_both_freq_4_sin_with_phase/',
        'model_path_5_sin': 'supervised/models/interp_sin_sum/interp/n_epochs-600_LSTM_dropout-0.5_hidden-zeros/interp_sin_sum_interp_both_freq_5_sin_with_phase/',
        'simulation_time': simulation_time,
        'fps': fps,
        'number_of_slots': number_of_slots,  # upsampling_factor from DQN dict
        'feature_dict': {'derivative': 0, 'FFT': 0},
        'state_sample_length': state_sample_length
    }

    show_examples_unif_model = False

    signals_count = 100  # for averaging
    freqs = (1, 12)

    error_type = "MSE"  # MSE ,  MaxError ,  L1
    error_func = get_error_func(error_type)

    features_mse_mat = np.zeros([len(features_dict), len(sampling_methods)])
    features_mse_avg_mat = np.zeros([len(features_dict), len(sampling_methods)])

    k = -1
    for feature_name, func_type in tqdm(features_dict.items()):
        k += 1
        for signal_num in range(signals_count):

            if not os.path.exists(directory + f"/{feature_name}"):
                os.mkdir(directory + f"/{feature_name}")

            signal1 = SignalClass(simulation_time, freqs=freqs, factor=factor)
            signal1.create_signal(func_type, op=lambda a, b: a + b)
            signal1.save_high_res(name=directory + f"/{feature_name}/signal{signal_num}")

            x_high_res, y_high_res = signal1.get_high_res_vec()

            for i, sampling_method in enumerate(sampling_methods):

                if sampling_method == "supervised_model":
                    model = load_model(supervised_parameters[f'model_path_{feature_name}'])
                    x_sampled, y_sampled = sample_with_model(model, supervised_parameters, x_high_res, y_high_res)
                    x_interp, y_interp = signal1.get_interpolation_vec(x_sampled, y_sampled, interpolation_method)
                    model_mse = error_func(y_high_res, y_interp)
                    signal1.save_signal_with_interpolations_from_x_y_sampled(interpolation_method, sampling_method,
                                                                             x_sampled, y_sampled,
                                                                             name=directory + f"/{feature_name}/signal{signal_num}_{sampling_method}_{error_type}_of_{round(model_mse, 5)}")
                    features_mse_mat[k, i] = model_mse

                    if show_examples_unif_model:
                        unif_slots = get_unif_vec(x_high_res, number_of_slots, simulation_time, fps)
                        y_sampled_unif = sample_function(x_high_res, y_high_res, unif_slots)
                        x_interp_unif, y_interp_unif = signal1.get_interpolation_vec(unif_slots, y_sampled_unif,
                                                                                     interpolation_method)

                        plt.plot(x_high_res, y_high_res, label='High Resolution')  # Plot x_high_res and y_high_res
                        plt.plot(x_interp_unif, y_interp_unif, label='Interpolation (Uniform)')  # Plot x_interp_unif and y_interp_unif
                        plt.plot(x_interp, y_interp, label=f'Interpolation ({sampling_method.capitalize()})')  # Plot x_interp and y_interp

                        # Set plot labels and title
                        plt.xlabel('Time [sec]')
                        plt.ylabel('signal')
                        plt.title('Signal Plot')
                        plt.legend()

                        plt.show()
                elif sampling_method == "uniform":
                    unif_slots = get_unif_vec(x_high_res, number_of_slots, simulation_time, fps)
                    y_interp = interpolate_samples(x_high_res, unif_slots,
                                                   sample_function(x_high_res, y_high_res, unif_slots),
                                                   interpolation_method)
                    unif_mse = error_func(y_high_res, y_interp)
                    y_sampled = sample_function(x_high_res, y_high_res, unif_slots)
                    signal1.save_signal_with_interpolations_from_x_y_sampled(interpolation_method, sampling_method,
                                                                             unif_slots, y_sampled,
                                                                             name=directory + f"/{feature_name}/signal{signal_num}_{sampling_method}_{error_type}_of_{round(unif_mse, 5)}")
                    features_mse_mat[k, i] = unif_mse
                else:
                    x_sampled, y_sampled = signal1.get_sampled_vec(fps, sampling_method)
                    x_interp, y_interp = signal1.get_interpolation_vec(x_sampled, y_sampled, interpolation_method)
                    others_mse = error_func(y_high_res, y_interp)
                    signal1.save_signal_with_interpolations(interpolation_method, fps, sampling_method,
                                                            name=directory + f"/{feature_name}/signal{signal_num}_{sampling_method}_{error_type}_of_{round(others_mse, 5)}")
                    features_mse_mat[k, i] = others_mse

        features_mse_avg_mat += features_mse_mat
    print(features_mse_mat)
    features_mse_avg_mat /= len(features_dict)

    # Plotting the bar chart
    x_labels = list(features_dict.keys())
    x = np.arange(len(x_labels))
    width = 0.2  # Width of the bars

    fig, ax = plt.subplots()
    for i, method in enumerate(sampling_methods):
        ax.bar(x + i * width, features_mse_avg_mat[:, i], width, label=method)

    ax.set_xlabel('Features')
    ax.set_ylabel(f'{str(error_type).capitalize()} Average')
    ax.set_title(f'{str(error_type).capitalize()} Average by sin addition and Sampling Methods\nWith linear interpolation')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    plt.show(block=False)

    # Define the file path and check if it already exists
    file_path_csv = directory + f"/signals_features_sin_sum_{str(error_type)}.csv"
    counter = 1
    while os.path.exists(file_path_csv):
        file_path_csv = directory + f"/signals_features_sin_sum_{str(error_type)}_{counter}.csv"
        counter += 1

    pd.DataFrame(features_mse_avg_mat, index=x_labels).to_csv(file_path_csv, header=sampling_methods, index=True)

    # Define the file path and check if it already exists
    file_path = directory + "/signals_features_sin_sum_plot.png"
    counter = 1
    while os.path.exists(file_path):
        file_path = directory + f"/signals_features_sin_sum_plot_{counter}.png"
        counter += 1

    # Save the plot
    plt.savefig(file_path)




if __name__ == '__main__':
    main()
