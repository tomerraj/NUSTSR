import pandas as pd
from shutil import rmtree

from Data_Base.Best_Slots import get_unif_vec, get_mse_from_interpolation, sample_function
from supervised.model_train import sample_with_model
from supervised.model_load import load_model
from signal_class.signal_class import *
from signal_class.utils import *


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
    factor = 2
    number_of_slots = 10
    state_sample_length = 20

    base_dir = "Data"


    show_examples_unif_model = False

    error_type = "MSE"  # Max_error ,  MSE ,  MAE
    func_type = 'sin_with_phase'

    sampling_methods = ["chebyshev", "boosting", "random", "uniform", "supervised_model"]
    interpolation_method = "interp"  # interp, CubicSpline ,  SecondOrder

    num_repetitions = 3  # for averaging
    signals_count = 85

    freq_min, freq_max = 1, 10

    func_type_name = func_type[:-11].capitalize() if func_type.endswith('_with_phase') else func_type.capitalize()

    name = "signals_sweep_" + interpolation_method + "_" + func_type_name + "_" + str(freq_min) + '-' + str(freq_max) + '_alot'

    model_path = 'supervised/models/interp_sin/interp_sin_interp_both_freq_sin_with_phase_1-10'
    model_path += '/'


    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    directory = base_dir + "/" + name

    # Check if the directory exists
    if os.path.exists(directory):
        # Get a list of all files in the directory
        file_list = os.listdir(directory)

        # Iterate over the files in the directory
        for filename in file_list:
            # Create the file path
            file_path = os.path.join(directory, filename)

            # Check if the file does not match the desired file names
            if not ((filename.startswith(name + "_plot") and filename.endswith(".png")) or
                     (filename.startswith(name) and filename.endswith(".csv"))):
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

    supervised_parameters = {
        'model_path': model_path,
        'simulation_time': simulation_time,
        'fps': fps,
        'number_of_slots': number_of_slots,  # upsampling_factor from DQN dict
        'feature_dict': {'derivative': 0, 'FFT': 0},
        'state_sample_length': state_sample_length
    }

    freqs = np.linspace(freq_min, freq_max, signals_count + 1)
    freqs_list = [(freqs[i], freqs[i + 1]) for i in range(len(freqs) - 1)]
    freqs_x_axis = np.zeros(signals_count)

    error_func = get_error_func(error_type)

    sweep_mse_mat = np.zeros([signals_count, len(sampling_methods)])
    sweep_mse_avg_mat = np.zeros([signals_count, len(sampling_methods)])

    for repetition in tqdm(range(num_repetitions)):
        for k, freq in enumerate(tqdm(freqs_list)):
            signal_num = str(k + 1).zfill(2)

            if not os.path.exists(directory + f"/signal{signal_num}"):
                os.mkdir(directory + f"/signal{signal_num}")

            signal1 = SignalClass(simulation_time, freqs=freq, factor=factor)
            freq_created = signal1.create_signal([func_type], op=lambda a, b: a + b)
            freqs_x_axis[k] += freq_created
            signal1.save_high_res(name=directory + f"/signal{signal_num}/signal{signal_num}_iter{repetition + 1}")

            print("\nstarting signal:", signal_num, "/", signals_count, ",repetition:", repetition + 1, "/", num_repetitions,
                  "; With freq:", round(freq_created, 6))

            x_high_res, y_high_res = signal1.get_high_res_vec()

            for i, sampling_method in enumerate(sampling_methods):

                if sampling_method == "supervised_model":
                    model = load_model(supervised_parameters['model_path'])
                    x_sampled, y_sampled = sample_with_model(model, supervised_parameters, x_high_res, y_high_res)
                    x_interp, y_interp = signal1.get_interpolation_vec(x_sampled, y_sampled, interpolation_method)
                    model_mse = error_func(y_high_res, y_interp)
                    signal1.save_signal_with_interpolations_from_x_y_sampled(interpolation_method, sampling_method,
                                                                            x_sampled, y_sampled,
                                                                            name=directory + f"/signal{signal_num}/signal{signal_num}_iter{repetition + 1}_{sampling_method}_{error_type}_of_{round(model_mse, 5)}")
                    sweep_mse_mat[k, i] = model_mse

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
                                                                            name=directory + f"/signal{signal_num}/signal{signal_num}_iter{repetition + 1}_{sampling_method}_{error_type}_of_{round(unif_mse, 5)}")
                    sweep_mse_mat[k, i] = unif_mse
                else:
                    x_sampled, y_sampled = signal1.get_sampled_vec(fps, sampling_method, op=lambda t: t * t)
                    x_interp, y_interp = signal1.get_interpolation_vec(x_sampled, y_sampled, interpolation_method)
                    others_mse = error_func(y_high_res, y_interp)
                    signal1.save_signal_with_interpolations(interpolation_method, fps, sampling_method, op=lambda t: t * t,
                                                            name=directory + f"/signal{signal_num}/signal{signal_num}_iter{repetition + 1}_{sampling_method}_{error_type}_of_{round(others_mse, 5)}")
                    sweep_mse_mat[k, i] = others_mse

        sweep_mse_avg_mat += sweep_mse_mat

    sweep_mse_avg_mat /= num_repetitions
    freqs_x_axis /= num_repetitions

    # Define the file path and check if it already exists
    file_path_csv = directory + '/' + name + f".csv"
    counter = 1
    while os.path.exists(file_path_csv):
        file_path_csv = directory + '/' + name + f"_{counter}.csv"
        counter += 1

    pd.DataFrame(sweep_mse_avg_mat, index=[f'Signal{j+1}-freq:{freqs_x_axis[j]}' for j in range(signals_count)]).to_csv(file_path_csv,
                                                                   header=sampling_methods, index=True)

    # Create a figure and axis
    fig, ax = plt.subplots()

    markers = ['s', 'd', 'X', '2', 'o']  # Example markers
    line_styles = ['-', ':', '-.', '--', '-']  # Example line styles
    colors = ['tab:purple', 'tab:red', 'tab:green', 'tab:orange', 'tab:blue']  # Example colors

    for i, sampling_method in enumerate(sampling_methods):
        marker = markers[i % len(markers)]  # Assign a marker based on the index, cycling through the available markers
        line_style = line_styles[i % len(line_styles)]  # Assign a line style based on the index, cycling through the available line styles
        color = colors[i % len(colors)]  # Assign a color based on the index, cycling through the available colors
        y = sweep_mse_avg_mat[:, i]  # Select the column corresponding to the current sampling method
        ax.plot(freqs_x_axis, y, marker=marker, linestyle=line_style, color=color, label=sampling_method.capitalize(), markersize=5)

    # Set title
    ax.set_title(f"{str(error_type)} vs. Frequency\nOf {func_type_name} signals with {interpolation_method}")
    # Set the x-axis and y-axis label
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(str(error_type))
    ax.legend()

    # Define the file path and check if it already exists
    file_path = directory + '/' + name + "_plot.png"
    counter = 1
    while os.path.exists(file_path):
        file_path = directory + '/' + name + f"_plot_{counter}.png"
        counter += 1

    # Save the plot
    plt.savefig(file_path)


if __name__ == '__main__':
    main()
