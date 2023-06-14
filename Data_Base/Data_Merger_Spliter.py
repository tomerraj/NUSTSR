import numpy as np
import os
import sys

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from Data_Base.Best_Slots import translate_from_absolute_time_to_index_of_vector

PATH_start = 'Data_Base'
PATH_temp_end = '_temp.npy'
PATH_end = '.npy'


# given a list of file paths (the npy data bases) to merge into Data_Base{base_index}.npy
def merge_files(file_paths, save_name):
    PATH = file_paths[0]
    print(PATH)
    dicts = np.load(PATH, allow_pickle=True)
    last_x_high_res_dict = dicts[0]

    # checks if all the first (high_res_x) dicts are equal
    for PATH in file_paths:
        dicts = np.load(PATH, allow_pickle=True)
        x_high_res_dict = dicts[0]

        equal = set(x_high_res_dict.keys()) == set(last_x_high_res_dict.keys())

        if equal:
            for key, value in x_high_res_dict.items():
                if isinstance(value, np.ndarray) and isinstance(last_x_high_res_dict[key], np.ndarray):
                    equal = equal and (value == last_x_high_res_dict[key]).all()
                else:
                    equal = equal and (value == last_x_high_res_dict[key])

        if not equal:
            raise Exception("The x_high_res_dict are not identical between all Data Bases")
        else:
            last_x_high_res_dict = x_high_res_dict

    # merge the temps dict into one
    merge_dict_list = [last_x_high_res_dict]

    for PATH in file_paths:
        dicts = np.load(PATH, allow_pickle=True)
        dicts = dicts[1:]  # removes the first x_high_res_dict
        merge_dict_list.extend(dicts)

    PATH_save = PATH_start + '_' + save_name + "_merged" + PATH_end
    np.save(PATH_save, merge_dict_list)


def split_by_interpolation_type(file_PATH):
    dicts = np.load(file_PATH, allow_pickle=True)

    x_high_res_dict = dicts[0]
    dict_list_interp = [x_high_res_dict]
    dict_list_cubic = [x_high_res_dict]

    dicts = dicts[1:]

    for i, dict in enumerate(dicts):
        best_slots_interp = dict["best_slots_interp"]
        best_slots_CubicSpline = dict["best_slots_CubicSpline"]

        if np.isnan(best_slots_interp).all():
            dict_list_cubic.append(dict)

        if np.isnan(best_slots_CubicSpline).all():
            dict_list_interp.append(dict)

    PATH_save_interp = file_PATH[:-len(PATH_end)] + '_interp' + PATH_end
    PATH_save_cubic = file_PATH[:-len(PATH_end)] + '_CubicSpline' + PATH_end
    np.save(PATH_save_interp, dict_list_interp)
    np.save(PATH_save_cubic, dict_list_cubic)


def split_by_freq(file_PATH, nyquist_freq):
    dicts = np.load(file_PATH, allow_pickle=True)

    x_high_res_dict = dicts[0]
    dict_list_high_freq = [x_high_res_dict]
    dict_list_low_freq = [x_high_res_dict]

    dicts = dicts[1:]

    for i, dict in enumerate(dicts):
        freq = dict["freq"]

        if freq >= nyquist_freq:
            dict_list_high_freq.append(dict)
        else:
            dict_list_low_freq.append(dict)

    PATH_save_high_freq = file_PATH[:-len(PATH_end)] + '_high_freq' + PATH_end
    PATH_save_low_freq = file_PATH[:-len(PATH_end)] + '_low_freq' + PATH_end
    np.save(PATH_save_high_freq, dict_list_high_freq)
    np.save(PATH_save_low_freq, dict_list_low_freq)


def filter_zeros(file_PATH, zero_percent, number_of_slots=10, fps=30):
    dicts = np.load(file_PATH, allow_pickle=True)

    x_high_res_dict = dicts[0]
    dict_list_filtered = [x_high_res_dict]

    dicts = dicts[1:]

    number_of_frames = len(dicts[0]["best_slots_CubicSpline"]) if np.isnan(dicts[0]["best_slots_interp"]).all() else\
                       len(dicts[0]["best_slots_interp"])
    min_num_of_zeros = round(zero_percent * number_of_frames)

    for i, dict in enumerate(dicts):
        best_slots_interp = dict["best_slots_interp"]
        best_slots_CubicSpline = dict["best_slots_CubicSpline"]

        if np.isnan(best_slots_interp).all():
            best_slots_CubicSpline_indexes = translate_from_absolute_time_to_index_of_vector(best_slots_CubicSpline,
                                                                                             number_of_slots, fps)
            num_of_zeros = np.count_nonzero(best_slots_CubicSpline_indexes == 0)
            if num_of_zeros >= min_num_of_zeros:
                continue

        if np.isnan(best_slots_CubicSpline).all():
            best_slots_interp_indexes = translate_from_absolute_time_to_index_of_vector(best_slots_interp,
                                                                                        number_of_slots, fps)
            num_of_zeros = np.count_nonzero(best_slots_interp_indexes == 0)
            if num_of_zeros >= min_num_of_zeros:
                continue

        dict_list_filtered.append(dict)
    print(len(dicts) - len(dict_list_filtered) + 1, "function with more than", min_num_of_zeros, "zeros were filtered.")
    PATH_save_filtered = file_PATH[:-len(PATH_end)] + '_filtered' + PATH_end
    np.save(PATH_save_filtered, dict_list_filtered)


def main():
    fps = 30
    number_of_slots = 10
    interpolation_method = 'interp'  # interp ,  CubicSpline ,  SecondOrder

    # # 5
    # file_PATHS = ["Data_Base/Data_Base_" + interpolation_method + "_both_freq_complex_temp.npy"]
    # file_PATHS.append("Data_Base/Data_Base_" + interpolation_method + "_both_freq_complex.npy")
    # file_PATHS.append("Data_Base/Data_Base_" + interpolation_method + "_both_freq_complex_temp1.npy")
    # merge_files(file_PATHS, interpolation_method + '_both_freq_complex')
    #
    # # 4
    # file_PATHS = ["Data_Base/Data_Base_" + interpolation_method + "_both_freq_intermediate_temp.npy"]
    # file_PATHS.append("Data_Base/Data_Base_" + interpolation_method + "_both_freq_intermediate.npy")
    # file_PATHS.append("Data_Base/Data_Base_" + interpolation_method + "_both_freq_intermediate_temp1.npy")
    # merge_files(file_PATHS, interpolation_method + '_both_freq_intermediate')
    #
    # # 3
    # file_PATHS = ["Data_Base/Data_Base_" + interpolation_method + "_both_freq_simple_temp.npy"]
    # file_PATHS.append("Data_Base/Data_Base_" + interpolation_method + "_both_freq_simple.npy")
    # file_PATHS.append("Data_Base/Data_Base_" + interpolation_method + "_both_freq_simple_temp1.npy")
    # merge_files(file_PATHS, interpolation_method + '_both_freq_simple')
    #
    # # 2
    # file_PATHS = ["Data_Base/Data_Base_" + interpolation_method + "_high_freq_simple_temp.npy"]
    # file_PATHS.append("Data_Base/Data_Base_" + interpolation_method + "_high_freq_simple.npy")
    # file_PATHS.append("Data_Base/Data_Base_" + interpolation_method + "_high_freq_simple_temp1.npy")
    # merge_files(file_PATHS, interpolation_method + '_high_freq_simple')
    #
    # # 1
    # file_PATHS = ["Data_Base/Data_Base_" + interpolation_method + "_low_freq_simple_temp.npy"]
    # file_PATHS.append("Data_Base/Data_Base_" + interpolation_method + "_low_freq_simple.npy")
    # file_PATHS.append("Data_Base/Data_Base_" + interpolation_method + "_low_freq_simple_temp1.npy")
    # merge_files(file_PATHS, interpolation_method + '_low_freq_simple')

    # # 1
    # for i in range(1, 5):
    #     interpolation_method = 'interp'  # interp ,  CubicSpline ,  SecondOrder
    #     file_PATHS = [f"Data_Base_" + interpolation_method + f"_both_freq_{i}_sin_with_phase_temp.npy"]
    #     file_PATHS.append(f"Data_Base_" + interpolation_method + f"_both_freq_{i}_sin_with_phase.npy")
    #     # file_PATHS.append(f"Data_Base/Data_Base_" + interpolation_method + f"_both_freq_{i}_sin_with_phase_temp1.npy")
    #     merge_files(file_PATHS, interpolation_method + f'_both_freq_{i}_sin_with_phase')


    # freqs = '10-20'  # 10-20
    # interpolation_method = 'interp'  # interp ,  CubicSpline ,  SecondOrder
    #
    # # 2 oscillating_chirp_with_phase
    # file_PATHS = ["Data_Base_" + interpolation_method + "_both_freq_oscillating_chirp_with_phase_" + freqs + "_temp.npy"]
    # # file_PATHS.append("Data_Base_" + interpolation_method + "_both_freq_oscillating_chirp_with_phase_" + freqs + "_temp1.npy")
    # file_PATHS.append("Data_Base_" + interpolation_method + "_both_freq_oscillating_chirp_with_phase_" + freqs + ".npy")
    # merge_files(file_PATHS, interpolation_method + "_both_freq_oscillating_chirp_with_phase_" + freqs)


    interpolation_method = 'interp'  # interp ,  CubicSpline ,  SecondOrder

    # 2 oscillating_chirp_with_phase
    file_PATHS = ["Data_Base_" + interpolation_method + "_both_freq_oscillating_chirp_with_phase_10-20.npy"]
    # file_PATHS.append("Data_Base_" + interpolation_method + "_both_freq_oscillating_chirp_with_phase_" + freqs + "_temp1.npy")
    file_PATHS.append("Data_Base_" + interpolation_method + "_both_freq_oscillating_chirp_with_phase_1-10.npy")
    merge_files(file_PATHS, interpolation_method + "_both_freq_oscillating_chirp_with_phase_1-20")


if __name__ == '__main__':
    main()
