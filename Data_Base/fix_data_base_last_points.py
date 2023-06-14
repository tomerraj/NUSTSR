import tqdm
import os
import sys

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from Data_Base.Best_Slots import *


PATH_end = '.npy'


def calc_last_points(x_high_res, y_high_res, x_slots, number_of_points, number_of_slots, simulation_time, fps, interpolation_method):
    number_of_frames = simulation_time * fps

    for frame in range(number_of_frames - number_of_points, number_of_frames):
        min_mse = get_mse_from_interpolation_cropped(x_high_res, y_high_res, x_slots, 0, simulation_time, fps, interpolation_method, min=20, max=20)
        min_slot = 0
        n = number_of_frames - frame - 1

        for i in range(0, number_of_slots ** (n + 1)):
            for j in range(n, 0, -1):
                x_slots = change_slot_by_index_in_x_slot(x_slots, frame + j, (i//10**j) % 10, number_of_slots, x_high_res, simulation_time, fps)

            x_slots = change_slot_by_index_in_x_slot(x_slots, frame, i % 10, number_of_slots, x_high_res, simulation_time, fps)
            new_mse = get_mse_from_interpolation_cropped(x_high_res, y_high_res, x_slots, frame, simulation_time, fps, interpolation_method, min=20, max=20)

            if new_mse < min_mse:
                min_mse = new_mse
                min_slot = i % 10

        x_slots = change_slot_by_index_in_x_slot(x_slots, frame, min_slot, number_of_slots, x_high_res, simulation_time, fps)
        for j in range(1, n):
            x_slots = change_slot_by_index_in_x_slot(x_slots, frame + j, 0, number_of_slots, x_high_res, simulation_time, fps)

    return x_slots


def fix_data_base(PATH):
    fps = 30
    simulation_time = 7
    number_of_slots = 10

    dicts = np.load(PATH, allow_pickle=True)

    x_high_res_dict = dicts[0]
    x_high_res = x_high_res_dict["x_high_res"]
    fixed_dict_list = [x_high_res_dict]

    dicts = dicts[1:]

    for i in tqdm(range(len(dicts))):
        dict = dicts[i]
        best_slots_interp = dict["best_slots_interp"]
        best_slots_CubicSpline = dict["best_slots_CubicSpline"]
        best_slots_SecondOrder = dict["best_slots_SecondOrder"]

        if np.isnan(best_slots_interp).all() and np.isnan(best_slots_SecondOrder).all():
            x_slot_index = translate_from_absolute_time_to_index_of_vector(best_slots_CubicSpline, number_of_slots, fps)
            number_of_points = 0
            for j in range(len(x_slot_index)-1, -1, -1):
                if x_slot_index[j] == 0: number_of_points += 1
                else: break
            if number_of_points != 4:
                fixed_dict_list.append(dict)
                continue
            x_slots_new = calc_last_points(x_high_res, dict["y_high_res"], best_slots_CubicSpline, 4, number_of_slots,
                             simulation_time, fps, 'CubicSpline')
            dict["best_slots_CubicSpline"] = x_slots_new

        if np.isnan(best_slots_CubicSpline).all() and np.isnan(best_slots_SecondOrder).all():
            x_slot_index = translate_from_absolute_time_to_index_of_vector(best_slots_interp, number_of_slots, fps)
            number_of_points = 0
            for j in range(len(x_slot_index)-1, -1, -1):
                if x_slot_index[j] == 0: number_of_points += 1
                else: break
            if number_of_points != 4:
                fixed_dict_list.append(dict)
                continue
            x_slots_new = calc_last_points(x_high_res, dict["y_high_res"], best_slots_interp, number_of_points, number_of_slots,
                             simulation_time, fps, 'interp')
            dict["best_slots_interp"] = x_slots_new

        if np.isnan(best_slots_CubicSpline).all() and np.isnan(best_slots_interp).all():
            x_slot_index = translate_from_absolute_time_to_index_of_vector(best_slots_SecondOrder, number_of_slots, fps)
            number_of_points = 0
            for j in range(len(x_slot_index)-1, -1, -1):
                if x_slot_index[j] == 0: number_of_points += 1
                else: break
            if number_of_points != 4:
                fixed_dict_list.append(dict)
                continue
            x_slots_new = calc_last_points(x_high_res, dict["y_high_res"], best_slots_SecondOrder, number_of_points, number_of_slots,
                             simulation_time, fps, 'SecondOrder')
            dict["best_slots_SecondOrder"] = x_slots_new

        fixed_dict_list.append(dict)

    np.save(PATH[:-len(PATH_end)] + '_fixed' + PATH_end, fixed_dict_list)


def main():
    interpolation_method = 'interp'  # interp ,  CubicSpline ,  SecondOrder
    PATHs = [
             'Data_Base_' + interpolation_method + '_both_freq_oscillating_chirp_with_phase_1-10_temp.npy',
             'Data_Base_' + interpolation_method + '_both_freq_oscillating_chirp_with_phase_10-20_temp.npy'
             ]

    for PATH in PATHs:
        print("Fixing the last points of data base:", PATH)
        fix_data_base(PATH)


if __name__ == '__main__':
    main()
