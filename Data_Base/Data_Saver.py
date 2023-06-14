import os
import sys

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

# Import the modules using relative paths
from signal_class.signal_generator import *
from signal_class.signal_class import *
from Data_Base.Best_Slots import *


def create_dict_for_x_high_res(simulation_time, num_of_samples):
    dict = {"x_high_res": np.linspace(0.0, simulation_time, num_of_samples),
            "simulation_time": simulation_time,
            "num_of_samples": num_of_samples}
    return dict


def create_dict(type_list, simulation_time, freq, factor, y_high_res, unif_mse, best_slots_interp, best_slots_SecondOrder, best_slots_CubicSpline):
    dict = {"type_list": type_list,
            "simulation_time": simulation_time,
            "freq": freq,
            "factor": factor,
            "y_high_res": y_high_res,
            "unif_mse": unif_mse,
            "best_slots_interp": best_slots_interp,
            "best_slots_SecondOrder": best_slots_SecondOrder,
            "best_slots_CubicSpline": best_slots_CubicSpline}
    return dict


interpolation_method = 'CubicSpline'  # interp , CubicSpline ,  SecondOrder
database_name = '_both_freq_sin_with_phase_1-10'
PATH = 'Data_Base/Data_Base_' + interpolation_method + database_name

num_of_samples = HIGH_RES_SAMPLES
simulation_time = 7
factor = 2
fps = 30
number_of_slots = 10
freqs = (1, 10)
unif_slots = get_unif_vec(np.linspace(0.0, simulation_time, num_of_samples), number_of_slots, simulation_time, fps)

all_type_list = ['sin_with_phase']

num_of_funcs_to_save = 4

dict_list = [create_dict_for_x_high_res(simulation_time, num_of_samples)]

for func_i in range(num_of_funcs_to_save):
    print(f"\nFunction number {func_i + 1} of {num_of_funcs_to_save}")
    type_list = random_pop(all_type_list)
    signal1 = SignalClass(simulation_time, freqs=freqs, factor=factor)
    freq = signal1.create_signal(type_list, op=lambda a, b: a + b)
    x_high_res, y_high_res = signal1.get_high_res_vec()

    if interpolation_method == 'CubicSpline':
        best_slots_interp = np.nan
        best_slots_CubicSpline = get_best_slots(x_high_res, y_high_res, number_of_slots, simulation_time, fps, 'CubicSpline')
        best_slots_SecondOrder = np.nan
        unif_mse = get_mse_from_interpolation(x_high_res, y_high_res, unif_slots, 'CubicSpline')
    if interpolation_method == 'interp':
        best_slots_interp = get_best_slots(x_high_res, y_high_res, number_of_slots, simulation_time, fps, 'interp')
        best_slots_CubicSpline = np.nan
        best_slots_SecondOrder = np.nan
        unif_mse = get_mse_from_interpolation(x_high_res, y_high_res, unif_slots, 'interp')
    if interpolation_method == 'SecondOrder':
        best_slots_interp = np.nan
        best_slots_CubicSpline = np.nan
        best_slots_SecondOrder = get_best_slots(x_high_res, y_high_res, number_of_slots, simulation_time, fps, 'SecondOrder')
        unif_mse = get_mse_from_interpolation(x_high_res, y_high_res, unif_slots, 'SecondOrder')

    dict_list.append(create_dict(type_list, simulation_time, freq, factor, y_high_res, unif_mse, best_slots_interp, best_slots_SecondOrder, best_slots_CubicSpline))
    if (func_i+1) % 1 == 0:
        np.save((PATH + '_temp.npy'), dict_list)
        print("Saves in:", (PATH + '_temp.npy'), '\n\n')
