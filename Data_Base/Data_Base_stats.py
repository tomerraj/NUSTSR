import os
import sys

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from Data_Base.Best_Slots import *

fps = 30
number_of_slots = 10

PATHs = [
    'Data_Base_interp_both_freq_sin_with_phase_1-10.npy',
    'Data_Base_interp_both_freq_sin_with_phase_10-20.npy'
]


dicts_join = []
for path in PATHs:
    dicts = np.load(path, allow_pickle=True)

    x_high_res_dict = dicts[0]
    x_high_res = x_high_res_dict["x_high_res"]

    dicts_join.extend(dicts[1:])

unif_slots = get_unif_vec(x_high_res, number_of_slots, x_high_res_dict["simulation_time"], fps)

max_mse = float('-inf')  # initialize with a very small number
sum_mse = 0.0

for i, dict in enumerate(dicts_join):
    best_slots_interp = dict["best_slots_interp"]
    best_slots_CubicSpline = dict["best_slots_CubicSpline"]

    if np.isnan(best_slots_interp).all():
        interpolation_method = 'CubicSpline'
        y_interp = interpolate_samples(x_high_res, best_slots_CubicSpline, sample_function(x_high_res, dict["y_high_res"], best_slots_CubicSpline), "CubicSpline")
        unif_slots_mse = get_mse_from_interpolation(x_high_res, dict["y_high_res"], unif_slots, interpolation_method)
        best_Slots_mse = get_mean_square_error(dict["y_high_res"], y_interp)
        # mse_improvement_percent = (unif_slots_mse - best_Slots_mse) / unif_slots_mse * 100
        mse_improvement_percent = (unif_slots_mse - best_Slots_mse) / best_Slots_mse * 100
        # mse_improvement_percent = unif_slots_mse / best_Slots_mse * 100
        max_mse = max(max_mse, mse_improvement_percent)
        sum_mse += mse_improvement_percent
        # print(f'{i}\tunif_slots_mse: {unif_slots_mse} \n\tbest_Slots_mse: {best_Slots_mse} \n\tthe improvement: {mse_improvement_percent}%\n')

    if np.isnan(best_slots_CubicSpline).all():
        interpolation_method = 'interp'
        y_interp = interpolate_samples(x_high_res, best_slots_interp, sample_function(x_high_res, dict["y_high_res"], best_slots_interp), "interp")
        unif_slots_mse = get_mse_from_interpolation(x_high_res, dict["y_high_res"], unif_slots, interpolation_method)
        best_Slots_mse = get_mean_square_error(dict["y_high_res"], y_interp)
        # mse_improvement_percent = (unif_slots_mse - best_Slots_mse) / unif_slots_mse * 100
        mse_improvement_percent = (unif_slots_mse - best_Slots_mse) / best_Slots_mse * 100
        # mse_improvement_percent = unif_slots_mse / best_Slots_mse * 100
        max_mse = max(max_mse, mse_improvement_percent)
        sum_mse += mse_improvement_percent
        # print(f'{i}\tunif_slots_mse: {unif_slots_mse} \n\tbest_Slots_mse: {best_Slots_mse} \n\tthe improvement: {mse_improvement_percent}%\n')

avg_mse = sum_mse / len(dicts_join)

print(PATHs[0])
print("max_mse:", max_mse, '%')
print("avg_mse:", avg_mse, '%')
