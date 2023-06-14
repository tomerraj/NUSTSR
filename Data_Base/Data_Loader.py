import os
import sys

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from signal_class.signal_class import *
from Data_Base.Best_Slots import *

fps = 30
number_of_slots = 10

dicts = np.load('Data_Base/Data_Base_interp_both_freq_oscillating_chirp_with_phase_1-10.npy', allow_pickle=True)
# dicts = np.load('Data_Base/Data_Base_interp_pixel_up_GOT_60fps.npy', allow_pickle=True)
# dicts = np.load('Data_Base/Data_Base_CubicSpline_both_freq_oscillating_chirp_with_phase_1-10.npy', allow_pickle=True)

#  the first dict is different
#  it is a dict of x_high_res

x_high_res_dict = dicts[0]
x_high_res = x_high_res_dict["x_high_res"]

dicts = dicts[1:]

unif_slots = get_unif_vec(x_high_res, number_of_slots, x_high_res_dict["simulation_time"], fps)

for i, dict in enumerate(dicts):
    print("function:", i + 1, " of:", len(dicts))
    print("freq: ", dict["freq"])
    print("types: ", dict["type_list"])
    signal1 = SignalClass(dict["simulation_time"], freqs=(dict["freq"], dict["freq"]), factor=dict["factor"])
    signal1.create_signal_from_xy(x_high_res, dict["y_high_res"])

    best_slots_interp = dict["best_slots_interp"]
    best_slots_CubicSpline = dict["best_slots_CubicSpline"]

    if np.isnan(best_slots_interp).all():
        interpolation_method = 'CubicSpline'
        print("Interpolation method: CubicSpline")
        y_interp = interpolate_samples(x_high_res, best_slots_CubicSpline,
                                       sample_function(x_high_res, dict["y_high_res"], best_slots_CubicSpline),
                                       "CubicSpline")
        unif_slots_mse = get_mse_from_interpolation(x_high_res, dict["y_high_res"], unif_slots, 'CubicSpline')
        best_Slots_mse = get_mean_square_error(dict["y_high_res"], y_interp)
        print(f'\tunif_slots_mse: {unif_slots_mse} \n\tbest_Slots_mse: {best_Slots_mse} \n')

    if np.isnan(best_slots_CubicSpline).all():
        interpolation_method = 'interp'
        print("Interpolation method: interp")
        y_interp = interpolate_samples(x_high_res, best_slots_interp,
                                       sample_function(x_high_res, dict["y_high_res"], best_slots_interp), "interp")
        unif_slots_mse = get_mse_from_interpolation(x_high_res, dict["y_high_res"], unif_slots, 'interp')
        best_Slots_mse = get_mean_square_error(dict["y_high_res"], y_interp)
        print(f'\tunif_slots_mse: {unif_slots_mse} \n\tbest_Slots_mse: {best_Slots_mse} \n')

    plt.plot(x_high_res, y_interp, label="best")
    plt.plot(x_high_res,
             interpolate_samples(x_high_res, unif_slots, sample_function(x_high_res, dict["y_high_res"], unif_slots),
                                 interpolation_method), label="unif")
    plt.legend()
    signal1.show_high_res(fig_num=1)
