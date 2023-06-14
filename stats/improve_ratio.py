from Data_Base.Best_Slots import get_mse_from_interpolation_cropped
from supervised.model_train import sample_with_model, get_device
from supervised.model_load import load_model
from signal_class.utils import *
from Data_Base.Best_Slots import get_unif_vec


def main():
    simulation_time = 7
    fps = 30
    number_of_slots = 10

    interpolation_method = "interp"  # interp, CubicSpline,  SecondOrder

    supervised_parameters = {
        'model_path': 'supervised/models4/to_poster_sin_22-5-23/interp/n_epochs-400_LSTM_dropout-0.5_hidden-zeros/to_poster_sin_22-5-23_interp_both_freq_sin_with_phase_10-20/',
        'simulation_time': simulation_time,
        'fps': fps,
        'number_of_slots': number_of_slots,  # upsampling_factor from DQN dict
        'feature_dict': {'derivative': 0, 'FFT': 0},
        'state_sample_length': 20
    }
    model = load_model(supervised_parameters['model_path'])

    data_path = 'Data_Base/Data_Base_interp_both_freq_sin_with_phase_10-20.npy'




    dicts = np.load(data_path, allow_pickle=True)
    x_high_res_dict = dicts[0]
    x_high_res = x_high_res_dict["x_high_res"]

    dicts = dicts[1:]  # remove x vec
    device = get_device()
    right_choice_vector = []
    wrong_choice_vector = []
    correct_prob_vector = []
    wrong_choice_vector_diff = [[] for _ in range(10)]

    x_time_unif = get_unif_vec(x_high_res, number_of_slots, simulation_time, fps)
    func_amount = len(dicts)
    for index, func in enumerate(dicts):

        if (index+1)*4 // func_amount > index*4 // func_amount:
            print((index+1)*4 // func_amount, "/4 way there at ", index)# stamped lines can delite if want

        y_high_res = func["y_high_res"]
        # unif_mse = func["unif_mse"]

        x_sampled, y_sampled = sample_with_model(model, supervised_parameters, x_high_res, y_high_res, device)
        # x_index_model = translate_from_absolute_time_to_index_of_vector(x_sampled, number_of_slots, fps)

        if interpolation_method == 'interp':
            x_time_opt = func["best_slots_interp"]

        if interpolation_method == 'CubicSpline':
            x_time_opt = func["best_slots_CubicSpline"]

        if interpolation_method == 'SecondOrder':
            x_time_opt = func["best_slots_SecondOrder"]

        # x_index_opt = translate_from_absolute_time_to_index_of_vector(x_time_opt, number_of_slots, fps)
        # x_time_unif = get_slot_in_frame_by_index_by_index(slot_index, frame_index, number_of_slots, x_high_res, simulation_time, fps)
        right_choice_array = []
        wrong_choice_array = []
        wrong_choice_array_diff = [[] for _ in range(10)]

        for i in range(supervised_parameters['state_sample_length'], fps*simulation_time):
            # running on all the points the model made a choice

            x_index_model_change = x_time_opt.copy()
            x_index_model_change[i] = x_sampled[i]
            x_index_unif_change = x_time_opt.copy()
            x_index_unif_change[i] = x_time_unif[i]
            opt_mse = get_mse_from_interpolation_cropped(x_high_res, y_high_res, x_time_opt, i, simulation_time, fps, interpolation_method, min=1, max=1)

            # if i < 40:
            #     # print(f"x_sampled[{i}]:", x_sampled[i])
            #     # print(f"x_time_unif[{i}]:", x_time_unif[i])
            #     # print(f"x_time_opt[{i}]:", x_time_opt[i])
            #     print("======", abs(x_sampled[i] - x_time_opt[i]) <= 0.001)
            if abs(x_sampled[i] - x_time_opt[i]) <= 0.001:  # model chose right
                unif_mse = get_mse_from_interpolation_cropped(x_high_res, y_high_res, x_index_unif_change, i,
                                                              simulation_time, fps, interpolation_method, min=1, max=1)
                mse_unif_ratio = unif_mse / opt_mse
                mse_model_ratio = 1
                right_choice_array.append(mse_unif_ratio - mse_model_ratio)  # add the difference in ratios. if positive then unif is worse

            else:  # model chose wrong
                unif_mse = get_mse_from_interpolation_cropped(x_high_res, y_high_res, x_index_unif_change, i,
                                                              simulation_time, fps, interpolation_method, min=1, max=1)
                model_mse = get_mse_from_interpolation_cropped(x_high_res, y_high_res,x_index_model_change, i,
                                                               simulation_time, fps, interpolation_method,min=1, max=1)

                mse_unif_ratio = unif_mse / opt_mse
                mse_model_ratio = model_mse / opt_mse
                wrong_choice_array.append(mse_unif_ratio - mse_model_ratio)  # add the difference in ratios. if positive then unif is worse

                index_diff = abs(round((x_sampled[i] - x_time_opt[i]) * fps * number_of_slots))
                wrong_choice_array_diff[index_diff].append(mse_unif_ratio - mse_model_ratio)


        correct_prob = len(right_choice_array) / (len(right_choice_array) + len(wrong_choice_array))

        # print(len(right_choice_array))
        # print(len(wrong_choice_array))

        avg_mse_improve_right = sum(right_choice_array) / len(right_choice_array) if len(right_choice_array) > 0 else 0
        avg_mse_improve_wrong = sum(wrong_choice_array) / len(wrong_choice_array) if len(wrong_choice_array) > 0 else 0
        for i,row in enumerate(wrong_choice_array_diff):
            row_average = sum(row) / len(row) if len(row) > 0 else 0
            wrong_choice_vector_diff[i].append(row_average)

        right_choice_vector.append(avg_mse_improve_right)
        wrong_choice_vector.append(avg_mse_improve_wrong)
        correct_prob_vector.append(correct_prob)

    avg_mse_improve_right_all_func = sum(right_choice_vector) / len(right_choice_vector)
    avg_mse_improve_wrong_all_func = sum(wrong_choice_vector) / len(wrong_choice_vector)

    avg_mse_improve_wrong_all_func_diff = [[] for _ in range(10)]
    for i,row in enumerate(wrong_choice_vector_diff):
        row_average = sum(row) / len(row) if len(row) > 0 else 0
        avg_mse_improve_wrong_all_func_diff[i] = row_average

    name = supervised_parameters['model_path']
    print(f'\nusing model: {name} \nover {func_amount} functions')
    print("correct_prob_avg:   ",np.average(correct_prob_vector))
    print("improve_right_avg:  ",avg_mse_improve_right_all_func)
    print("improve_wrong_avg: ",avg_mse_improve_wrong_all_func)
    print("improve_wrong_avg by index: ",avg_mse_improve_wrong_all_func_diff)



if __name__ == '__main__':
    main()
