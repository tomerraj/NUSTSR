import os
import sys
from tqdm import tqdm

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from signal_class.signal_error import *
from signal_class.signal_interpolation import *
from signal_class.utils import *


def get_frame_boundaries_by_index(frame_index, x_high_res, simulation_time, fps):
    number_of_frames = simulation_time * fps
    frame_length = len(x_high_res) / number_of_frames

    x_left = frame_length * frame_index
    x_right = x_left + frame_length - 1

    return x_high_res[round(x_left)], x_high_res[round(x_right)]


def get_slot_in_frame_by_index_by_index(slot_index, frame_index, number_of_slots, x_high_res, simulation_time, fps):

    number_of_frames = simulation_time * fps
    frame_length = len(x_high_res) / number_of_frames
    slot_length = frame_length / number_of_slots

    x_slot = frame_length * frame_index + slot_index * slot_length

    return x_high_res[round(x_slot)]


def sample_function(x_high_res, y_high_res, x_slots):
    mask = np.searchsorted(x_high_res, x_slots)
    return y_high_res[mask]


def translate_from_absolute_time_to_index(x_slots, frame_index, number_of_slots, fps):
    x_norm = x_slots[frame_index] - (frame_index / fps)
    return round(x_norm * fps * number_of_slots)


def translate_from_absolute_time_to_index_of_vector(x_slots, number_of_slots, fps):
    frame_indexes = (np.arange(len(x_slots))) / fps
    x_norm = x_slots - frame_indexes
    return np.around(x_norm * fps * number_of_slots)


def get_unif_vec(x_high_res, number_of_slots, simulation_time, fps):
    """
    returns the uniform x coordinates (selects the 5th slot in each frame)
    :param x_high_res:
    :param simulation_time:
    :param fps:
    :return:
    """
    x_unif = np.array([get_slot_in_frame_by_index_by_index(0, frame_index, number_of_slots, x_high_res, simulation_time, fps) for frame_index in range(simulation_time * fps)])
    return x_unif


def get_mse_from_interpolation(x_high_res, y_high_res, x_slots, interpolation_method):
    y_interp = interpolate_samples(x_high_res, x_slots, sample_function(x_high_res, y_high_res, x_slots), interpolation_method)
    return get_mean_square_error(y_high_res, y_interp)


def get_mse_from_interpolation_cropped(x_high_res, y_high_res, x_slots, main_frame_index, simulation_time, fps, interpolation_method, min=20, max=20):
    frame_length = len(x_high_res) / (simulation_time * fps)
    upper_high_res = int(np.ceil((main_frame_index + max)*frame_length))
    lower_high_res = int(np.ceil((main_frame_index - 1 - min)*frame_length))
    upper_slots = int(np.ceil((main_frame_index + max+1)))
    lower_slots = int(np.ceil((main_frame_index - min)))

    if main_frame_index <= min:
        y_slots = sample_function(x_high_res, y_high_res, x_slots[0:upper_slots])
        y_interp = interpolate_samples(x_high_res[0:upper_high_res], x_slots[0:upper_slots], y_slots, interpolation_method)
        return get_mean_square_error(y_high_res[0:upper_high_res], y_interp)

    # The upper boundaries can go beyond the length of the array
    y_slots = sample_function(x_high_res, y_high_res, x_slots[lower_slots:upper_slots])
    y_interp = interpolate_samples(x_high_res[lower_high_res:upper_high_res], x_slots[lower_slots:upper_slots], y_slots, interpolation_method)
    return get_mean_square_error(y_high_res[lower_high_res:upper_high_res], y_interp)


def change_slot_by_index_in_x_slot(x_slot, frame_index, new_slot_index, number_of_slots, x_high_res, simulation_time, fps):
    x_slot_copy = np.copy(x_slot)
    x_slot_copy[frame_index] = get_slot_in_frame_by_index_by_index(new_slot_index, frame_index, number_of_slots, x_high_res, simulation_time, fps)
    return x_slot_copy


def get_best_slots(x_high_res, y_high_res, number_of_slots, simulation_time, fps, interpolation_method):

    number_of_frames = simulation_time * fps


    # from here: look up to 4 points -------

    x_slot_index = []
    x_slots = get_unif_vec(x_high_res, number_of_slots, simulation_time, fps)
    unif_mse = get_mse_from_interpolation(x_high_res, y_high_res, x_slots, interpolation_method)

    print(f'unif mse: {unif_mse}')

    for frame in tqdm(range(1, number_of_frames - 4)):

        min_mse = get_mse_from_interpolation_cropped(x_high_res, y_high_res, x_slots, frame, simulation_time, fps, interpolation_method)
        # min_mse = get_mse_from_interpolation(x_high_res, y_high_res, x_slots, interpolation_method)

        min_slot = 0

        for i3 in range(number_of_slots):
            x_slots = change_slot_by_index_in_x_slot(x_slots, frame + 3, i3, number_of_slots, x_high_res, simulation_time, fps)
            for i2 in range(number_of_slots):
                x_slots = change_slot_by_index_in_x_slot(x_slots, frame + 2, i2, number_of_slots, x_high_res, simulation_time, fps)
                for i1 in range(number_of_slots):
                    x_slots = change_slot_by_index_in_x_slot(x_slots, frame + 1, i1, number_of_slots, x_high_res, simulation_time, fps)
                    for i in range(number_of_slots):
                        x_slots = change_slot_by_index_in_x_slot(x_slots, frame, i, number_of_slots, x_high_res, simulation_time, fps)

                        new_mse = get_mse_from_interpolation_cropped(x_high_res, y_high_res, x_slots, frame, simulation_time, fps, interpolation_method)
                        # new_mse = get_mse_from_interpolation(x_high_res, y_high_res, x_slots, interpolation_method)
                        if new_mse < min_mse:
                            min_mse = new_mse
                            min_slot = i

        # print('min_slot:', min_slot, '  min_mse:', min_mse, ' ; frame:', frame)
        x_slots = change_slot_by_index_in_x_slot(x_slots, frame, min_slot, number_of_slots, x_high_res, simulation_time, fps)
        x_slots = change_slot_by_index_in_x_slot(x_slots, frame + 1, 0, number_of_slots, x_high_res, simulation_time, fps)
        x_slots = change_slot_by_index_in_x_slot(x_slots, frame + 2, 0, number_of_slots, x_high_res, simulation_time, fps)
        x_slots = change_slot_by_index_in_x_slot(x_slots, frame + 3, 0, number_of_slots, x_high_res, simulation_time, fps)

        x_slot_index.append(min_slot)

    print("min mse (all):", get_mse_from_interpolation(x_high_res, y_high_res, x_slots, interpolation_method))
    print("x_slot_index 4:", x_slot_index, '\n')

    return x_slots  # comment this line if you want to change to the other loop


    # until here: look up to 4 points -------


    # from here: look up to 5 points -------


    x_slot_index = []
    x_slots = get_unif_vec(x_high_res, number_of_slots, simulation_time, fps)
    unif_mse = get_mse_from_interpolation(x_high_res, y_high_res, x_slots, interpolation_method)

    print(f'\nFive points ahead 55555\nunif mse: {unif_mse}')

    for frame in tqdm(range(1, number_of_frames - 4)):
    # for frame in tqdm(range(15)):

        min_mse = get_mse_from_interpolation_cropped(x_high_res, y_high_res, x_slots, frame, simulation_time, fps, interpolation_method)
        min_slot = 0

        for i4 in range(number_of_slots):
            x_slots = change_slot_by_index_in_x_slot(x_slots, frame + 4, i4, number_of_slots, x_high_res, simulation_time, fps)
            for i3 in range(number_of_slots):
                x_slots = change_slot_by_index_in_x_slot(x_slots, frame + 3, i3, number_of_slots, x_high_res, simulation_time, fps)
                for i2 in range(number_of_slots):
                    x_slots = change_slot_by_index_in_x_slot(x_slots, frame + 2, i2, number_of_slots, x_high_res, simulation_time, fps)
                    for i1 in range(number_of_slots):
                        x_slots = change_slot_by_index_in_x_slot(x_slots, frame + 1, i1, number_of_slots, x_high_res, simulation_time, fps)
                        for i in range(number_of_slots):
                            x_slots = change_slot_by_index_in_x_slot(x_slots, frame, i, number_of_slots, x_high_res, simulation_time, fps)
                            new_mse = get_mse_from_interpolation_cropped(x_high_res, y_high_res, x_slots, frame, simulation_time, fps, interpolation_method)

                            if new_mse < min_mse:
                                min_mse = new_mse
                                min_slot = i
        x_slot_index.append(min_slot)

        x_slots = change_slot_by_index_in_x_slot(x_slots, frame, min_slot, number_of_slots, x_high_res, simulation_time, fps)
        x_slots = change_slot_by_index_in_x_slot(x_slots, frame + 1, 0, number_of_slots, x_high_res, simulation_time, fps)
        x_slots = change_slot_by_index_in_x_slot(x_slots, frame + 2, 0, number_of_slots, x_high_res, simulation_time, fps)
        x_slots = change_slot_by_index_in_x_slot(x_slots, frame + 3, 0, number_of_slots, x_high_res, simulation_time, fps)
        x_slots = change_slot_by_index_in_x_slot(x_slots, frame + 4, 0, number_of_slots, x_high_res, simulation_time, fps)

    print("min mse (all):", get_mse_from_interpolation(x_high_res, y_high_res, x_slots, interpolation_method))
    print("x_slot_index 5:", x_slot_index, '\n\n')

    return x_slots


    # until here: look up to 5 points -------



    #
    # x_slot_index = []
    # x_slots = get_unif_vec(x_high_res, number_of_slots, simulation_time, fps)
    #
    # min_mse = get_mse_from_interpolation(x_high_res, y_high_res, x_slots, interpolation_method)
    #
    # # x_slots1 = change_slot_by_index_in_x_slot(x_slots, 110, 0, number_of_slots, x_high_res, simulation_time, fps)
    # # x_slots2 = change_slot_by_index_in_x_slot(x_slots, 110, 1, number_of_slots, x_high_res, simulation_time, fps)
    # #
    # # y_interp1 = interpolate_samples(x_high_res, x_slots1, sample_function(x_high_res, y_high_res, x_slots1), 'CubicSpline')
    # # y_interp2 = interpolate_samples(x_high_res, x_slots2, sample_function(x_high_res, y_high_res, x_slots2), 'CubicSpline')
    # #
    # # print(np.where((y_interp1 - y_interp2) != 0)[0])
    # # print(len(np.where((y_interp1 - y_interp2) != 0)[0]))
    #
    # print("original mse:", min_mse, '\n')
    #
    # # for frame in tqdm(range(1, number_of_frames - 4)):
    # for frame in tqdm(range(6)):
    #
    #     min_mse = get_mse_from_interpolation(x_high_res, y_high_res, x_slots, interpolation_method)
    #     min_slot = 0
    #
    #     for i3 in range(number_of_slots):
    #         x_slots = change_slot_by_index_in_x_slot(x_slots, frame + 3, i3, number_of_slots, x_high_res, simulation_time, fps)
    #         for i2 in range(number_of_slots):
    #             x_slots = change_slot_by_index_in_x_slot(x_slots, frame + 2, i2, number_of_slots, x_high_res, simulation_time, fps)
    #             for i1 in range(number_of_slots):
    #                 x_slots = change_slot_by_index_in_x_slot(x_slots, frame + 1, i1, number_of_slots, x_high_res, simulation_time, fps)
    #                 for i in range(number_of_slots):
    #                     x_slots = change_slot_by_index_in_x_slot(x_slots, frame, i, number_of_slots, x_high_res, simulation_time, fps)
    #                     new_mse = get_mse_from_interpolation(x_high_res, y_high_res, x_slots, interpolation_method)
    #
    #                     if new_mse < min_mse:
    #                         min_mse = new_mse
    #                         min_slot = i
    #
    #     x_slots = change_slot_by_index_in_x_slot(x_slots, frame, min_slot, number_of_slots, x_high_res, simulation_time, fps)
    #     x_slots = change_slot_by_index_in_x_slot(x_slots, frame + 1, 0, number_of_slots, x_high_res, simulation_time, fps)
    #     x_slots = change_slot_by_index_in_x_slot(x_slots, frame + 2, 0, number_of_slots, x_high_res, simulation_time, fps)
    #     x_slots = change_slot_by_index_in_x_slot(x_slots, frame + 3, 0, number_of_slots, x_high_res, simulation_time, fps)
    #
    #     x_slot_index.append(min_slot)
    #
    # print()
    # print("min mse:", min_mse)
    # print("x_slot_index:", x_slot_index, '\n\n')
    #
    # print()
    # print()
    # print(f'min mse cropped by the fast x_slot: {get_mse_from_interpolation_cropped(x_high_res, y_high_res, x_slots, 5, simulation_time, fps, interpolation_method)}')
    # print("x_slots_fast - x_slots: ", x_slots_fast - x_slots, '\n')
    #
    # return x_slots
    #
    # x_slot_index = []
    # x_slots = get_unif_vec(x_high_res, number_of_slots, simulation_time, fps)
    #
    # n = 4
    # from math import fmod, pow
    # for frame in tqdm(range(6)):
    #     min_mse = get_mse_from_interpolation_cropped(x_high_res, y_high_res, x_slots, 0, simulation_time, fps, interpolation_method)
    #     min_slot = 0
    #
    #     for i in range(0, number_of_slots ** n):
    #         for j in range(n, 0, -1):
    #             x_slots = change_slot_by_index_in_x_slot(x_slots, frame + j, (i//10**j) % 10, number_of_slots, x_high_res, simulation_time, fps)
    #
    #         x_slots = change_slot_by_index_in_x_slot(x_slots, frame, i % 10, number_of_slots, x_high_res, simulation_time, fps)
    #         new_mse = get_mse_from_interpolation_cropped(x_high_res, y_high_res, x_slots, frame, simulation_time, fps, interpolation_method)
    #
    #         if new_mse < min_mse:
    #             min_mse = new_mse
    #             min_slot = i % 10
    #
    #     x_slots = change_slot_by_index_in_x_slot(x_slots, frame, min_slot, number_of_slots, x_high_res, simulation_time, fps)
    #     x_slots = change_slot_by_index_in_x_slot(x_slots, frame + 1, 0, number_of_slots, x_high_res, simulation_time, fps)
    #     x_slots = change_slot_by_index_in_x_slot(x_slots, frame + 2, 0, number_of_slots, x_high_res, simulation_time, fps)
    #     x_slots = change_slot_by_index_in_x_slot(x_slots, frame + 3, 0, number_of_slots, x_high_res, simulation_time, fps)
    #     x_slot_index.append(min_slot)
    #
    #
    # # n - how much we look ahead
    # # look [behind 2 ,haead n+1] range of mse and interp
    # # 30000 high res vector
    # # 7 sec function and 30 fps
    # print("\n min mse:", min_mse)
    # print("x_slot_index:", x_slot_index, '\n\n')
    #
    # return x_slots
