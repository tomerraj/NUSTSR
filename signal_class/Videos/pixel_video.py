from signal_class.signal_generator import HIGH_RES_SAMPLES
from signal_class.utils import get_rnd_pixel_interpolated_from_video_by_loc, dump_all_intensities_of_all_pixels_in_video
import matplotlib.pyplot as plt
import numpy as np

num_of_samples = HIGH_RES_SAMPLES
simulation_time = 7
x_high_res = np.linspace(0.0, simulation_time, HIGH_RES_SAMPLES)


video_path = "signal_class/Videos/Treadmill_Running_500fps.avi"
npy_path = "signal_class/Videos/Treadmill_Running_500fps.npy"


dump_all_intensities_of_all_pixels_in_video(video_path, npy_path)

# y_high_res = get_rnd_pixel_interpolated_from_video_by_loc(video_path, npy_path, 'up', x_high_res, simulation_time, interpolation_method='secondorder', sigma=150)
# y_high_res = get_rnd_pixel_interpolated_from_video_by_loc(video_path, npy_path, 'up', x_high_res, interpolation_method='secondorder')

# mean_y = np.mean(y_high_res)
# # Subtract the mean from each element in y_high_res
# y_high_res = y_high_res - mean_y

# plt.plot(x_high_res, y_high_res)
# plt.xlabel('Time (sec)')
# plt.ylabel('Intensity')
# plt.title('Pixel Intensity over Time')
# plt.show()
