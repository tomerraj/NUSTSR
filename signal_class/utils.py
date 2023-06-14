from random import randrange, sample
import numpy as np
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from signal_class.signal_interpolation import interpolate_samples


# remove a given amount of items from the given list. if no amount was given then arandom number is chosen.
def random_pop(list, amount=-1):
    if len(list) == 1:
        return list
    if amount == -1:
        amount = randrange(len(list))
    to_delete = set(sample(range(len(list)), amount))

    return [
        item for index, item in enumerate(list)
        if not index in to_delete
    ]


def print_error_matt(error_mat, interpolation_methods, sampling_methods):
    print(interpolation_methods)

    print(error_mat)
    print("avgage:\n", np.mean(error_mat, axis=0))

    print("best preforming interp methood: ", interpolation_methods[np.argmin(np.mean(error_mat, axis=0))])
    print("best preforming sampling methood: ", sampling_methods[np.argmin(np.mean(error_mat, axis=1))])


def dump_all_intensities_of_all_pixels_in_video(video_path, path_to_save):
    """video_path: Path to the video file
       output_file: Path to save the intensities array as .npy"""

    video = VideoFileClip(video_path)
    duration = video.duration
    height, width = video.size
    num_frames = int(duration * video.fps)

    intensities = np.zeros((num_frames, width, height), dtype=np.float32)

    for t in tqdm(range(num_frames)):
        frame = video.get_frame(t / video.fps)

        if frame.ndim == 3:  # Color frame
            frame_gray = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
            intensities[t] = frame_gray
        else:  # Grayscale frame
            intensities[t] = frame

    np.save(path_to_save, intensities)


def get_pixel_from_npy_by_pos(npy_array_path, pixel_pos):
    """npy_array_path: Path to the .npy file containing the 3D array of intensities
       pixel_pos: (X, Y) coordinates of the pixel"""

    # Load the intensities array from the .npy file
    intensities = np.load(npy_array_path)

    # Get the coordinates of the pixel
    x, y = pixel_pos

    # Retrieve the pixel intensity at the specified position
    pixel_intensity = intensities[:, y, x]

    return pixel_intensity


def get_rnd_pixel_interpolated_from_video_by_loc(video_path, npy_array_path, loc, x_high_res, simulation_time=None,
                                                 interpolation_method='interp', sigma=130):
    """npy_array_path: Path to the .npy file containing the 3D array of intensities
       loc as of what half from the video to take the pixel from:
       loc='up' or loc='down' """

    video = VideoFileClip(video_path)
    # print(f'video dim: ({video.size[0]}, {video.size[1]})')
    if simulation_time is None:
        simulation_time = video.duration

    # Determine the y-coordinate range based on the location
    height = video.size[1]
    if loc == 'up':
        y_range = (int(height * 0.15), int(height * 0.5))
    elif loc == 'down':
        y_range = (int(height * 0.5), int(height * 0.85))
    else:
        print("Invalid location. Choose 'up' or 'down'.")
        return None

    # Randomly select a pixel within the specified location
    x = np.random.randint(0, video.size[0])
    y = np.random.randint(*y_range)
    # x, y = 585, 151
    # print(f'chosen pixel:({x}, {y})')
    # Get the pixel intensity over time
    intensity = get_pixel_from_npy_by_pos(npy_array_path, (x, y))
    # print("length of intensity array:", len(intensity))
    # Interpolate the intensity to fit x_high_res

    x_vec = np.linspace(0, simulation_time, len(intensity))
    # print("video.duration:", video.duration)
    y_high_res = interpolate_samples(x_high_res, x_vec, intensity, interpolation_method)

    y_high_res = gaussian_filter1d(y_high_res, sigma)

    return y_high_res
