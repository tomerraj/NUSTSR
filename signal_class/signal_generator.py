import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

from signal_class.utils import get_rnd_pixel_interpolated_from_video_by_loc

HIGH_RES_SAMPLES = 50_000


def moving_average_on_vec(vec, order=3):
    kernel = np.ones(order).astype(np.float32) / order
    return np.convolve(vec, kernel, 'same')


# returns a frequency between min and max. and returns a amplitud factor between 1/factor to factor
def rnd_arg(fmin, fmax, factor=2):
    return (fmin + np.random.rand() * (fmax - fmin)), (
            (0 if factor == 0 else 1 / factor) + np.random.rand() * (factor - (0 if factor == 0 else 1 / factor)))


def get_fun_modulation(simulation_time, fmin, fmax, factor=2, type_list=['sin']):
    funcs = []

    # for the 'pixel_up' , 'pixel_down' signals
    video_path = "signal_class/Videos/Treadmill_Running_500fps.avi"
    npy_path = "signal_class/Videos/Treadmill_Running_500fps.npy"
    interpolation_method = 'secondorder'
    sigma = 100

    for type in type_list:
        freq_rnd, amp_rnd = rnd_arg(fmin, fmax, factor)
        if type == 'linear':
            funcs.append(lambda t, freq=freq_rnd, amp=amp_rnd: amp * t)
        if type == 'square':
            funcs.append(lambda t, freq=freq_rnd, amp=amp_rnd: amp * t * t)
        if type == 'oscillating_chirp':  # move between freq of +-3 of the freq
            funcs.append(lambda t, freq=freq_rnd, amp=amp_rnd: amp * np.sin(
                3 * np.sin((t * freq / 10) * 2 * np.pi) + freq * 2 * np.pi * t))
        if type == 'oscillating_chirp_with_phase':  # move between freq of +-3 of the freq
            funcs.append(lambda t, freq=freq_rnd, amp=amp_rnd: amp * np.sin(
                3 * np.sin((t * freq / 10) * 2 * np.pi) + 2 * np.pi * (freq * t + np.random.rand())))
        if type == 'sin':
            funcs.append(lambda t, freq=freq_rnd, amp=amp_rnd: amp * np.sin(freq * 2 * np.pi * t))
        if type == 'sin_with_phase':
            funcs.append(lambda t, freq=freq_rnd, amp=amp_rnd: amp * np.sin(2 * np.pi * (freq * t + np.random.rand())))
        if type == 'cos':
            funcs.append(lambda t, freq=freq_rnd, amp=amp_rnd: amp * np.cos(freq * 2 * np.pi * t))
        if type == 'chirp_with_amplitude':
            funcs.append(lambda t, freq=freq_rnd, amp=amp_rnd: 2 * amp * np.exp(-t / (0.25 * simulation_time)) * np.sin(
                0.05 * freq * 2 * np.pi * t * t))
        if type == 'square_wave':
            funcs.append(lambda t, freq=freq_rnd, amp=amp_rnd: np.where(np.cos(freq * 2 * np.pi * t) >= 0, amp, 0))
        if type == 'square_times_sin':
            funcs.append(
                lambda t, freq=freq_rnd, amp=amp_rnd: np.where(np.sin(freq * 2 * np.pi * t) >= 0, amp, 0) * np.sin(
                    freq * t))
        if type == 'square_changing_height':
            funcs.append(lambda t, freq=freq_rnd, amp=amp_rnd: np.where(np.sin(freq * 2 * np.pi * t) >= 0, amp, 0) *
                                                               np.resize(np.repeat(
                                                                   np.random.uniform(amp - 0.45, amp + 0.45,
                                                                                     size=int(t[-1] * freq)),
                                                                   len(t) // int(t[-1] * freq)), len(t)))
        if type == 'triangular_wave':
            funcs.append(lambda t, freq=freq_rnd, amp=amp_rnd: amp * (2 * np.abs(t - np.floor(t + 0.5)) - 1))
        if type == 'triangular_spike':
            funcs.append(lambda t, freq=freq_rnd, amp=amp_rnd: np.where(np.mod(np.floor(t * freq), 2) == 0,
                                                                        amp * (2 * np.abs(t - np.floor(t + 0.5))), 0))
        if type == 'sawtooth':
            funcs.append(lambda t, freq=freq_rnd, amp=amp_rnd: signal.sawtooth(2 * np.pi * freq * t))
        if type == 'sawtooth_with_phase':
            funcs.append(
                lambda t, freq=freq_rnd, amp=amp_rnd: signal.sawtooth(2 * np.pi * freq * (t + np.random.rand())))
        if type == 'chirp':
            funcs.append(lambda t, freq=freq_rnd, amp=amp_rnd: amp * np.sin(0.05 * freq * 2 * np.pi * t * t))
        if type == 'double_chirp':
            funcs.append(lambda t, freq=freq_rnd, amp=amp_rnd: amp * np.sin(freq * 2 * np.pi * t * t * t))
        if type == 'periodic_frequency':
            funcs.append(
                lambda t, freq=freq_rnd, amp=amp_rnd: amp * np.sin(freq * np.sin(0.01 * 2 * np.pi * t) * 2 * np.pi * t))
        if type == 'square_periodic':
            funcs.append(lambda t, freq=freq_rnd, amp=amp_rnd: amp * signal.square(freq * 2 * np.pi * t))
        if type == 'smooth_square_periodic':
            funcs.append(
                lambda t, freq=freq_rnd, amp=amp_rnd: amp * gaussian_filter1d(signal.square(freq * 2 * np.pi * t),
                                                                              sigma=50))
        if type == 'sawtooth_periodic':
            funcs.append(lambda t, freq=freq_rnd, amp=amp_rnd: amp * signal.sawtooth(freq * 2 * np.pi * t))
        if type == 'pisewise_linear_periodic':
            x_samples = np.linspace(0.0, simulation_time, int((simulation_time * freq_rnd)) + 2)
            y_samples = np.random.randn(x_samples.shape[0]) * 0.5
            funcs.append(
                lambda t, amp=amp_rnd: amp * interp1d(x_samples, y_samples, kind='linear', fill_value='extrapolate')(t))
        if type == 'pisewise_linear_aperiodic':
            x_samples = np.linspace(0.0, simulation_time, int((simulation_time * freq_rnd)) + 2)
            x_samples += np.random.randn(x_samples.shape[0]) * 0.5 / freq_rnd
            y_samples = np.random.randn(x_samples.shape[0])
            funcs.append(lambda t, freq=freq_rnd, amp=amp_rnd: amp * interp1d(x_samples, y_samples, kind='linear',
                                                                              fill_value='extrapolate')(t))
        if type == 'sqrt_sin':
            funcs.append(lambda t, freq=freq_rnd, amp=amp_rnd: amp * freq * np.sin(freq * 2 * np.pi * np.sqrt(t)))
        if type == 'log_sin':
            funcs.append(lambda t, freq=freq_rnd, amp=amp_rnd: amp * freq * np.sin(freq * 2 * np.pi * np.log(t + 1)))
        if type == 'exp_sin':
            funcs.append(lambda t, freq=freq_rnd, amp=amp_rnd: amp * freq * np.sin(freq * 2 * np.pi * np.exp(t) / 5))
        if type == 'pixel_up':
            funcs.append(lambda t: get_rnd_pixel_interpolated_from_video_by_loc(video_path, npy_path, 'up', t,
                                                                                simulation_time=simulation_time,
                                                                                interpolation_method=interpolation_method,
                                                                                sigma=sigma))
        if type == 'pixel_down':
            funcs.append(lambda t: get_rnd_pixel_interpolated_from_video_by_loc(video_path, npy_path, 'down', t,
                                                                                simulation_time=simulation_time,
                                                                                interpolation_method=interpolation_method,
                                                                                sigma=sigma))
        if type.startswith('rand_with_moving_avg_order_'):
            order = int(type[len('rand_with_moving_avg_order_'):])
            funcs.append(lambda t, amp=amp_rnd: 0.5 * order * amp_rnd * moving_average_on_vec(np.random.randn(t.shape[0]), order=order))
        if type.startswith('rand_with_gaussian_sigma_'):
            sigma = int(type[len('rand_with_gaussian_sigma_'):])
            funcs.append(lambda t, amp=amp_rnd: 0.5 * sigma * amp_rnd * gaussian_filter1d(np.random.randn(t.shape[0]), sigma))
    return freq_rnd, funcs


def get_fun(simulation_time, freqs, type_list, op=lambda a, b: a + b, factor=2):
    freq, funcs = get_fun_modulation(simulation_time, fmin=freqs[0], fmax=freqs[1], factor=factor, type_list=type_list)
    x_high_res = np.linspace(0.0, simulation_time, HIGH_RES_SAMPLES)
    y_high_res = funcs[0](x_high_res)
    if len(funcs) == 1:
        return x_high_res, y_high_res, freq

    for func in funcs[1:]:
        y_tmp = func(x_high_res)
        y_high_res = op(y_high_res, y_tmp)

    return x_high_res, y_high_res, freq
