from matplotlib import pyplot as plt
import os
import sys

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from signal_class.signal_generator import *
from signal_class.signal_interpolation import *
from signal_class.signal_sampler import *
from signal_class.signal_error import *
from DQN.DQN_main import RLModel


class SignalClass():

    def __init__(self, simulation_time, freqs, factor=2, name="Signal "):
        self.simulation_time = simulation_time
        self.freqs = freqs
        self.factor = factor
        self.name = name
        self.name_sampeld = name
        self.name_interpolation = name
        self.x_high_res = None
        self.y_high_res = None

    def create_signal(self, type_list, op=lambda a, b: a + b):
        self.x_high_res, self.y_high_res, freq = get_fun(self.simulation_time, self.freqs, type_list,
                                                         factor=self.factor,
                                                         op=op)
        self.name += "with types: "
        for type in type_list:
            self.name += (type + ", ")
        self.name = self.name[:-2]
        return freq

    def create_signal_from_xy(self, x_high_res, y_high_res):
        self.x_high_res, self.y_high_res = x_high_res, y_high_res
        self.name += "from vectors"

    def sample_signal(self, fps, sampling_method, N_B=5, N_C=5, op=lambda t: t, NUS_parameters=None):
        self.name_sampeld = self.name
        sam = Sampler(self.simulation_time, fps)
        if sampling_method == "uniform":
            x_sampled, y_sampled = sam.get_uniform_sampling_vector(self.x_high_res, self.y_high_res)
            self.name_sampeld += (" " + sampling_method + " sampling")
        elif sampling_method == "random":
            x_sampled, y_sampled = sam.get_random_sampling_vector(self.x_high_res, self.y_high_res)
            self.name_sampeld += (" " + sampling_method + " sampling")
        elif sampling_method == "boosting":
            x_sampled, y_sampled = sam.get_boosting_sampling_vector(self.x_high_res, self.y_high_res, N_boosting=N_B)
            self.name_sampeld += (" " + sampling_method + " sampling")
        elif sampling_method == "lambda":
            x_sampled, y_sampled = sam.get_lambda_sampling_vector(self.x_high_res, self.y_high_res, op)
            self.name_sampeld += (" " + sampling_method + " sampling")
        elif sampling_method == "chebyshev":
            x_sampled, y_sampled = sam.get_chebyshev_sampling_vector(self.x_high_res, self.y_high_res, N_chebyshev=N_C)
            self.name_sampeld += (" " + sampling_method + " sampling")
        elif sampling_method == "DQN":
            rl_model = RLModel(parameters=NUS_parameters)
            x_sampled, y_sampled = rl_model.get_DQN_sampling_vector(fps, self, interpolation_method='interp')
            self.name_sampeld += (" " + sampling_method + " sampling")
        return x_sampled, y_sampled

    def get_interpolation_vec(self, x_sampled, y_sampled, interpolation_method):
        self.name_interpolation = self.name + " interpolated, method: " + interpolation_method
        y_interp = interpolate_samples(self.x_high_res, x_sampled, y_sampled, interpolation_method=interpolation_method)
        return self.x_high_res, y_interp

    def get_interpolation_vec_with_sampling(self, interpolation_method, fps, sampling_method, N_B=5, N_C=5,
                                            op=lambda t: t):
        self.name_interpolation = self.name + " interpolated, method: " + interpolation_method
        x_sampled, y_sampled = self.get_sampled_vec(fps, sampling_method, N_B=N_B, N_C=N_C, op=op)
        y_interp = interpolate_samples(self.x_high_res, x_sampled, y_sampled, interpolation_method=interpolation_method)
        return self.x_high_res, y_interp

    def get_high_res_vec(self):
        return self.x_high_res, self.y_high_res

    def get_sampled_vec(self, fps, sampling_method, N_B=5, N_C=5, op=lambda t: t, NUS_parameters=None):
        x_sampled, y_sampled = self.sample_signal(fps, sampling_method, N_B=N_B, N_C=N_C, op=op,
                                                  NUS_parameters=NUS_parameters)
        return x_sampled, y_sampled

    def integrate_descrete_interval(self, x1, x2):
        xloc = lambda x: np.argmin(np.abs(self.x_high_res - x))
        # return self.y_high_res[ xloc( (x1+x2)/2 ) ]
        return self.y_high_res[xloc(x1):xloc(x2) + 1].mean()

    # input is an interped version of y (len(y) = len(high res)) and 2 indexes of start and end index in the high ress signal to compare
    # comparison_method chose error func
    def get_comparison_error(self, interped_y, min_y_index=0, max_y_index=np.inf, comparison_method='l1'):
        min_y_index = np.clip(min_y_index, 0, len(self.y_high_res))
        max_y_index = -1 if (max_y_index == np.inf) else max_y_index

        if comparison_method == 'l1' or comparison_method == 'L1':
            return get_onenorm_error(self.y_high_res[min_y_index:max_y_index], interped_y[min_y_index:max_y_index])
        elif comparison_method == 'l2' or comparison_method == 'L2':
            return get_mean_square_error(self.y_high_res[min_y_index:max_y_index], interped_y[min_y_index:max_y_index])
        elif comparison_method == 'lmax' or comparison_method == 'linf':
            return get_max_error(self.y_high_res[min_y_index:max_y_index], interped_y[min_y_index:max_y_index])

    def show_high_res(self, fig_num=None, block=True):
        if fig_num == None:
            plt.figure(figsize=(15, 5))
        else:
            plt.figure(fig_num, figsize=(15, 5))

        plt.plot(self.x_high_res, np.real(self.y_high_res))
        plt.title(self.name)
        plt.xlabel('Time [s]'), plt.ylabel('Amplitude')
        plt.show(block=block)
        return

    def show_samples(self, fps, sampling_methods, N_B=5, N_C=5, op=lambda t: t, scatter=False, fig_num=None,
                     block=True):
        if not (isinstance(sampling_methods, list)): sampling_methods = [sampling_methods]
        fig, axs = plt.subplots(len(sampling_methods) + 1)
        if fig_num == None:
            plt.figure(figsize=(15, 5))
        else:
            plt.figure(fig_num, figsize=(15, 5))

        axs[0].plot(self.x_high_res, np.real(self.y_high_res))
        axs[0].set_title(self.name)
        axs[0].set_xlabel('Time [s]'), axs[0].set_ylabel('Amplitude')
        for i, method in enumerate(sampling_methods):
            x_sampled, y_sampled = self.get_sampled_vec(fps, method, N_B=N_B, N_C=N_C, op=op)
            if scatter:
                axs[i + 1].scatter(x_sampled, np.real(y_sampled))
            else:
                axs[i + 1].plot(x_sampled, np.real(y_sampled), marker='.')
            axs[i + 1].set_title(self.name_sampeld)
            axs[i + 1].set_xlabel('Time [s]'), axs[i + 1].set_ylabel('Amplitude')
        plt.show(block=block)
        return

    def show_interpolations(self, interpolation_methods, fps, sampling_method, N_B=5, N_C=5, op=lambda t: t,
                            fig_num=None, block=True):
        fig, axs = plt.subplots(len(interpolation_methods) + 2)
        if fig_num == None:
            plt.figure(figsize=(15, 5))
        else:
            plt.figure(fig_num, figsize=(15, 5))

        axs[0].plot(self.x_high_res, np.real(self.y_high_res))
        axs[0].set_title(self.name)
        axs[0].set_xlabel('Time [s]'), axs[0].set_ylabel('Amplitude')
        x_sampled, y_sampled = self.get_sampled_vec(fps, sampling_method, N_B=N_B, N_C=N_C, op=op)
        axs[1].step(x_sampled, np.real(y_sampled), marker='.', where='post')
        axs[1].set_title(self.name_sampeld)
        axs[1].set_xlabel('Time [s]'), axs[0].set_ylabel('Amplitude')
        for i, method in enumerate(interpolation_methods):
            x_interp, y_interp = self.get_interpolation_vec(x_sampled, y_sampled, method)
            axs[i + 2].plot(x_interp, np.real(y_interp))
            axs[i + 2].set_title(self.name_interpolation)
            axs[i + 2].set_xlabel('Time [s]'), axs[i + 1].set_ylabel('Amplitude')
        plt.show(block=block)
        return

    def show_signal_with_samples(self, interpolation_methods, fps, sampling_method, N_B=5, N_C=5, op=lambda t: t,
                                 fig_num=None, block=True):
        if fig_num == None:
            plt.figure(figsize=(15, 5))
        else:
            plt.figure(fig_num, figsize=(15, 5))

        plt.title(self.name + ". " + sampling_method + " sampled. " + str(interpolation_methods) + " interpolation")
        plt.plot(self.x_high_res, np.real(self.y_high_res), label="Original signal")
        x_sampled, y_sampled = self.get_sampled_vec(fps, sampling_method, N_B=N_B, N_C=N_C, op=op)

        if isinstance(interpolation_methods, list):
            for method in interpolation_methods:
                x_interp, y_interp = self.get_interpolation_vec(x_sampled, y_sampled, method)
                plt.plot(x_interp, np.real(y_interp), label="Interpolation method: " + method)
        else:
            x_interp, y_interp = self.get_interpolation_vec(x_sampled, y_sampled, interpolation_methods)
            plt.plot(x_interp, np.real(y_interp), label="Interpolation method: " + interpolation_methods)

        plt.legend()
        plt.xlabel('Time [s]'), plt.ylabel('Amplitude')
        plt.show(block=block)
        return

    def save_high_res(self, name=None, fig_num=None):
        if name is None:
            name = self.name
        if fig_num != None:
            plt.figure(fig_num)

        plt.plot(self.x_high_res, np.real(self.y_high_res))
        plt.title(self.name)
        plt.xlabel('Time [s]'), plt.ylabel('Amplitude')
        plt.savefig(name + ".png")
        plt.close()
        return

    def show_signal_with_samples_interpolation(self, interpolation_method, fps, sampling_methods, N_B=5, N_C=5,
                                               op=lambda t: t
                                               , xlim=None, fig_num=None, block=True):
        if fig_num == None:
            plt.figure(figsize=(15, 5))
        else:
            plt.figure(fig_num, figsize=(15, 5))

        for method in sampling_methods:
            x_sampled, y_sampled = self.get_sampled_vec(fps, method, N_B=N_B, N_C=N_C, op=op)
            plt.scatter(x_sampled, np.real(y_sampled))
            x_interp, y_interp = self.get_interpolation_vec(x_sampled, y_sampled, interpolation_method)
            plt.plot(x_interp, np.real(y_interp), label="Sampling method: " + method)

        plt.title(self.name + ", with interpolation_method: " + interpolation_method)
        plt.plot(self.x_high_res, np.real(self.y_high_res), label="Original signal")
        plt.legend()
        plt.xlabel('Time [s]'), plt.ylabel('Amplitude')
        if xlim is not None:
            plt.xlim(*xlim)
        plt.show(block=block)
        return

    def save_signal_with_samples_interpolation(self, interpolation_method, fps, sampling_methods, N_B=5, N_C=5,
                                               op=lambda t: t
                                               , xlim=None, name=None):
        if name is None:
            name = self.name

        plt.figure(figsize=(15, 5))

        for method in sampling_methods:
            x_sampled, y_sampled = self.get_sampled_vec(fps, method, N_B=N_B, N_C=N_C, op=op)
            plt.scatter(x_sampled, np.real(y_sampled))
            x_interp, y_interp = self.get_interpolation_vec(x_sampled, y_sampled, interpolation_method)
            plt.plot(x_interp, np.real(y_interp), label="Sampling method: " + method)

        plt.title(self.name + ", with interpolation_method: " + interpolation_method)
        plt.plot(self.x_high_res, np.real(self.y_high_res), label="Original signal")
        plt.legend(loc='lower right')
        plt.xlabel('Time [s]'), plt.ylabel('Amplitude')
        if xlim is not None:
            plt.xlim(*xlim)
        plt.savefig(name + ".png")
        plt.close()
        return

    def show_signal_with_interpolations(self, interpolation_methods, fps, sampling_method, N_B=5, N_C=5, op=lambda t: t,
                                        fig_num=None, block=True):
        if fig_num == None:
            plt.figure(figsize=(15, 5))
        else:
            plt.figure(fig_num, figsize=(15, 5))

        plt.title(self.name + ". " + sampling_method + " sampled. " + str(interpolation_methods) + " interpolation")
        plt.plot(self.x_high_res, np.real(self.y_high_res), label="Original signal")
        x_sampled, y_sampled = self.get_sampled_vec(fps, sampling_method, N_B=N_B, N_C=N_C, op=op)

        if isinstance(interpolation_methods, list):
            for method in interpolation_methods:
                x_interp, y_interp = self.get_interpolation_vec(x_sampled, y_sampled, method)
                plt.plot(x_interp, np.real(y_interp), label="Interpolation method: " + method)
        else:
            x_interp, y_interp = self.get_interpolation_vec(x_sampled, y_sampled, interpolation_methods)
            plt.plot(x_interp, np.real(y_interp), label="Interpolation method: " + interpolation_methods)

        plt.legend()
        plt.xlabel('Time [s]'), plt.ylabel('Amplitude')
        plt.show(block=block)
        return

    def save_signal_with_interpolations(self, interpolation_methods, fps, sampling_method, N_B=5, N_C=5, op=lambda t: t,
                                        NUS_parameters=None,
                                        name=None):
        if name is None:
            name = self.name

        plt.figure(figsize=(15, 5))

        plt.title(self.name + ". " + sampling_method + " sampled. " + str(interpolation_methods) + " interpolation")
        plt.plot(self.x_high_res, np.real(self.y_high_res), label="Original signal")
        x_sampled, y_sampled = self.get_sampled_vec(fps, sampling_method, N_B=N_B, N_C=N_C, op=op,
                                                    NUS_parameters=NUS_parameters)
        if isinstance(interpolation_methods, list):
            for method in interpolation_methods:
                x_interp, y_interp = self.get_interpolation_vec(x_sampled, y_sampled, method)
                plt.plot(x_interp, np.real(y_interp), label="Interpolation method: " + method)
        else:
            x_interp, y_interp = self.get_interpolation_vec(x_sampled, y_sampled, interpolation_methods)
            plt.plot(x_interp, np.real(y_interp), label="Interpolation method: " + interpolation_methods)

        plt.legend()
        plt.xlabel('Time [s]'), plt.ylabel('Amplitude')
        plt.savefig(name + ".png")
        plt.close()
        return

    def save_signal_with_interpolations_from_x_y_sampled(self, interpolation_methods, sampling_name, x_sampled, y_sampled, name=None):
        if name is None:
            name = self.name

        plt.figure(figsize=(15, 5))

        plt.title(self.name + ". " + sampling_name + " sampled. " + str(interpolation_methods) + " interpolation")
        plt.plot(self.x_high_res, np.real(self.y_high_res), label="Original signal")

        if isinstance(interpolation_methods, list):
            for method in interpolation_methods:
                x_interp, y_interp = self.get_interpolation_vec(x_sampled, y_sampled, method)
                plt.plot(x_interp, np.real(y_interp), label="Interpolation method: " + method)
        else:
            x_interp, y_interp = self.get_interpolation_vec(x_sampled, y_sampled, interpolation_methods)
            plt.plot(x_interp, np.real(y_interp), label="Interpolation method: " + interpolation_methods)

        plt.legend()
        plt.xlabel('Time [s]'), plt.ylabel('Amplitude')
        plt.savefig(name + ".png")
        plt.close()
        return

    def print_error(self, error_type, y_interpolated, text=""):
        if not (isinstance(error_type, list)): error_type = [error_type]

        for type in error_type:

            if type == "max" or type == "max_error":
                print(text, "Max error: ", '{:.2e}'.format(get_max_error(self.y_high_res, y_interpolated)))
            elif type == "one norm" or type == "onenorm" or type == "onenorm_error":
                print(text, "One norm error: ", '{:.2e}'.format(get_onenorm_error(self.y_high_res, y_interpolated)))
            elif type == "mean square" or type == "meansquare" or type == "mean_square_error":
                print(text, "Mean square error: ",
                      '{:.2e}'.format(get_mean_square_error(self.y_high_res, y_interpolated)))
