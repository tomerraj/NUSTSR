import numpy as np


class Sampler():
    ''' This class down-sample a signal according to the sampling location vector '''
    def __init__(self, sample_time, fps):
        self.sample_time = sample_time
        self.period = 1/fps
        self.fps = fps

    def get_uniform_sampling_vector(self, x_high_res, y_high_res):
        x_down = (np.arange((x_high_res[-1] - x_high_res[0]) *self.fps) + 0.5) * self.period
        y_down = np.interp(x_down, x_high_res, y_high_res)
        return x_down, y_down

    def get_random_sampling_vector(self, x_high_res, y_high_res):
        x_uni, y_uni = self.get_uniform_sampling_vector(x_high_res, y_high_res)
        x_down = x_uni + (np.random.rand(np.shape(x_uni)[0])*0.9 - 0.45)*self.period
        y_down = np.interp(x_down, x_high_res, y_high_res)
        return x_down, y_down

    def get_boosting_sampling_vector(self, x_high_res, y_high_res, N_boosting=2):
        W_boosting = self.period / N_boosting
        delta = (self.period * (1 - W_boosting)) / (N_boosting-1)
        x_uni, y_uni = self.get_uniform_sampling_vector(x_high_res, y_high_res)
        x_down = x_uni - 0.5*self.period + ((N_boosting - (np.arange(len(x_uni)) % N_boosting))-1) * delta + W_boosting*self.period/2
        y_down = np.interp(x_down, x_high_res, y_high_res)
        return x_down, y_down

    def get_lambda_sampling_vector(self, x_high_res, y_high_res, func_t):
        x_down = (np.arange((x_high_res[-1] - x_high_res[0]) *self.fps) + 0.5) * self.period
        x_down = func_t(x_down)
        x_down = (x_down-x_down[0]) * (x_high_res[-1] - x_high_res[0])/(x_down[-1] - x_down[0])
        y_down = np.interp(x_down, x_high_res, y_high_res)
        return x_down, y_down

    def get_chebyshev_sampling_vector(self,x_high_res,y_high_res, N_chebyshev=4):
        # First get Chebyshev roots:
        cbs_roots = np.flip(np.cos((2 * (np.arange(0, N_chebyshev)) + 1) * np.pi / (2 * N_chebyshev)))
        # Then Map to the domain:
        time_list = []
        for i in range(int((x_high_res[-1]-x_high_res[0]) / (N_chebyshev * self.period))):
            time_list += list(i * N_chebyshev * self.period + (cbs_roots + 1) * self.period * N_chebyshev / 2)
        x_down = np.array(time_list)
        y_down = np.interp(x_down, x_high_res, y_high_res)
        return x_down, y_down    
