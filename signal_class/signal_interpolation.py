import numpy as np
from scipy import interpolate
from scipy.interpolate import barycentric_interpolate, KroghInterpolator, CubicSpline, lagrange


# return y_high_res interpolated from the y_vec with the given method
def interpolate_samples(x_high_res, x_vec, y_vec, interpolation_method='interp'):  
    if interpolation_method == 'lagrange':
        return lagrange(x_vec, y_vec)( x_high_res )
    if interpolation_method == 'CubicSpline':
        return CubicSpline(x_vec, y_vec)( x_high_res )
    if interpolation_method == 'zeroorder':
        return interpolate.interp1d(x_vec, y_vec, kind='zero', axis=0)( x_high_res )
    if interpolation_method == 'secondorder':
        return interpolate.interp1d(x_vec, y_vec, kind='quadratic', bounds_error=False, fill_value=(y_vec[0], y_vec[-1]))(x_high_res)
    return np.interp(x_high_res, x_vec, y_vec)


def interpolate_chebyshev_method(self, x, y, N, true_time_vec, simulation_time, T):
        time_vec_res = true_time_vec[1] - true_time_vec[0]
        total_interpulation = []
        for i in range( int( simulation_time/(N*T)) ):
            total_interpulation += list( lagrange(x[i*N:(i+1)*N], y[i*N:(i+1)*N])( \
                true_time_vec[int( round(i*T*N/time_vec_res) ) : \
                                   int( round( (i+1)**N/time_vec_res) ) ]) )
        derivative_upper_limit = 1.0
        print('Estimation error:',np.abs(derivative_upper_limit) / ( np.math.factorial(N) * (2**(N-1)) ))
        return np.array( total_interpulation )