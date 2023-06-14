import numpy as np

# https://netlib.org/lapack/lug/node75.html  some errors options


# input is 2 same size arrays 
# mean square error element wise
def get_mean_square_error(real_y, comper_y):
    if comper_y.shape[0] == 0:
        return 0
    mse = (np.square(real_y - comper_y)).mean()
    return mse


# input is 2 same size arrays 
# max error element wise
def get_max_error(real_y, comper_y):
    max_error = np.amax(np.abs(real_y - comper_y))
    return max_error


# input is 2 same size arrays        #L1
# L1
def get_onenorm_error(real_y, comper_y):
    onenorm = (np.abs(real_y - comper_y)).mean()
    return onenorm


# input is 2 same size arrays 
# calculate mse for the arrays and the mse of the arrays with out the last and first elements
# keep cutting the last and first element untill the mse is bigger then with out the cut
# returns the mse of the small leanth array and the ammunt cut from each side.  (mse , cut from start, cut from end)
# that way hopefuly lagrange edge errors will be ditected and removed
def get_no_edge_minsquer_error(real_y, comper_y):
    mse = get_mean_square_error(real_y, comper_y)
    mse_next = get_mean_square_error(real_y[:-1], comper_y[:-1])
    i, j = 0, 0
    while mse > mse_next:
        i += 1
        mse = mse_next
        mse_next = get_mean_square_error(real_y[:-i - 1], comper_y[:-i - 1])

    # now mse is the mse of [:-i]

    mse_next = get_mean_square_error(real_y[1:-i], comper_y[1:-i])
    while mse > mse_next:
        j += 1
        mse = mse_next
        mse_next = get_mean_square_error(real_y[1 + j:-i], comper_y[1 + j:-i])

    return mse, j, i


# input is 2 same size arrays
# calculate mse for the arrays and the mse of the arrays with out the last and first elements
# keep cutting the last and first element untill the mse is bigger then with out the cut
# returns the mse of the small leanth array and the ammunt cut from each side.  (mse , cut from start, cut from end)
#
# we keep preforming the cutting recursivly untill the mse is smaller then the cup or if we got an empty signal
# for an empty signal we return i,j = -1
# that way hopefuly lagrange edge errors will be ditected and removed get_caped_no_edge_mean_square_error
def get_caped_no_edge_mean_square_error(real_y, comper_y, cup=2):
    if 0 == real_y.shape[0]:  # empty signal is bad
        return 6969, -1, -1
    mse, j, i = get_no_edge_minsquer_error(real_y, comper_y)

    if i + j >= real_y.shape[0]:  # empty signal is bad
        return 6969, -1, -1

    if mse > cup:
        return get_caped_no_edge_mean_square_error(real_y[j:-i], comper_y[j:-i])

    return mse, j, i


def get_cooler_mse(real_y, comper_y, factor=0.2):
    amount_to_cut = int(0.5 * factor * real_y.shape[0])
    return get_mean_square_error(real_y[amount_to_cut:-amount_to_cut], comper_y[amount_to_cut:-amount_to_cut])


def get_mse_remove_edges_by_frac(real_y, comper_y, factor=0.2):
    amount_to_cut = int(0.5 * factor * real_y.shape[0])
    return get_mean_square_error(real_y[amount_to_cut:-amount_to_cut], comper_y[amount_to_cut:-amount_to_cut])
