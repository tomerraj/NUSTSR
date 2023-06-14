import copy
import os
import sys
import matplotlib.pyplot as plt

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from supervised.model_train import *
from supervised.make_data_supervized import *
from supervised.model_load import *


def simple_test(parameters, interpolation_method, test_percent=0.2, real_time_graph=0):
    """
    this function will take a parameters of a model, initialize it then train if on the given data set in DATA_set_path

    right now we will start by trusting the number of epochs that are in parameters


    we return a vector of the losses and correct percent
     and the trained model
    """
    path = parameters["path_to_save"] + '/' + parameters["MODEL_NAME"] + '/'
    if os.path.exists(path) and parameters["MODEL_NAME"][:4] != "temp":
        raise Exception(f'The Model Name  {parameters["MODEL_NAME"]}  already exists!')

    if parameters['features'] == -1:
        parameters['features'] = 2 + (1 if parameters['feature_dict'].get('derivative') == 1 else 0) + \
                                 (2 if parameters['feature_dict'].get(
                                     'FFT') == 1 else 0)  # WARNING this code is in 2 places

    model_in_test = get_model(parameters)
    print("model_in_test:", model_in_test)

    train_mat, test_mat = call_get_training_data(parameters, test_percent=test_percent)

    fig1, ax1 = plt.subplots()
    line, = ax1.plot([], [])
    update_loss_n = 2

    accumulated_losses_array = []
    avg_loss_array = []
    correct_prob_array = []
    correct_prob_test_array = []
    x_values_test = []
    device = get_device()

    avg_loss, correct_prob = test(parameters, test_mat, model_in_test, device=device)
    avg_loss_test, correct_prob_test = test(parameters, train_mat, model_in_test, device=device)
    avg_loss_array.append(avg_loss)
    correct_prob_array.append(correct_prob)
    correct_prob_test_array.append(correct_prob_test)
    x_values_test.append(0)

    for i in range(0, parameters['n_epochs'], parameters['evaluation_interval']):
        model_in_test, accumulated_losses = train(parameters, train_mat, model_in_test, update_loss_n=update_loss_n,
                                                  start_epoch=i, device=device)
        accumulated_losses_array += accumulated_losses  # this is for graphing the accumulated loss over time

        if real_time_graph == 1:
            line.set_data(range(len(accumulated_losses_array)), accumulated_losses_array)
            ax1.relim()
            ax1.autoscale_view(True, True, True)
            ax1.set_xlabel(f'{update_loss_n} Epochs')
            ax1.set_ylabel('Accumulated average Loss')
            fig1.canvas.draw()
            fig1.canvas.flush_events()
            plt.pause(0.001)

        avg_loss, correct_prob = test(parameters, test_mat, model_in_test, device=device)
        avg_loss_test, correct_prob_test = test(parameters, train_mat, model_in_test, device=device)
        avg_loss_array.append(avg_loss)
        x_values_test.append(i + parameters['evaluation_interval'])
        correct_prob_array.append(correct_prob)
        correct_prob_test_array.append(correct_prob_test)

        # can add here a sample_with_model check ass well

    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + f'{parameters["MODEL_NAME"]}_parameters.json', 'w') as fp:
        print('\nSaved Model', path + parameters["MODEL_NAME"], '\n')
        parameters_to_save = copy.deepcopy(parameters)
        json.dump(parameters_to_save, fp)

    fig11, ax11 = plt.subplots()
    # last graph any way for save
    ax11.plot(range(update_loss_n, len(accumulated_losses_array) * update_loss_n + 1, update_loss_n),
              accumulated_losses_array, label='Accumulated average Loss')
    ax11.set_xlabel(f'Epoch')
    ax11.set_ylabel('Accumulated average Loss')
    ax11.set_title(f'Accumulated average Loss per Epoch')
    ax11.legend()
    fig11.savefig(path + f'Accumulated_Loss_per_Epoch_for_{parameters["MODEL_NAME"]}.png')

    fig2, ax21 = plt.subplots()

    # avg_loss_array_np = avg_loss_array.cpu().numpy()
    # plot y1 on ax1
    avg_loss_list = np.array([tensor.cpu().item() for tensor in avg_loss_array], dtype=float)
    ax21.scatter(x_values_test, avg_loss_list, color='b', label='Avg Loss')
    ax21.set_xlabel('Epoch')
    ax21.set_ylabel('y1', color='b')
    ax21.tick_params('y', colors='b')

    # create a second y-axis
    ax22 = ax21.twinx()

    # plot y2 on ax22
    correct_prob_list = [tensor.item() for tensor in correct_prob_array]
    correct_prob_test_list = [tensor.item() for tensor in correct_prob_test_array]
    ax22.scatter(x_values_test, correct_prob_list, color='r', label='Correct Prob')
    ax22.set_ylabel('y2', color='r')
    ax22.tick_params('y', colors='r')

    lines, labels = ax21.get_legend_handles_labels()
    lines2, labels2 = ax22.get_legend_handles_labels()
    ax22.legend(lines + lines2, labels + labels2, loc='upper right')

    if real_time_graph == 1:
        fig2.show()
        plt.pause(0.001)
    fig2.savefig(path + f'Avg_Loss_and_Correct_Prob_for_{parameters["MODEL_NAME"]}_on_test_mat.png')

    fig3, ax3 = plt.subplots()
    ax3.plot(range(update_loss_n, len(accumulated_losses_array) * update_loss_n + 1, update_loss_n),
             accumulated_losses_array, 'b-', label='Training Loss')
    ax3.scatter(x_values_test, avg_loss_list, color='r', label='Testing Loss')
    ax3.set_xlabel(f'Epoch')
    ax3.set_ylabel('Accumulated average Loss per Epoch')
    ax3.set_title(f'Accumulated average Loss on train vs test')
    ax3.legend()
    fig3.savefig(path + f'Accumulated_Loss_per_Epoch_TNT_for_{parameters["MODEL_NAME"]}.png')

    fig4, ax4 = plt.subplots()
    ax4.scatter(x_values_test, correct_prob_list, color='r', label='correct prob test')
    ax4.scatter(x_values_test, correct_prob_test_list, color='b', label='correct prob train')
    ax4.set_xlabel(f'Epoch')
    ax4.set_ylabel('correct prob per Epoch')
    ax4.set_title(f'correct prob train vs test')
    ax4.legend()
    fig4.savefig(path + f'Accumulated_test_vs_train_prob_{parameters["MODEL_NAME"]}.png')

    plt.cla()

    resaults_vector = []
    avg_loss, correct_prob = test(parameters, train_mat, model_in_test)
    resaults_vector += [avg_loss.item(), correct_prob.item()]
    avg_loss, correct_prob = test(parameters, test_mat, model_in_test)
    resaults_vector += [avg_loss.item(), correct_prob.item()]

    # Add results from sample_with_model to results vector
    dicts = np.load(parameters['data_path'], allow_pickle=True)
    x_high_res_dict = dicts[0]
    x_high_res = x_high_res_dict["x_high_res"]

    del train_mat
    del test_mat

    mse_mat = []

    dicts = dicts[1:]
    dicts = dicts[parameters['start_from_func']:] if parameters['end_in_func'] == -1 else dicts[
                                                                                          parameters['start_from_func']:
                                                                                          parameters['end_in_func']]

    split_index = int(len(dicts) * (1 - test_percent))
    training_funcs = dicts[:split_index]
    test_funcs = dicts[split_index:]

    for func in test_funcs:
        y_high_res = func["y_high_res"]
        unif_mse = func["unif_mse"]
        x_sampled, y_sampled = sample_with_model(model_in_test, parameters, x_high_res, y_high_res, device)
        y_interp = interpolate_samples(x_high_res, x_sampled, y_sampled, interpolation_method)
        rnn_mse = get_mean_square_error(y_high_res, y_interp)
        mse_mat.append([rnn_mse, unif_mse])

    mse_mat = np.array(mse_mat)
    mse_avg = np.mean(mse_mat, axis=0)
    ratio_improve = np.divide(mse_mat[:, 1], mse_mat[:, 0])
    ratio_improve_avg = np.mean(ratio_improve)
    resaults_vector.extend([mse_avg[0], mse_avg[1], ratio_improve_avg])

    mse_mat = []

    for func in training_funcs:
        y_high_res = func["y_high_res"]
        unif_mse = func["unif_mse"]
        x_sampled, y_sampled = sample_with_model(model_in_test, parameters, x_high_res, y_high_res, device)
        y_interp = interpolate_samples(x_high_res, x_sampled, y_sampled, interpolation_method)
        rnn_mse = get_mean_square_error(y_high_res, y_interp)
        mse_mat.append([rnn_mse, unif_mse])

    mse_mat = np.array(mse_mat)
    mse_avg = np.mean(mse_mat, axis=0)
    ratio_improve = np.divide(mse_mat[:, 1], mse_mat[:, 0])
    ratio_improve_avg = np.mean(ratio_improve)
    resaults_vector.extend([mse_avg[0], mse_avg[1], ratio_improve_avg])

    return np.array(resaults_vector)


def complex_test(parameters, interpolation_method, DATA_path_array):
    """
    takes in dict parameters of the model that we want to train. a list of data paths that we want to run a test on.
    it duplicates the parameters dict with the correct path and name change and runs the functions in parallel.

    for now, we will assume that the DATA_path_array is ordered such that the order is:
    low_freq_ez, high_freq_ez, low_&_high_freq_ez, low_and_high_mid, low_and_high_hard
    """
    if parameters['features'] == -1:
        parameters['features'] = 2 + (1 if parameters['feature_dict'].get('derivative') == 1 else 0) + \
                                 (2 if parameters['feature_dict'].get(
                                     'FFT') == 1 else 0)  # WARNING this code is in 2 places

    test_names = [(i.split('/')[-1])[10:-4] for i in DATA_path_array]
    dup_param_dict = {}
    result_vectors = []
    for i, data_path in enumerate(DATA_path_array):
        # Create a copy of the parameters dictionary and update the relevant fields
        dup_dict = copy.deepcopy(parameters)
        dup_dict["MODEL_NAME"] = dup_dict["MODEL_NAME"] + '_' + test_names[i]
        dup_dict["data_path"] = data_path
        # maybe also add device as param

        dup_param_dict[test_names[i]] = dup_dict

        result_vectors.append(simple_test(dup_dict, interpolation_method, test_percent=0.2, real_time_graph=0))

    return np.array(result_vectors)


"""
the matrix is 2D
first D is the difrent models we trained 
second dim is allways size 7 witch are the following:
train_mat avg_loss
train_mat correct prob
test_mat avg_loss
test_mat correct prob
mse_avg from sampel on all the data base
mse_avg from unif
ratio_improve_avg
"""
