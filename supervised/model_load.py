import json
import os
import sys

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from supervised.model_architectures import *


def load_model(path):
    model_name = path.split('/')[-2]
    with open(path + f'{model_name}_parameters.json', 'r') as fp:
        parameters = json.load(fp)

    model = get_model(parameters)
    model.load_state_dict(torch.load(path + model_name))
    model.eval()

    return model
