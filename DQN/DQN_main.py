import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import namedtuple
from itertools import count
import torch.optim as optim
import os
import sys

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from signal_class.signal_class import *
from signal_class.utils import *
from DQN.DQN_replay_memory import *


class NUSDQNNet(nn.Module):
    '''
    This Model choose the next sampling location
    '''

    # upsampling_factor is the amount of diffrent options the module has as an output
    # in our case its the amaunt we devide each frame to take the sample

    # state_sample_length the leanth of the finle x axis of the sampled signal
    # =fps*(sim_time)

    def __init__(self, state_sample_length, upsampling_factor=10):
        super(NUSDQNNet, self).__init__()
        self.fc1 = nn.Linear(2 * state_sample_length, state_sample_length*16)  # 2 becasue x axis and y axis are input. [x  y]
        self.fc2 = nn.Linear(state_sample_length*16, 300)
        self.fc3 = nn.Linear(300, 100)
        self.fc4 = nn.Linear(100, 50)
        self.fc5 = nn.Linear(50, 10)
        self.fc6 = nn.Linear(10, upsampling_factor, bias=True) # init the starting value of cnvs
        return

    def forward(self, state):
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = F.leaky_relu(self.fc5(x))
        x = self.fc6(x)
        return x


class RLModel():
    def __init__(self, parameters):
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        if torch.cuda.is_available():
            print("cuda is available!")
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.device = 'cpu'
        self.steps_done = 0
        self.PATH = parameters['PATH']  # path
        self.MODEL_NAME = parameters['MODEL_NAME']  # path
        self.EXP_START_DECAY = parameters['EXP_START_DECAY']  #
        self.EPS_START = parameters['EPS_START']  #
        self.EPS_END = parameters['EPS_END']  # epsilon like temp for 'eps_greedy' exploration_strategy
        self.upsampling_factor = parameters['upsampling_factor']  # amount of action options
        self.BATCH_SIZE = parameters['BATCH_SIZE']  # bathch size
        self.GAMMA = parameters['GAMMA']  # gamma: discount factor
        self.state_sample_length = parameters['state_sample_length']  # state_sample_length
        self.episode_durations = []  # for plot ??
        self.NUM_OF_FUNCTIONS = parameters['NUM_OF_FUNCTIONS']  #
        self.NUM_OF_EPISODES = parameters['NUM_OF_EPISODES']  # training loops
        self.TARGET_UPDATE = parameters['TARGET_UPDATE']  # num of episodes between update target_net
        self.initial_temperature = parameters['initial_temperature']  # temp
        self.exploration_strategy = parameters['exploration_strategy']  # = 'eps_greedy' or  'softmax'

        self.policy_net = NUSDQNNet(self.state_sample_length, self.upsampling_factor).to(self.device)
        self.target_net = NUSDQNNet(self.state_sample_length, self.upsampling_factor).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=0.002, momentum=0.2)
        self.memory = ReplayMemory(10000, self.Transition)

    def select_action(self, state):

        if self.exploration_strategy == 'eps_greedy':
            sample = random.random()
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
                -1. * self.steps_done / self.EXP_START_DECAY)
            self.steps_done += 1

            # as steps done increace there is a bigger chance to chose the best action
            # else chose action at random
            if sample > eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    return self.policy_net(state).max(1)[1].view(1, 1)
            else:
                return torch.tensor([[random.randrange(self.upsampling_factor)]], device=self.device, dtype=torch.long)

        elif self.exploration_strategy == 'softmax':
            # https://towardsdatascience.com/deep-q-network-with-pytorch-and-gym-to-solve-acrobot-game-d677836bda9b#9bf2
            # temperature = self.initial_temperature * np.log(2) / np.log( self.steps_done+2 )
            temperature = self.initial_temperature * np.exp(-(self.steps_done) / (self.EXP_START_DECAY))
            temperature = max(temperature, 1e-8)  # set a minimum to the temperature for numerical stability
            # print( self.policy_net(state), temperature )
            soft = nn.Softmax(dim=-1)
            # print( state, 'ss9')
            # print('state: ', state)
            # print('forward: ', self.policy_net(state), '\n')
            prob = soft(self.policy_net(state) / temperature)
            # print(prob,'s2')
            prob = prob.cpu().detach().numpy()[0]
            # print(prob,'s3')
            action = np.random.choice([i for i in range(self.upsampling_factor)], p=prob)
            # print('prob',self.policy_net(state), prob)
            # print('act', action)
            self.steps_done += 1

            return torch.tensor([[action]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = self.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.bool)
        # non_final_mask is a bool 1D tensor (batch size)  of if the next state is terminal

        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        # array of the element next_state of each batch element
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad() # hereeee
        loss.backward()
        for param in self.policy_net.parameters(): # hereeee
            param.grad.data.clamp_(-1, 1) # hereeee
        self.optimizer.step()

    #  return (reward, is_failed, new_cur_x vector , new_cur_y vector , net_interpolation_error)
    def get_reward_and_new_state(self, cur_x, cur_y, action, current_frame_n,
                                 SignalClass, camera_fps, max_n, interp_nn_model=None, interpolation_method='interp'):
        if max_n <= current_frame_n:
            return 0.0, True, None, None, None

        T = 1 / camera_fps
        high_res_low_res_ratio = round(T / (SignalClass.x_high_res[1] - SignalClass.x_high_res[0]))
        x1 = T * (current_frame_n + (
                float(action[0, 0]) / self.upsampling_factor))  # start and end of camera shater open in [sec]
        x2 = x1 + T / self.upsampling_factor

        # plt.plot( T * ( 1+ torch.arange( 9 ) + cur_x[0,1:].numpy() ), cur_y[0,1:].numpy(), 'o')

        new_cur_x = torch.cat((cur_x[:, 1:], action / self.upsampling_factor),
                              1).float()  # x[i] is the window index chosen for each frame
        new_cur_y = torch.cat((cur_y[:, 1:], torch.tensor( \
            SignalClass.integrate_descrete_interval(x1, x2)).unsqueeze(0).unsqueeze(0)), 1).float()

        new_cur_x_un_normalized = T * (
                current_frame_n + 1 - self.state_sample_length + torch.arange(new_cur_x.shape[1]) + new_cur_x)
        # the normalized vector is the acual time each shot was taken

        # print( cur_x, action )
        # print( new_cur_x_un_normalized );
        x, intepolated_y = SignalClass.get_interpolation_vec(new_cur_x_un_normalized.numpy()[0, :],
                                                             new_cur_y.numpy()[0, :], \
                                                             interpolation_method=interpolation_method)

        min_i = (current_frame_n - self.state_sample_length) * high_res_low_res_ratio
        max_i = current_frame_n * high_res_low_res_ratio
        # the start/end indexes of the points we interped correctly

        ### ---- Step of Optimization the NN interpolator ---- ###
        net_interpolation_error = None
        if interp_nn_model is None:
            interpolation_error = SignalClass.get_comparison_error(intepolated_y, min_y_index=min_i, max_y_index=max_i, \
                                                                   comparison_method='L2')


        reward = -float(interpolation_error)

        return reward, False, new_cur_x.float(), new_cur_y.float(), net_interpolation_error

    def training_model(self, signal_type_list, camera_fps, freqs, simulation_time, interpolation_method='interp'):
        # from Boosting.NUSClass import get_sampling_function, NUSSampler, NUSInterpolator

        T = 1 / camera_fps
        x2normedx = lambda x: (x % T)
        # x2normedx = lambda x: (x % T) * camera_fps
        normedx2x = lambda x: T * (torch.arange(x.shape[1]) + x)

        for i in range(self.NUM_OF_FUNCTIONS):
            print('Function number:', i + 1, 'of', self.NUM_OF_FUNCTIONS)

            # Define high resolution vector:
            # __, f_fun = get_fun_modulation(simulation_time, fmin, fmax, signal_type_list)
            # x_high_res, y_high_res = get_sampling_function(f_fun, simulation_time)

            signal_class = SignalClass(simulation_time, freqs=freqs)
            signal_class.create_signal(random_pop(signal_type_list), op=random.choice([ lambda a, b: a + np.cos(b), lambda a, b: a + b ]) )
            signal_class.save_high_res(name=self.PATH + f"/{self.MODEL_NAME}_functions/training_function_{i + 1}")
            # Samples (start with uniform, regular sampling):
            x_vector, y_vector = signal_class.sample_signal(camera_fps, "uniform")

            x_vector = torch.tensor(x_vector).unsqueeze(0).float()
            y_vector = torch.tensor(y_vector).unsqueeze(0).float()
            x_normed_vector = x2normedx(x_vector)

            reward_list = []
            for i_episode in range(self.NUM_OF_EPISODES):
                # Initialize the environment and state
                cur_x = x_normed_vector[:, :self.state_sample_length]
                cur_y = y_vector[:, :self.state_sample_length]
                state = torch.cat((cur_x, cur_y), 1)
                accumulated_reward = (0.0)
                for t in count():
                    t += self.state_sample_length
                    # Select and perform an action
                    action = self.select_action(state)
                    print('action: ',  action, f'(Function number:{i+1} of {self.NUM_OF_FUNCTIONS}. with episode: {i_episode} of {self.NUM_OF_EPISODES})')
                    reward, done, cur_x, cur_y, int_model_error = self.get_reward_and_new_state(cur_x, cur_y, action, t, \
                                                                                                signal_class,
                                                                                                camera_fps, \
                                                                                                x_normed_vector.shape[
                                                                                                    1], \
                                                                                                interpolation_method=interpolation_method)
                    reward = torch.tensor([reward], device=self.device)
                    # Observe new state
                    if not done:
                        next_state = torch.cat((cur_x, cur_y), 1)
                    else:
                        next_state = None

                    # Store the transition in memory
                    self.memory.push(state, action, next_state, reward)

                    # Move to the nextstate
                    state = next_state

                    # Perform one step of the optimization (on the policy network)
                    self.optimize_model()

                    accumulated_reward += reward  # reward for target net

                    if done:
                        self.episode_durations.append(t + 1)
                        # plot_durations()
                        reward_list += [accumulated_reward]
                        accumulated_reward = 0.0

                        break

                # Update the target network, copying all weights and biases in DQN
                if i_episode % self.TARGET_UPDATE == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                # runtime_clock()
            # print('Complete')

            torch.save(self.target_net.state_dict(), self.PATH + self.MODEL_NAME)
            # torch.save(self.policy_net.state_dict(), self.PATH + self.MODEL_NAME)
            print('Saved Model', self.PATH + self.MODEL_NAME)

            plt.ioff()

    def get_DQN_sampling_vector(self, fps, signal_class, interpolation_method='interp'):

        self.policy_net.load_state_dict(torch.load(self.PATH + self.MODEL_NAME))
        # self.target_net.load_state_dict(torch.load(self.PATH + self.MODEL_NAME))
        self.policy_net.eval()

        T = 1/fps
        # Samples (start with uniform, regular sampling):
        x_vector_list, y_vector = signal_class.sample_signal(fps, "uniform")

        x_vector = torch.tensor(x_vector_list).unsqueeze(0).float()
        y_vector = torch.tensor(y_vector).unsqueeze(0).float()
        x2normedx = lambda x: (x % T)
        x_normed_vector = x2normedx(x_vector)

        cur_x = x_normed_vector[:, :self.state_sample_length]
        cur_y = y_vector[:, :self.state_sample_length]
        state = torch.cat((cur_x, cur_y), 1)
        accumulated_reward = 0.0

        samples_vec = list(x_vector_list[:self.state_sample_length])
        with torch.no_grad():

            for t in range(self.state_sample_length, len(x_vector_list)):
                # Select and perform an action
                # action = self.select_action(state)
                print("state: ", state)
                action = self.policy_net(state)
                print(f"the modle took action # {action}")
                print(f"the modle took action #0 {action[0][0]}")
                print(f"the modle took action #-1 {action[0][-1]}")
                action = action.max(1)[1].view(1, 1)
                print(f"the modle took action # {action}")
                reward, done, cur_x, cur_y, __ = self.get_reward_and_new_state( \
                    cur_x, cur_y, action, t, signal_class, fps, x_normed_vector.shape[1], \
                    interpolation_method=interpolation_method)

                # Observe new state
                if not done:
                    next_state = torch.cat((cur_x, cur_y), 1)
                else:
                    next_state = None

                # Move to the next state
                state = next_state
                accumulated_reward += reward

                samples_vec += [T * (t + float(cur_x[0, -1]))]  # + (1/self.upsampling_factor)/2)*T ]

        # print('RL Model Total Reward:', accumulated_reward)
        # print( samples_vec )
        x_down = np.array(samples_vec)
        y_down = np.interp(x_down, signal_class.x_high_res, signal_class.y_high_res)
        return x_down, y_down


if __name__ == '__main__':
    NUS_parameters = {
        'PATH': 'DQN/model/',
        'MODEL_NAME': 'model_10_01_23-nice',
        'EXP_START_DECAY': 550,
        'EPS_START': 0.8,
        'EPS_END': 0.05,
        'upsampling_factor': 10,
        'BATCH_SIZE': 5,
        'GAMMA': 0.98,
        'state_sample_length': 100,
        'NUM_OF_FUNCTIONS': 20,
        'NUM_OF_EPISODES': 700,
        'TARGET_UPDATE': 40,
        'initial_temperature': 0.7,
        'exploration_strategy': 'softmax'
    }


    simulation_time = 5
    freq = (5, 30)
    fps = 15

    if not os.path.exists(f'DQN/model'):
        os.mkdir(f'DQN/model')

    if not os.path.exists(f'DQN/model/{NUS_parameters["MODEL_NAME"]}_functions'):
        os.mkdir(f'DQN/model/{NUS_parameters["MODEL_NAME"]}_functions')

    signal_type_list = ['sin', 'sin_with_phase', 'pisewise_linear_aperiodic', 'pisewise_linear_periodic']
    # model1 , model_9-1-23'
    rl_model = RLModel(parameters=NUS_parameters)
    rl_model.training_model(signal_type_list=signal_type_list, camera_fps=fps, freqs=freq, simulation_time=simulation_time, interpolation_method='interp')
