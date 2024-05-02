import numpy as np
import torch
import torch.optim as optim
from agent.replay_buffer import ReplayBuffer


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DQNAgent:

    def __init__(
            self,
            Q,
            Q_target,
            num_actions,
            gamma=0.95,
            batch_size=64,
            epsilon=0.2,
            tau=0.01,
            lr=1e-4,
            history_length=0,
            explore_probs=None
    ):
        """
        Q-Learning agent for off-policy TD control using Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.

        Args:
           Q: Action-Value function estimator (Neural Network)
           Q_target: Slowly updated target network to calculate the targets.
           num_actions: Number of actions of the environment.
           gamma: discount factor of future rewards.
           batch_size: Number of samples per batch.
           tau: indicates the speed of adjustment of the slowly updated target network.
           epsilon: Chance to sample a random action. Float betwen 0 and 1.
           lr: learning rate of the optimizer
        """
        # setup networks
        if explore_probs is None:
            explore_probs = np.ones(num_actions)
        self.explore_probs = explore_probs / np.sum(np.array(explore_probs))
        self.Q = Q.cuda()
        self.Q_target = Q_target.cuda()
        self.Q_target.load_state_dict(self.Q.state_dict())

        # define replay buffer
        self.replay_buffer = ReplayBuffer(history_length, explore_probs=explore_probs)

        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)

        self.num_actions = num_actions

    def train(self, state, action: int, next_state, reward: float, terminal: bool):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """

        # TODO:
        # 1. add current transition to replay buffer
        # 2. sample next batch and perform batch update:
        #       2.1 compute td targets and loss
        #              td_target =  reward + discount * max_a Q_target(next_state_batch, a)
        #       2.2 update the Q network
        #       2.3 call soft update for target network
        #           soft_update(self.Q_target, self.Q, self.tau)
        self.replay_buffer.add_transition(state.cuda(), action, next_state.cuda(), reward, terminal)

        batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminal_flags = (
            self.replay_buffer.next_batch(self.batch_size))
        batch_states, batch_next_states = batch_states.float(), batch_next_states.float()
        target = batch_rewards + ~batch_terminal_flags * self.gamma * torch.max(self.Q_target(batch_next_states), dim=1)[0]
        current_prediction = torch.max(self.Q(batch_states), dim=1)[0]
        loss = self.loss_function(current_prediction, target.detach().float())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        soft_update(self.Q_target, self.Q, self.tau)
        return loss.item()

    def act(self, state, deterministic):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        r = np.random.uniform()
        if deterministic or r > self.epsilon:
            # TODO: take greedy action (argmax)
            action_id = torch.argmax(self.Q(state)).item()
        else:
            pass
            # TODO: sample random action
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work.
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
            action_id = np.random.choice(self.num_actions, p=self.explore_probs)
        self.epsilon = max(0.01, self.epsilon * 0.99)
        return action_id

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name))
        self.Q_target.load_state_dict(torch.load(file_name))
