from collections import namedtuple
import numpy as np
import torch
import os
import gzip
import pickle

import utils


class ReplayBuffer:
    def __init__(self, history_length=1, max_size=1e4, explore_probs=None):
        self._max_size = int(max_size)
        self.states = torch.empty((self._max_size, history_length, 96, 96), dtype=torch.float16, device=torch.device('cuda'))
        self.actions = torch.empty((self._max_size), dtype=torch.int, device=torch.device('cuda'))
        self.next_states = torch.empty((self._max_size, history_length, 96, 96), dtype=torch.float16, device=torch.device('cuda'))
        self.rewards = torch.empty((self._max_size), dtype=torch.float32, device=torch.device('cuda'))
        self.dones = torch.empty((self._max_size), dtype=torch.bool, device=torch.device('cuda'))
        self._size = 0
        self.explore_probs = explore_probs


    def add_transition(self, state, action, next_state, reward, done):
        """
        Adds a transition to the replay buffer using tensors.
        """

        # Concatenate the new transition to the replay buffer
        if self._size < self._max_size:
            self.states[self._size] = state
            self.actions[self._size] = action
            self.next_states[self._size] = next_state
            self.rewards[self._size] = reward
            self.dones[self._size] = done

            self._size += 1
        else:
            # Remove the oldest transition using slicing
            self.states = torch.cat((self.states[1:], state))
            self.actions = torch.cat((self.actions[1:], torch.tensor([action], device=torch.device('cuda'))))
            self.next_states = torch.cat((self.next_states[1:], next_state))
            self.rewards = torch.cat((self.rewards[1:], torch.tensor([reward], device=torch.device('cuda'))))
            self.dones = torch.cat((self.dones[1:], torch.tensor([done], device=torch.device('cuda'))))

    def next_batch(self, batch_size):
        """
        This method samples a batch of transitions.
        """
        batch_indices = np.random.choice(len(self.states), batch_size)
        # batch_indices = utils.weighted_sampling(self.actions[:self._size], self.explore_probs, batch_size)

        batch_states = self.states[batch_indices]
        batch_actions = self.actions[batch_indices]
        batch_next_states = self.next_states[batch_indices].cuda()
        batch_rewards = self.rewards[batch_indices]
        batch_dones = self.dones[batch_indices]
        return (
            batch_states,
            batch_actions,
            batch_next_states,
            batch_rewards,
            batch_dones,
        )
