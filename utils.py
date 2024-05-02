import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader

LEFT = 1
RIGHT = 2
STRAIGHT = 0
ACCELERATE = 3
BRAKE = 4

import numpy as np
import torch


def remove_redundant_samples(X, y):
    """
    Removes redundant samples from the data based on identical feature vectors.
    - X: Input data (NumPy array).
    - y: Target labels (NumPy array).
    Returns:
        - X_unique: Array of unique samples.
        - y_unique: Array of labels corresponding to unique samples.
    """
    # Create a dictionary to keep track of unique samples and their indices
    unique_samples = {}

    # Iterate through the dataset
    for idx, sample in enumerate(X):
        # Convert the sample to a hashable tuple (e.g., a tuple of features)
        sample_tuple = tuple(sample.flatten())

        # If the sample is not already in the dictionary, add it
        if sample_tuple not in unique_samples:
            unique_samples[sample_tuple] = idx

    # Get the indices of unique samples
    unique_indices = list(unique_samples.values())

    # Filter the input data and labels to keep only unique samples
    X_unique = X[unique_indices]
    y_unique = y[unique_indices]

    return X_unique, y_unique


def weighted_sampling(y, weights=None, batch_size=32):
    """
    Weighted sampling function to create a mini-batch.
    - y: Target labels
    - weights: Optional list or array of weights for each class.
    - batch_size: Size of the batch.
    Returns:
        - batch_ids: Array of selected indices for the mini-batch.
    """
    if weights is not None:
        # Convert weights to NumPy array and normalize them
        weights = np.array(weights)

        # Calculate actual class distribution
        # class_counts = np.bincount(y)
        class_counts = torch.bincount(y).cpu().numpy()
        y_length = y.size(dim=0)

        # Calculate the adjusted weights and batch proportions
        # Cap weights by actual class count to prevent over-sampling a class
        class_count_ids = np.arange(len(class_counts))
        weights = weights[class_count_ids]
        adjusted_weights = np.minimum(weights, class_counts)

        # Normalize the adjusted weights to sum to the batch size
        adjusted_weights = adjusted_weights / adjusted_weights.sum() * batch_size

        # If adjusted weights do not sum up to batch size, redistribute the remainder
        remainder = batch_size - adjusted_weights.sum()
        if remainder > 0:
            adjusted_weights += remainder / len(adjusted_weights)

        # Convert adjusted weights to integers
        adjusted_weights = adjusted_weights.astype(int)

        # Sampling process
        batch_ids = []
        for class_id in range(len(weights)):
            # Get indices of the current class
            # class_indices = np.where(y == class_id)[0]
            class_indices = torch.nonzero(y == class_id)
            class_indices_length = class_indices.size(dim=0)
            # class_indices_length = len(class_indices)

            # Sample according to the adjusted weight
            # if len(class_indices) > 0:
            if class_indices_length > 0:
                num_samples = min(adjusted_weights[class_id], class_indices_length)
                # sampled_indices = np.random.choice(class_indices, num_samples, replace=True)
                probabilities = torch.full_like(class_indices, 1 / class_indices_length, dtype=torch.float32).squeeze(1)
                sampled_indices = torch.multinomial(probabilities, num_samples, replacement=True).cpu().tolist()
                batch_ids.extend(sampled_indices)

        # If the total number of samples is less than the batch size due to rounding errors,
        # randomly sample more from the overall indices.
        if len(batch_ids) < batch_size:
            all_indices = np.arange(y_length)
            needed_samples = batch_size - len(batch_ids)
            additional_samples = np.random.choice(all_indices, needed_samples)
            batch_ids.extend(additional_samples)

        # Convert list to numpy array and return it
        batch_ids = np.array(batch_ids)
        return batch_ids

    else:
        # If no weights provided, randomly sample a batch
        batch_ids = np.random.choice(y.size().item(), batch_size, replace=False)
        return batch_ids

def sample_minibatch(X, y, batch_size, weights=None):
    """
    this method samples a minibatch from the data.
    """
    # if weights is not None:
    #     # normalize the weights
    #     weights = np.array(weights)
    #     weights = weights / weights.sum() * batch_size
    #     if weights.sum() != batch_size:
    #         weights[0] += batch_size - weights.sum()
    #     weights = weights.astype(int)
    #
    #     a0 = np.where(y == 0)[0]
    #     a1 = np.where(y == 1)[0]
    #     a2 = np.where(y == 2)[0]
    #     a3 = np.where(y == 3)[0]
    #     a4 = np.where(y == 4)[0]
    #     x0 = np.random.choice(a0, weights[0], replace=False)
    #     x1 = np.random.choice(a1, weights[1], replace=False)
    #     x2 = np.random.choice(a2, weights[2], replace=False)
    #     x3 = np.random.choice(a3, weights[3], replace=False)
    #     x4 = np.random.choice(a4, weights[4], replace=True)
    #     batch_ids = np.concatenate((x0, x1, x2, x3, x4), axis=0)
    # else:
    #     batch_ids = np.random.choice(len(X), batch_size, replace=False)
    # np.random.shuffle(batch_ids)
    batch_ids = weighted_sampling(y, weights, batch_size)
    batch_X, batch_y = X[batch_ids].float(), y[batch_ids].long()
    batch_X = torch.reshape(batch_X, (batch_size, X.size(dim=1), 96, 96))
    # batch_X = torch.FloatTensor(batch_X)
    # batch_y = torch.LongTensor(batch_y)
    return batch_X, batch_y


def rgb2gray(rgb):
    """
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[..., :3], [0.2125, 0.7154, 0.0721])
    return gray.astype("float32")


def rgb2gray_batch(rgb):
    """
    This method converts a batch of RGB images to grayscale.
    The input tensor is assumed to have shape (batch_size, channels, height, width),
    where channels is expected to be 3 for RGB images.

    :param rgb: A tensor of shape (batch_size, 3, height, width) containing RGB images.
    :return: A tensor of shape (batch_size, 1, height, width) containing grayscale images.
    """
    rgb = rgb.unsqueeze(0)
    if rgb.shape[-1] == 3:  # If channels are in the last dimension (NHWC format)
        rgb = rgb.permute(0, 3, 1, 2)  # Rearrange dimensions to (batch_size, channels, height, width)
    weights = torch.tensor([0.2125, 0.7154, 0.0721], device=rgb.device)
    gray = torch.einsum('bcij,c->bij', rgb, weights)
    gray = gray.unsqueeze(1)

    return gray


def action_to_id(a):
    """
    this method discretizes the actions.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    if np.array_equal(a, np.array([-1.0, 0.0, 0.0])):
        return LEFT  # LEFT: 1
    elif np.array_equal(a, np.array([1.0, 0.0, 0.0])):
        return RIGHT  # RIGHT: 2
    elif np.array_equal(a, np.array([0.0, 1.0, 0.0])):
        return ACCELERATE  # ACCELERATE: 3
    elif a[2] > 0.0:
        return BRAKE  # BRAKE: 4
    else:
        return STRAIGHT  # STRAIGHT = 0


def action_to_id_batch(actions):
    """
        This function discretizes the actions for a whole batch.
        It assumes that each action is a tensor of shape (batch_size, 3),
        where each row contains the three action components (e.g., [-1.0, 0.0, 0.0]).

        :param actions: A tensor of shape (batch_size, 3) containing the actions.
        :return: A tensor of shape (batch_size,) containing the discretized action IDs.
        """
    # Define the action templates
    left_action = torch.tensor([-1.0, 0.0, 0.0], device=actions.device)
    right_action = torch.tensor([1.0, 0.0, 0.0], device=actions.device)
    accelerate_action = torch.tensor([0.0, 1.0, 0.0], device=actions.device)

    # Perform batch comparisons
    left_mask = torch.all(actions == left_action, dim=1)
    right_mask = torch.all(actions == right_action, dim=1)
    accelerate_mask = torch.all(actions == accelerate_action, dim=1)
    brake_mask = actions[:, 2] > 0.0  # Check if the third component is greater than 0.0

    # Initialize the output tensor with STRAIGHT (0)
    action_ids = torch.full((actions.shape[0],), STRAIGHT, device=actions.device)

    # Set the action IDs based on the masks
    action_ids[left_mask] = LEFT
    action_ids[right_mask] = RIGHT
    action_ids[accelerate_mask] = ACCELERATE
    action_ids[brake_mask] = BRAKE

    return action_ids


def id_to_action(action_id, max_speed=0.8):
    """
    this method makes actions continous.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    a = np.array([0.0, 0.0, 0.0])

    if action_id == LEFT:
        return np.array([-1.0, 0.05, 0.05])
    elif action_id == RIGHT:
        return np.array([1.0, 0.05, 0.05])
    elif action_id == ACCELERATE:
        return np.array([0.0, max_speed, 0.0])
    elif action_id == BRAKE:
        return np.array([0.0, 0.0, 0.1])
    else:
        return np.array([0.0, 0.0, 0.0])


class EpisodeStats:
    """
    This class tracks statistics like episode reward or action usage.
    """

    def __init__(self):
        self.cumulative_loss = 0
        self.episode_reward = 0
        self.actions_ids = []

    def step(self, reward, action_id, loss=0):
        self.cumulative_loss += loss
        self.episode_reward += reward
        self.actions_ids.append(action_id)

    def get_action_usage(self, action_id):
        ids = np.array(self.actions_ids)
        return len(ids[ids == action_id]) / len(ids)
