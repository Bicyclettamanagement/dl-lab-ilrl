import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import sys

import torch
import torch.nn.functional as F

from imitation_learning.agent.networks import CNN

sys.path.append(".")

import utils
from agent.bc_agent import BCAgent
from tensorboard_evaluation import Evaluation


def read_data_old(datasets_dir="./data"):
    """
    This method reads the states and actions recorded in drive_manually.py
    and splits it into training/ validation set.
    """
    print("... read data")
    initial = True
    data = dict()
    for file in os.listdir(datasets_dir):
        if file.endswith(".pkl.gzip"):
            print("reading data from %s" % file)
        data_file = os.path.join(datasets_dir, file)
        f = gzip.open(data_file, "rb")
        if initial:
            data = pickle.load(f)
            initial = False
        while 1:
            try:
                more_data = pickle.load(f)
                data["state"] = np.append(data["state"], more_data["state"], axis=0)  # state has shape (96, 96, 3)
                data["action"] = np.append(data["action"], more_data["action"], axis=0)  # action has shape (1, 3)
                data["next_state"] = np.append(data["next_state"], more_data["next_state"], axis=0)
                data["reward"] = np.append(data["reward"], more_data["reward"], axis=0)
                data["terminal"] = np.append(data["terminal"], more_data["terminal"], axis=0)
            except EOFError:
                break
        f.close()
    print("loaded %d samples" % len(data["state"]))

    # get images as features and actions as targets
    X = np.array(data["state"]).astype("float32")
    y = np.array(data["action"]).astype("float32")
    return X, y


def read_data(datasets_dir="./data"):
    """
    Reads the states and actions recorded in drive_manually.py and splits it into training/ validation set.
    """
    print("... read data")

    # Initialize lists to hold data
    states_list = []
    actions_list = []
    # next_states_list = []
    # rewards_list = []
    # terminals_list = []

    # Iterate through each file in the directory
    for file in os.listdir(datasets_dir):
        if file.endswith(".pkl.gzip"):
            print(f"Reading data from {file}")

            # Open the file using gzip
            data_file = os.path.join(datasets_dir, file)
            with gzip.open(data_file, "rb") as f:
                while True:
                    try:
                        # Load the data from the file
                        more_data = pickle.load(f)
                        x = np.array(more_data["state"])
                        y = np.array(more_data["action"])

                        # Collect data in lists
                        states_list.extend(more_data["state"])
                        actions_list.extend(more_data["action"])
                        # next_states_list.extend(more_data["next_state"])
                        # rewards_list.extend(more_data["reward"])
                        # terminals_list.extend(more_data["terminal"])
                        if len(states_list) % 1000 == 0:
                            print(f"Loaded {len(states_list)} samples")
                        if len(states_list) >= 40000:
                            break
                    except EOFError:
                        break
    # X, y = utils.remove_redundant_samples(states_list, actions_list)

    # Convert lists to NumPy arrays after all data is collected
    X = np.array(states_list, dtype=np.float16).reshape(-1, 96, 96, 3)  # Assuming state shape is (96, 96, 3)
    y = np.array(actions_list, dtype=np.float16).reshape(-1, 3)  # Assuming action shape is (3,)

    # Output the number of samples loaded
    print(f"Loaded {X.shape[0]} samples")

    return X, y


def preprocessing(X_train, y_train, history_length=1, device='cuda'):
    """
    Preprocesses the input data and labels for training.

    Parameters:
    - X_train: Input data (NumPy array or PyTorch tensor).
    - y_train: Target labels (NumPy array or PyTorch tensor).
    - history_length: Number of previous images to include in the input.
    - device: PyTorch device (e.g., 'cuda' or 'cpu') where the tensors will be stored.

    Returns:
    - X_tensor: Preprocessed input data as a tensor.
    - y_tensor: Target labels as a tensor.
    """
    print("... preprocess data")

    # Convert input data and labels to PyTorch tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_train, dtype=torch.long, device=device)

    # Convert images to grayscale using rgb2gray_batch and reshape to (batch_size, 96, 96)
    X_tensor = utils.rgb2gray_batch(X_tensor).squeeze(1)  # Remove the channel dimension

    # Define padding for the batch dimension: Symmetric padding around the history_length
    padding_shape = (0, 0, 0, 0, history_length - 1, 0)  # Pad along the batch dimension

    # Apply padding to handle the history_length
    X_tensor = F.pad(X_tensor, padding_shape, mode='constant', value=0)

    # Check tensor shape after padding
    print(f"Shape after padding: {X_tensor.shape}")

    # Create a view of the data with the specified history length
    # Use unfold to create the sliding window view of data with the history_length
    X_tensor = X_tensor.unfold(0, history_length, 1).permute(0, 3, 1, 2).contiguous()

    # Check shape after unfolding
    print(f"Shape after unfolding: {X_tensor.shape}")
    y_tensor = utils.action_to_id_batch(y_tensor)

    # Move the tensors to the specified device
    X_tensor = X_tensor.to('cpu')
    y_tensor = y_tensor.to('cpu')

    return X_tensor, y_tensor


def train_model(
        agent,
        X_train,
        y_train,
        X_valid,
        y_valid,
        n_minibatches,
        batch_size,
        weights=None,
        model_dir="./models",
        tensorboard_dir="./tensorboard",
):
    # create result and model folders
    if weights is None:
        weights = [1, 1, 1, 1]
    if not os.path.exists(model_dir) and model_dir != "":
        os.mkdir(model_dir)

    print("... train model")
    valid_accuracy = 0
    X_valid, y_valid = X_valid.to('cuda'), y_valid.to('cuda')

    # TODO: specify your agent with the neural network in agents/bc_agent.py
    agent = agent

    if tensorboard_dir != "":
        tensorboard_eval = Evaluation(tensorboard_dir, "Imitation Learning")

    # TODO: implement the training
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training *during* the training in your web browser
    #
    # training loop

    for i in range(n_minibatches):
        # sample a minibatch
        X_batch, y_batch = utils.sample_minibatch(X_train, y_train, batch_size, weights=weights)
        X_batch = X_batch.to('cuda')
        y_batch = y_batch.to('cuda')
        # perform an update step
        loss = agent.update(X_batch, y_batch)
        predictions = agent.predict(X_batch)

        if i % 10 == 0:
            # compute training/ validation accuracy and write it to tensorboard
            correct = torch.tensor(predictions == y_batch, device='cuda').float()
            train_accuracy = correct.mean() * 100
            correct = torch.tensor(agent.predict(X_valid) == y_valid, device='cuda').float()
            valid_accuracy = correct.mean() * 100
            if tensorboard_dir != "":
                tensorboard_eval.write_episode_data(i, {"loss": loss, "train_accuracy": train_accuracy,
                                                        "valid_accuracy": valid_accuracy})
            print(
                "Step: %d, Loss: %f, Train accuracy: %f, Valid accuracy: %f"
                % (i, loss, train_accuracy, valid_accuracy)
            )

    # TODO: save your agent
    if model_dir != "":
        model_dir = agent.save(os.path.join(model_dir, "agent.pt"))
        print("Model saved in file: %s" % model_dir)
    return valid_accuracy


if __name__ == "__main__":
    history_length = 5
    X, y = read_data("./data")
    size = len(X) // 4
    start_indices = [0, size, 2 * size, 3 * size]
    end_indices = [size, 2 * size, 3 * size, len(X)]
    # Split the data into 4 equal parts
    X1, y1 = X[start_indices[0]:end_indices[0]], y[start_indices[0]:end_indices[0]]
    X2, y2 = X[start_indices[1]:end_indices[1]], y[start_indices[1]:end_indices[1]]
    X3, y3 = X[start_indices[2]:end_indices[2]], y[start_indices[2]:end_indices[2]]
    X4, y4 = X[start_indices[3]:end_indices[3]], y[start_indices[3]:end_indices[3]]

    X1, y1 = preprocessing(X1, y1, history_length)
    X2, y2 = preprocessing(X2, y2, history_length)
    X3, y3 = preprocessing(X3, y3, history_length)
    X4, y4 = preprocessing(X4, y4, history_length)
    config = {'lr': 0.0001, 'optimizer': 'adam', 'loss': 'crossentropy', 'class_weights': [4, 2, 2, 3]}
    X1_train, X1_valid, y1_train, y1_valid = train_test_split(
        X1,
        y1,
        test_size=0.2,
        random_state=42,  # Seed for reproducibility
        stratify=y1  # Stratify to maintain class distribution
    )
    X2_train, X2_valid, y2_train, y2_valid = train_test_split(
        X2,
        y2,
        test_size=0.2,
        random_state=42,  # Seed for reproducibility
        stratify=y2  # Stratify to maintain class distribution
    )
    X3_train, X3_valid, y3_train, y3_valid = train_test_split(
        X3,
        y3,
        test_size=0.2,
        random_state=42,  # Seed for reproducibility
        stratify=y3  # Stratify to maintain class distribution
    )
    X4_train, X4_valid, y4_train, y4_valid = train_test_split(
        X4,
        y4,
        test_size=0.2,
        random_state=42,  # Seed for reproducibility
        stratify=y4  # Stratify to maintain class distribution
    )
    X1_train = torch.tensor(X1_train, dtype=torch.float16)
    X2_train = torch.tensor(X2_train, dtype=torch.float16)
    X3_train = torch.tensor(X3_train, dtype=torch.float16)
    X4_train = torch.tensor(X4_train, dtype=torch.float16)
    X_train = torch.cat((X1_train, X2_train, X3_train, X4_train))
    y_train = torch.cat((y1_train, y2_train, y3_train, y4_train))
    X_valid = torch.cat((X1_valid, X2_valid, X3_valid, X4_valid))
    y_valid = torch.cat((y1_valid, y2_valid, y3_valid, y4_valid))

    # preprocess data
    # X_train, y_train, X_valid, y_valid = preprocessing(
        # X_train, y_train, X_valid, y_valid, history_length=1
    #
    agent_config = {'lr': config["lr"], 'optimizer': config["optimizer"], 'loss': config["loss"]}
    class_weights = config["class_weights"]
    net = CNN(history_length=history_length, n_classes=4)
    net.to('cuda')
    agent = BCAgent(network=net, config=agent_config)
    # train model, 1250 minibatches = 1 episode (80000 samples)
    train_model(agent, X_train, y_train, X_valid, y_valid, n_minibatches=100000, batch_size=64, weights=class_weights)
