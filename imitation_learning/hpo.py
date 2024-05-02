import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from train import read_data, preprocessing, train_model
import random

import sys

import torch

sys.path.append(".")

import utils
from agent.bc_agent import BCAgent
from tensorboard_evaluation import Evaluation


def randomSearch(X, y, iters=100):
    # Define the search space
    search_space = {
        "lr": [1e-2, 1e-3, 1e-4, 1e-5],
        # "batch_size": [32, 64, 128, 256],
        # "n_minibatches": [100, 500, 1000, 5000],
        "optimizer": ["adam", "sgd"],
        "loss": ["crossentropy"],
        "class_weights": [1, 5]
    }

    # Define the number of iterations
    n_iter = iters

    # Initialize the best hyperparameters and the best accuracy
    best_hyperparameters = None
    best_accuracy = 0

    X_train, X_valid, y_train, y_valid = train_test_split(
        X.cpu(),
        y.cpu(),
        test_size=0.1,
        random_state=42,  # Seed for reproducibility
        stratify=y.cpu()  # Stratify to maintain class distribution
    )

    for i in range(n_iter):
        # Sample hyperparameters
        hyperparameters = {
            "lr": np.random.choice(search_space["lr"]),
            # "batch_size": np.random.choice(search_space["batch_size"]),
            # "n_minibatches": np.random.choice(search_space["n_minibatches"]),
            "optimizer": np.random.choice(search_space["optimizer"]),
            "loss": np.random.choice(search_space["loss"]),
            "class_weights": [random.randrange(search_space["class_weights"][0], search_space["class_weights"][1]) for _ in range(4)]
        }
        print("Hyperparameters: ", hyperparameters)
        agent_config = {'lr': hyperparameters['lr'], 'optimizer': hyperparameters['optimizer'], 'loss': hyperparameters['loss']}

        # Initialize the agent
        agent = BCAgent(config=agent_config)

        # Train the agent
        accuracy = train_model(agent, X_train, y_train, X_valid, y_valid, weights=hyperparameters['class_weights'],
                               batch_size=64, n_minibatches=300, tensorboard_dir="", model_dir="")
        # Update the best hyperparameters and the best accuracy
        if accuracy > best_accuracy:
            best_hyperparameters = hyperparameters
            best_accuracy = accuracy
        if i == 100 and accuracy < 60:
            continue

    return best_hyperparameters, best_accuracy


if __name__ == "__main__":
    # read data
    X, y = read_data("./data")
    X, y = preprocessing(X, y, 1)

    config, accuracy = randomSearch(X, y, iters=100)
    print("Best hyperparameters: ", config, "Best accuracy: ", accuracy)
    with open("best_hyperparameters.pkl", "wb") as f:
        pickle.dump(config, f)

