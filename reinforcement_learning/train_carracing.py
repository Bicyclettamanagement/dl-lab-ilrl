# export DISPLAY=:0

import sys

from imitation_learning.agent.networks import CNN

sys.path.append("../")

import numpy as np
import gym
from tensorboard_evaluation import *
from utils import EpisodeStats, rgb2gray
from utils import *
from agent.dqn_agent import DQNAgent


def run_episode(
        env,
        agent,
        deterministic,
        skip_frames=0,
        do_training=True,
        rendering=False,
        max_timesteps=1000,
        history_length=1,
):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    # image_hist = []

    step = 0
    state = env.reset()
    state_history = torch.zeros(1, history_length, 96, 96, device='cuda')

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events()

    # append image history to first state
    state = state_preprocessing(state)
    # image_hist.extend([state] * (history_length))
    # state = np.array(image_hist).reshape(history_length, 96, 96)
    if history_length > 1:
        state_history = torch.cat((state_history[:, 1:], state), dim=1)
        state = state_history.clone()

    while True:

        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly.

        action_id = agent.act(state_history, deterministic=deterministic)
        action = id_to_action(action_id)

        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal:
                break

        next_state = state_preprocessing(next_state)
        # image_hist.append(next_state)
        # image_hist.pop(0)
        # next_state = np.array(image_hist).reshape(history_length, 96, 96)
        if history_length > 1:
            state_history = torch.cat((state_history[:, 1:], next_state), dim=1)
            next_state = state_history.clone()
        loss = 0
        if do_training:
            loss = agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id, loss)

        state = next_state

        if terminal or (step * (skip_frames + 1)) > max_timesteps:
            break

        step += 1

    print("episode reward: ", stats.episode_reward, ", episode length: ", step)
    return stats


def train_online(
        env,
        agent,
        num_episodes,
        history_length=1,
        skip_frames=0,
        model_dir="./models_carracing",
        tensorboard_dir="./tensorboard",
        max_timesteps=100,
        render=False,
):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train agent")
    tensorboard = Evaluation(
        os.path.join(tensorboard_dir, "train"),
        ["episode_reward", "loss" "straight", "left", "right", "accel", "brake"],
    )

    for i in range(num_episodes):
        print("epsiode %d" % i)

        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)

        stats = run_episode(
            env,
            agent,
            max_timesteps=max_timesteps + i,
            deterministic=False,
            do_training=True,
            history_length=history_length,
            skip_frames=skip_frames,
            rendering=render,
        )

        tensorboard.write_episode_data(
            i,
            eval_dict={
                "1_episode_reward": stats.episode_reward,
                "loss": stats.cumulative_loss,
                "straight": stats.get_action_usage(STRAIGHT),
                "left": stats.get_action_usage(LEFT),
                "right": stats.get_action_usage(RIGHT),
                "accel": stats.get_action_usage(ACCELERATE),
                "brake": stats.get_action_usage(BRAKE),
            },
        )

        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        # ...
        if i % eval_cycle == 0:
            for j in range(num_eval_episodes):
                stats = run_episode(env, agent, deterministic=True, do_training=False,
                                    history_length=history_length, skip_frames=skip_frames, rendering=render)
            mean_reward = np.mean(stats.episode_reward)
            tensorboard.write_episode_data(
                i,
                eval_dict={
                    "mean_reward": mean_reward,
                    "straight": stats.get_action_usage(STRAIGHT),
                    "left": stats.get_action_usage(LEFT),
                    "right": stats.get_action_usage(RIGHT),
                    "accel": stats.get_action_usage(ACCELERATE),
                    "brake": stats.get_action_usage(BRAKE),
                },
            )
            print("Evaluation complete, mean reward: ", mean_reward)

        # store model.
        if i % eval_cycle == 0 or (i >= num_episodes - 1):
            # agent.saver.save(agent.sess, os.path.join(model_dir, "dqn_agent.ckpt"))
            agent.save(os.path.join(model_dir, "dqn_agent.pt"))

    tensorboard.close_session()


def state_preprocessing(state):
    # Convert input data and labels to PyTorch tensors
    state_tensor = torch.tensor(state.copy(), dtype=torch.float32, device='cuda')

    # Convert images to grayscale using rgb2gray_batch and reshape to (batch_size, 96, 96)
    state_tensor = rgb2gray_batch(state_tensor)
    return state_tensor


if __name__ == "__main__":
    num_episodes = 10000
    num_eval_episodes = 5
    eval_cycle = 20
    history_length = 2
    skip_frames = 2
    explore_probs = np.array([5, 2, 2, 4, 1])

    torch.cuda.empty_cache()

    env = gym.make("CarRacing-v0").unwrapped

    # TODO: Define Q network, target network and DQN agent
    # ...
    Q_net = CNN(history_length=history_length, n_classes=5)
    Q_net.to("cuda")
    Target_net = CNN(history_length=history_length, n_classes=5)
    Target_net.to("cuda")
    agent = DQNAgent(Q_net, Target_net, num_actions=5, explore_probs=explore_probs, history_length=history_length)

    train_online(
        env, agent, num_episodes=num_episodes, history_length=history_length,
        model_dir="./models_carracing", max_timesteps=100, skip_frames=skip_frames, render=False
    )
