# This file is almost identical to frozen_lake_q.py, except this uses the frozen_lake_enhanced.py environment.

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Register the enhanced frozen lake environment
# Sample of registration entry found in C:\Users\<username>\.conda\envs\gymenv\Lib\site-packages\gymnasium\envs\__init__.py
gym.register(
    id="FrozenLake-enhanced",  # give it a unique id
    entry_point="frozen_lake_enhanced:FrozenLakeEnv",  # frozen_lake_enhanced = name of file 'frozen_lake_enhanced.py'
    kwargs={"map_name": "8x8"},
    max_episode_steps=200,
    reward_threshold=0.85,  # optimum = 0.91
)


def run(episodes, is_training=True, render=False):

    # 'FrozenLake-enhanced' is the id specified above
    env = gym.make(
        "FrozenLake-enhanced",
        desc=None,
        map_name="8x8",
        is_slippery=True,
        success_rate=3 / 4,
        reward_schedule=(1, -1, 0),
        render_mode="human" if render else None,
    )

    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        f = open("frozen_lake8x8.pkl", "rb")
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.1  # alpha or learning rate
    discount_factor_g = 0.9  # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    epsilon = 1.0  # 1 = 100% random actions
    epsilon_decay_rate = 0.0001  # epsilon decay rate. 1/0.0001 = 10,000
    show_q_values = True
    rng = np.random.default_rng()  # random number generator

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[
            0
        ]  # states: 0 to 63, 0=top left corner,63=bottom right corner
        terminated = False  # True when fall in hole or reached goal
        truncated = False  # True when actions > 200

        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                action = (
                    env.action_space.sample()
                )  # actions: 0=left,1=down,2=right,3=up
            else:
                max_q = np.max(q[state, :])
                max_actions = np.where(np.abs(q[state, :] - max_q) < 1e-4)[0]
                action = rng.choice(max_actions)

            new_state, reward, terminated, truncated, _ = env.step(action)

            # "Illegal" move:
            # if new_state == state:
            #    reward = -.1

            if is_training:
                q[state, action] = q[state, action] + learning_rate_a * (
                    reward
                    + discount_factor_g * np.max(q[new_state, :])
                    - q[state, action]
                )

            # pass the q table and episode count to the environment for rendering
            if env.render_mode == "human":
                env.unwrapped.set_show_q_labels(show_q_values)
                env.unwrapped.set_q(q)
                env.unwrapped.set_episode(i)
                env.unwrapped.set_info({
                    "Epsilon": f"{epsilon:.4f}",
                    "Learning Rate": f"{learning_rate_a:.4f}",
                    "Discount Factor": f"{discount_factor_g:.2f}",
                })

            state = new_state

        if terminated:
            print(new_state, reward, terminated, truncated)

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if epsilon == 0:
            learning_rate_a = 0.01

        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100) : (t + 1)])
    plt.plot(sum_rewards)
    plt.savefig("frozen_lake8x8.png")

    if is_training:
        f = open("frozen_lake8x8.pkl", "wb")
        pickle.dump(q, f)
        f.close()


if __name__ == "__main__":
    run(15000, is_training=True, render=True)
