# Manual play mode for Frozen Lake with Q-learning.
# The user selects actions via arrow keys, but Q-values are still learned and displayed.

import gymnasium as gym
import numpy as np
import pygame
import pickle

gym.register(
    id="FrozenLake-enhanced",
    entry_point="frozen_lake_enhanced:FrozenLakeEnv",
    kwargs={"map_name": "8x8"},
    max_episode_steps=200,
    reward_threshold=0.85,
)


def play(episodes=1000):
    env = gym.make(
        "FrozenLake-enhanced",
        desc=None,
        map_name="8x8",
        is_slippery=True,
        success_rate=3 / 4,
        reward_schedule=(1, -1, 0),
        render_mode="human",
    )

    q = np.zeros((env.observation_space.n, env.action_space.n))

    learning_rate_a = 0.1
    discount_factor_g = 0.9

    key_to_action = {
        pygame.K_LEFT: 0,
        pygame.K_DOWN: 1,
        pygame.K_RIGHT: 2,
        pygame.K_UP: 3,
    }

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while not terminated and not truncated:
            # Update display with current Q-values
            env.unwrapped.set_q(q)
            env.unwrapped.set_episode(i)
            env.unwrapped.set_info({
                "Mode": "Manual (Arrow Keys)",
            })

            # Wait for an arrow key press
            action = None
            while action is None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            env.close()
                            return
                        if event.key in key_to_action:
                            action = key_to_action[event.key]
                pygame.time.wait(30)

            new_state, reward, terminated, truncated, _ = env.step(action)

            # Q-learning update
            q[state, action] = q[state, action] + learning_rate_a * (
                reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
            )

            state = new_state

        # Brief pause after episode ends so the player can see the result
        pygame.time.wait(1000)

    env.close()

    f = open("frozen_lake8x8_manual.pkl", "wb")
    pickle.dump(q, f)
    f.close()


if __name__ == "__main__":
    play()
