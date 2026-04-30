import os
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

gym.register(
    id="FrozenLake-v2",
    entry_point="techdays26.frozen_lake.frozen_lake_enhanced:FrozenLakeEnv",
    kwargs={"map_name": "8x8"},
    max_episode_steps=200,
    reward_threshold=0.85,
)


def make_frozen_lake(render_mode="rgb_array", show_values=False, slippery=False):
    """Create a Frozen Lake env with our default parameters.

    - show_values=True overlays V(s) (or Q(s, a)) on the grid.
    - slippery=True turns on stochastic transitions (see the Bonus section).
    """
    return gym.make(
        "FrozenLake-v2",
        desc=None,  # pass a custom map here, or use map_name
        map_name="8x8",
        show_q_labels=show_values,
        is_slippery=slippery,
        success_rate=3
        / 4,  # probability the intended action happens (only if slippery)
        reward_schedule=(1, -1, 0),  # (goal, hole, every other step)
        render_mode=render_mode,
    )


# ... any other imports your env needs ...


def generate_frozen_lake_gif(
    out_path: str = "frozen_lake_elf.gif",
    num_steps: int = 300,
    fps: int = 10,
    seed: int | None = 0,
) -> str:
    """Run a random policy on the Frozen Lake env and store a GIF.

    Parameters
    ----------
    out_path : str
        Output gif path.
    num_steps : int
        Number of steps to simulate.
    fps : int
        Frames per second for the GIF.
    seed : int | None
        Optional random seed.

    Returns:
    -------
    str
        Path to the created GIF.
    """
    if seed is not None:
        np.random.seed(seed)

    # -------------------------------------------------------------------------
    # Create the environment
    # Adjust this factory call to match your repo's API:
    # e.g. FrozenLakeEnhanced(), make_frozen_lake_env(...), etc.
    # -------------------------------------------------------------------------
    env = make_frozen_lake()  # <-- adapt if needed

    # New-style Gymnasium API often returns (obs, info)
    reset_out = env.reset()
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        obs, info = reset_out
    else:
        obs = reset_out

    frames: list[np.ndarray] = []

    # initial frame
    frame = env.render()

    print("Frame shape:", frame.shape, "dtype:", frame.dtype)

    frames.append(frame)

    for _ in range(num_steps):
        # Choose random action
        action = env.action_space.sample()

        # New-style API can be: obs, reward, terminated, truncated, info
        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            obs, reward, done, info = step_out

        frame = env.render()

        frames.append(frame)

        if done:
            reset_out = env.reset()
            if isinstance(reset_out, tuple) and len(reset_out) == 2:
                obs, info = reset_out
            else:
                obs = reset_out

            frame = env.render()

            frames.append(frame)

    env.close()

    # Ensure directory exists
    out_path = str(Path(out_path))
    Path(os.path.dirname(out_path) or ".").mkdir(exist_ok=True, parents=True)

    if out_path.endswith(".gif"):
        # Save frames as GIF
        imageio.mimsave(out_path, frames, fps=fps)
    elif out_path.endswith(".mp4"):
        imageio.mimsave(out_path, frames, fps=fps, codec="libx264")
    else:
        assert False

    return out_path


if __name__ == "__main__":
    path = generate_frozen_lake_gif(
        out_path="artifacts/frozen_lake_elf.gif",
        num_steps=400,
        fps=8,
        seed=42,
    )
    print(f"Saved GIF to: {path}")
