import time
import warnings
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from bitbully import BitBully

# Silence the non-interactive FigureCanvasAgg warning from bitbully.gui_c4
warnings.filterwarnings(
    "ignore",
    message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
)

from bitbully.gui_c4 import GuiC4


def _fig_to_rgb_array(fig) -> np.ndarray:
    """Render a Matplotlib figure to an RGB array (H, W, 3, uint8), using
    FigureCanvasAgg.tostring_argb().
    """
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    # FigureCanvasAgg provides ARGB bytes
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    argb = buf.reshape(h, w, 4)

    # ARGB -> RGB by dropping alpha and reordering channels
    rgb = argb[..., 1:]  # (A, R, G, B) -> (R, G, B)
    return rgb


def generate_c4_animation_mp4(
    out_path: str = "artifacts/c4_bitbully_game.mp4",
    fps: int = 2,
    sleep_between_moves: float = 0.0,
    max_moves: int = 42,
    max_depth_yellow: int = 4,
    max_depth_red: int = 4,
) -> str:
    """Play an automatic BitBully-vs-BitBully game in GuiC4 and record it as an MP4.

    Parameters
    ----------
    out_path : str
        Output MP4 path.
    fps : int
        Frames per second for the video.
    sleep_between_moves : float
        Optional real-time delay between moves when capturing.
    max_moves : int
        Safety cap on number of half-moves (plies).
    max_depth_yellow : int
        Search depth for Yellow BitBully.
    max_depth_red : int
        Search depth for Red BitBully.

    Returns:
    -------
    str
        Path to the created MP4.
    """
    # -------------------------------------------------------------------------
    # 1. Set up BitBully agents and GuiC4
    # -------------------------------------------------------------------------
    yellow = BitBully(opening_book=None, tie_break="center", max_depth=max_depth_yellow)
    red = BitBully(opening_book=None, tie_break="center", max_depth=max_depth_red)

    agents = {
        "Yellow (BitBully)": yellow,
        "Red (BitBully)": red,
    }

    # Non-interactive use: autoplay=False; we drive _computer_move() manually
    c4gui = GuiC4(agents=agents, autoplay=False)

    # Make both sides agent-controlled
    c4gui.yellow_player = "Yellow (BitBully)"
    c4gui.red_player = "Red (BitBully)"
    c4gui._update_insert_buttons()

    frames: list[np.ndarray] = []

    # Initial frame (empty board)
    frames.append(_fig_to_rgb_array(c4gui.m_fig))

    # -------------------------------------------------------------------------
    # 2. Play game and capture after each move
    # -------------------------------------------------------------------------
    moves_played = 0
    while not c4gui.m_gameover and moves_played < max_moves:
        c4gui._computer_move()
        moves_played += 1

        frame = _fig_to_rgb_array(c4gui.m_fig)
        frames.append(frame)

        if sleep_between_moves > 0:
            time.sleep(sleep_between_moves)

    # -------------------------------------------------------------------------
    # 3. Save frames as MP4
    # -------------------------------------------------------------------------
    out_path = str(Path(out_path))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    imageio.mimsave(
        out_path,
        frames,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,  # avoid auto-resize to multiples of 16
    )

    return out_path


if __name__ == "__main__":
    mp4_path = generate_c4_animation_mp4(
        out_path="artifacts/c4_bitbully_game.mp4",
        fps=2,
        sleep_between_moves=0.0,
        max_moves=42,
        max_depth_yellow=4,
        max_depth_red=4,
    )
    print(f"Saved Connect-4 animation to: {mp4_path}")
