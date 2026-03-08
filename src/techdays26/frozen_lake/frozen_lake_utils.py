import pygame


def get_action_from_keyboard(key_to_action):
    """
    Waits for keyboard input and returns the corresponding action.
    Returns None if the window should be closed.
    """
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key in key_to_action:
                    return key_to_action[event.key]
        pygame.time.wait(10)


key_to_action = {
    pygame.K_LEFT: 0,
    pygame.K_DOWN: 1,
    pygame.K_RIGHT: 2,
    pygame.K_UP: 3,
}

action_to_arrow = {
    0: "←",
    1: "↓",
    2: "→",
    3: "↑",
}
