"""Base class for all robots."""


class Robot:
    """Robot Base

    A `Robot` is composed of joints described in the urdf.
    """

    def __init__(self, urdf_path: str = None,) -> None:
        """Constructs a base robot and resets it to the initial states.
        """
        self._urdf_path = urdf_path
        self._step_count = 0

    @property
    def step_count(self) -> int:
        return self._step_count

    def reset(self) -> None:
        self._step_count = 0

    def step(self) -> None:
        self._step_count += 1
