"""Unit tests for the motor model class."""
import pytest
from robolearn.robots.robot import Robot


# This is a test fixture for an instance of Robot
@pytest.fixture
def robot():
    return Robot(urdf_path="./data/urdfs")


# This is a single, parameterized test.
# To see how tests work, read:
# https://docs.pytest.org/en/6.2.x/contents.html
@pytest.mark.parametrize(
    "on_rack, reset",
    [
        pytest.param(True, True, id="0",),
        pytest.param(True, False, id="1",),
        pytest.param(False, True, id="2",),
        pytest.param(False, False, id="3",),
    ],
)
def test_reset(on_rack, reset):
    # obviously all of this testing is bogus,
    # replace the logic here.
    if reset:
        assert True
    if on_rack:
        assert True
    assert True


def test_step(robot):
    robot.reset()
    robot.step()
    assert robot.step_count == 1
