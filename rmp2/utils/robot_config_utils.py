import yaml
import os

def load_robot_config(robot_name=None, config_path=None):
    if config_path is None:
        if robot_name == 'franka':
            config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'franka_config.yaml')
        elif robot_name == '3link':
            config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', '3link_config.yaml')
        else:
            raise ValueError

    print(config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)
    if robot_name is not None:
        assert(robot_name == config['robot_name'])
    else:
        robot_name = config['robot_name']

    return robot_name, config


def get_robot_urdf_path(robot_name):
    if robot_name == 'franka':
        urdf_path = os.path.join(os.path.dirname(__file__), '..', 'urdf', 'panda.urdf')
    elif robot_name == '3link':
        urdf_path = os.path.join(os.path.dirname(__file__), '..', 'urdf', 'three_link_planar_robot.urdf')
    else:
        raise ValueError

    return urdf_path

def get_robot_eef_uid(robot_name):
    if robot_name == "franka":
        eef_uid = 14
    elif robot_name == "3link":
        eef_uid = 6
    else:
        raise ValueError
    return eef_uid
