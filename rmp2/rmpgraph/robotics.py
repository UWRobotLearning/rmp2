
from rmp2.rmpgraph.rmpgraph import RMPGraph
from rmp2.kinematics import Robot
from rmp2.rmpgraph.rmps import get_rmp, ExternalCanonicalRMP
from rmp2.rmpgraph.taskmaps import dist2balls, distmap
from rmp2.utils.robot_config_utils import load_robot_config, get_robot_urdf_path
import tensorflow as tf

ROBOT_RMPS = [
    'cspace_target_rmp',
    'joint_limit_rmp',
    'joint_velocity_cap_rmp',
    'target_rmp',
    'collision_rmp',
    'damping_rmp',
    'external_rmps'
    ]


def get_control_points(link_positions, arm_collision_controllers):
    control_points = []
    for controller in arm_collision_controllers:
        end1, end2 = controller['segment']
        interpolation_points = controller['interpolation_pts']
        for k in range(interpolation_points):
            alpha = 1. * (k + 1) / interpolation_points
            point = alpha * link_positions[end2] + (1 - alpha) * link_positions[end1]
            control_points.append(point)
    control_points = tf.stack(control_points, axis=1)
    return control_points



class RobotRMPGraph(RMPGraph):
    def __init__(self, robot_name=None, config_path=None, config=None, workspace_dim=3, dtype=tf.float32, rmp_type='canonical', timed=False, offset=1e-3, name='robot'):
        """
        documentation....
        """
        assert(robot_name is not None or config_path is not None or config is not None)
        assert(config_path is None or config is None)
        # set up robot
        if config is None:
            robot_name, config = load_robot_config(robot_name=robot_name, config_path=config_path)
        elif robot_name is not None:
            assert robot_name == config['robot_name']
        else:
            robot_name = config['robot_name']
            
        urdf_path = get_robot_urdf_path(robot_name)

        self.robot = Robot(urdf_path, workspace_dim=workspace_dim, dtype=dtype)
        self.eef_link = config['eef_link']
        self.workspace_dim = workspace_dim
        self.cspace_dim = self.robot.cspace_dim
        self.dtype = dtype

        self.joint_lower_limits = self.robot.joint_lower_limits + config['joint_limit_buffers']
        self.joint_upper_limits = self.robot.joint_upper_limits - config['joint_limit_buffers']
        self.joint_velocity_limits = self.robot.joint_velocity_limits

        self.default_config = tf.constant(config['default_q'], dtype=self.dtype)

        self.rmp_config = config['rmp_params']

        # set up RMPs
        rmps = []
        self.keys = []

        for key in ROBOT_RMPS:
            if key in self.rmp_config:
                if key is not 'external_rmps':
                    rmps.append(get_rmp(key, self.rmp_config[key], dtype=dtype))
                    self.keys.append(key)
                else:
                    for external_rmp in self.rmp_config[key]:
                        rmps.append(ExternalCanonicalRMP(**external_rmp, dtype=dtype))
                        self.keys.append('external_rmp')
        
        self.arm_collision_controllers = config['arm_collision_controllers']

        arm_collision_radii = []
        for arm_controller in self.arm_collision_controllers:
            arm_collision_radii += [arm_controller['radius']] * arm_controller['interpolation_pts']

        self.arm_collision_radii = tf.expand_dims(tf.constant(arm_collision_radii, dtype=dtype), 0)

        body_obstacles = dict()
        body_obstacle_box_mins = []
        body_obstacle_box_maxs = []
        body_obstacle_ball_centers = []
        body_obstacle_ball_radii = []
        for body_obstacle in config['body_obstacles']:
            if body_obstacle['type'] == 'box':
                body_obstacle_box_mins.append(body_obstacle['mins'])
                body_obstacle_box_maxs.append(body_obstacle['maxs'])
            elif body_obstacle['type'] == 'ball':
                body_obstacle_ball_centers.append(body_obstacle['center'])
                body_obstacle_ball_radii.append(body_obstacle['radius'])

        if len(body_obstacle_box_mins):
            body_obstacle_box_mins = tf.convert_to_tensor([body_obstacle_box_mins], dtype=dtype)
            body_obstacle_box_maxs = tf.convert_to_tensor([body_obstacle_box_maxs], dtype=dtype)

            body_obstacles['box'] = {
                'obstacle_mins': body_obstacle_box_mins, 
                'obstacle_maxs': body_obstacle_box_maxs
            }

        if len(body_obstacle_ball_centers):
            body_obstacle_ball_centers = tf.convert_to_tensor([body_obstacle_ball_centers], dtype=dtype)
            body_obstacle_ball_radii = tf.convert_to_tensor([body_obstacle_ball_radii], dtype=dtype)

            body_obstacles['ball'] = {
                'obstacle_centers': body_obstacle_ball_centers,
                'obstacle_radii': body_obstacle_ball_radii
            }

        self.body_obstacles = body_obstacles

        super().__init__(rmps, rmp_type=rmp_type, timed=timed, dtype=dtype, name=name)


    def forward_mapping(self, q, **features):
        """
        forward mapping from root node to leaf nodes given environment features
        --------------------------------------------
        :param q: root node coordinate
        :param features: environment features, lists/dicts, e.g. goals, obstacles, etc.
        :return xs: list of leaf node coordinates
        """

        batch_size, input_dim = q.shape
        x = []

        if input_dim < self.cspace_dim:
            q = tf.concat([q, tf.zeros((batch_size, self.cspace_dim - input_dim), dtype=self.dtype)], axis=1)

        # forward kinematics
        link_positions = self.robot.forward_kinematics(q)

        # ----------------------------
        #         task spaces
        # ----------------------------

        for i, key in enumerate(self.keys):
            # ----------------------------
            # default configuration
            if key == 'cspace_target_rmp':
                x.append(q - self.default_config)

            # ----------------------------
            # joint limits
            elif key == 'joint_limit_rmp':
                x_upper = self.joint_upper_limits - q
                x_lower = q - self.joint_lower_limits
                x_joint_limit = tf.reshape(tf.concat([x_upper, x_lower], axis=1), (-1, 1))
                x.append(x_joint_limit)

            # ----------------------------
            # joint velocity limits
            elif key == 'joint_velocity_cap_rmp':
                x_velocity_cap = tf.reshape(q, (-1, 1))
                x.append(x_velocity_cap)

            # ----------------------------
            # goal attractor
            elif key == 'target_rmp':
                eef_pos = link_positions[self.eef_link]
                x.append(eef_pos * 1.)
            
            # ----------------------------
            # collision avoidance
            elif key == 'collision_rmp':
                control_points = get_control_points(link_positions, self.arm_collision_controllers)

                obstacles = features['obstacles']
                obstacles = tf.reshape(obstacles, (batch_size, -1, self.workspace_dim + 1))
                obstacle_centers = obstacles[:, :, :self.workspace_dim]
                obstacle_radii = obstacles[:, :, -1]

                obstacle_dists = dist2balls(
                    control_points, 
                    self.arm_collision_radii, 
                    obstacle_centers, obstacle_radii)

                for body_obstacle_type, body_obstacle in self.body_obstacles.items():
                    body_obstacle_dists = distmap(
                        body_obstacle_type,
                        control_points,
                        self.arm_collision_radii,
                        **body_obstacle
                    )
                    obstacle_dists = tf.concat([obstacle_dists, body_obstacle_dists], axis=0)

                x.append(obstacle_dists)

            # ----------------------------
            # joint damping
            elif key == 'damping_rmp':
                x.append(q * 1.)

            # ----------------------------
            # neural network rmp
            elif key == 'external_rmp':
                link = self.rmps[i].link
                if link == 'joint':
                    x.append(q * 1.)
                else:
                    x.append(link_positions[link] * 1.)

            else:
                raise ValueError

        return x