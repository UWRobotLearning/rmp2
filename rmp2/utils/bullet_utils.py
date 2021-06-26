"""
helper functions for pybullet
"""

import pybullet as p

def add_goal(bullet_client, position, radius=0.05, color=[0.0, 1.0, 0.0, 1]):
    collision = -1
    visual = bullet_client.createVisualShape(p.GEOM_SPHERE, radius=radius,
                                             rgbaColor=color)
    goal = bullet_client.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=collision,
                                    baseVisualShapeIndex=visual,
                                    basePosition=position)
    return goal

def add_collision_goal(bullet_client, position, radius=0.05, color=[0.0, 1.0, 0.0, 1]):
    collision = bullet_client.createCollisionShape(p.GEOM_SPHERE, radius=0.01)
    visual = -1
    goal = bullet_client.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=collision,
                                    baseVisualShapeIndex=visual,
                                    basePosition=position)
    return goal


def add_obstacle_ball(bullet_client, center, radius=0.1, color=[0.4, 0.4, 0.4, 1]):
    collision = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius,
                                             rgbaColor=color)
    obstacle = p.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=collision,
                                    baseVisualShapeIndex=visual,
                                    basePosition=center)
    return obstacle


def add_obstacle_cylinder(bullet_client, center, radius=0.1, length=0.1, color=[0.4, 0.4, 0.4, 1]):
    collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=length)
    visual = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=length, 
                                             rgbaColor=color)
    obstacle = p.createMultiBody(baseMass=0,
                                    baseCollisionShapeIndex=collision,
                                    baseVisualShapeIndex=visual,
                                    basePosition=center)
    return obstacle