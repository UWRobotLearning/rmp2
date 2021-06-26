"""
helper functions for sampling from torus in 2d and 3d
"""

import numpy as np

def sample_from_torus_3d(random_generator, angle_center, angle_range, major_radius, minor_radius, height):
    while True:
        theta = random_generator.uniform(
            low=angle_center - angle_range / 2, 
            high=angle_center + angle_range / 2)
        psi = random_generator.uniform(low=-np.pi, high=np.pi)
        r = random_generator.uniform(low=0., high=minor_radius)
        w = random_generator.random()
        if w <= (major_radius + r * np.cos(psi)) / (major_radius + minor_radius) * r / minor_radius:
            return np.array([
                (major_radius + r * np.cos(psi)) * np.cos(theta),
                (major_radius + r * np.cos(psi)) * np.sin(theta),
                height + r * np.sin(psi)
            ])


def sample_from_torus_2d(random_generator, angle_center, angle_range, major_radius, minor_radius):
    while True:
        theta = random_generator.uniform(
            low=angle_center - angle_range / 2, 
            high=angle_center + angle_range / 2)
        r = random_generator.uniform(
            low=major_radius - minor_radius, 
            high=minor_radius + minor_radius)
        w = random_generator.random()
        if w <= r / (major_radius + minor_radius):
            return np.array([
                r * np.cos(theta),
                r * np.sin(theta),
            ])

