import numpy as np
from matplotlib import pyplot as plt

def plot_robot_2D(robot, q, lw=2, handle_list=None, link_order=None):
    link_positions = robot.forward_all(q)
    num_links = len(link_positions)

    if link_order is None:
        link_names = [link_name for link_name in robot.link_names]
        link_order = [(link_names[i], link_names[i + 1]) for i in range(num_links - 1)]

    if handle_list is None:
        handle_list = []

        for curr_link_names in link_order:
            pt1 = [link_positions[curr_link_names[0]][0], link_positions[curr_link_names[0]][1]]
            pt2 = [link_positions[curr_link_names[1]][0], link_positions[curr_link_names[1]][1]]

            h1 = plt.plot(pt1[0], pt1[1], marker='o', color='black', linewidth=3 * lw)
            handle_list.append(h1[0])
            h2 = plt.plot(pt2[0], pt2[1], marker='o', color='black', linewidth=3 * lw)
            handle_list.append(h2[0])
            h3 = plt.plot(np.array([pt1[0], pt2[0]]), np.array([pt1[1], pt2[1]]), color='blue', linewidth=lw)
            handle_list.append(h3[0])
    else:
        m = 0
        for curr_link_names in link_order:
            pt1 = [link_positions[curr_link_names[0]][0], link_positions[curr_link_names[0]][1]]
            pt2 = [link_positions[curr_link_names[1]][0], link_positions[curr_link_names[1]][1]]

            handle_list[m].set_data(pt1[0], pt1[1])
            m += 1
            handle_list[m].set_data(pt2[0], pt2[1])
            m += 1
            handle_list[m].set_data(np.array([pt1[0], pt2[0]]), np.array([pt1[1], pt2[1]]))
            m += 1
    return handle_list,