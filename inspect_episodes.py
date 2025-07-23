#!/usr/bin/env python3

import numpy as np

import cv2
from PIL import Image

import matplotlib.pyplot as plt

import tools

train_eps = tools.load_episodes("/home/adam/adam/1_projects/labyrinth/minotaur/logdir/minotaur_briowhiteeasy_29/train_eps/", limit=1)
generator = tools.sample_episodes(train_eps, 64)
dataset = tools.from_generator(generator, 1)

ep_original = next(dataset)
ep_original = next(dataset)
ep_original = next(dataset)
ep_original = next(dataset)
ep_original = next(dataset)
while True:
    print("NEXT")
    ep_original = next(dataset)

    # ep_original = np.load("home/adam/adam/1_projects/labyrinth/minotaur/logdir/minotaur_briowhiteeasy_29/train_eps/20250722T043838-26cd974c43254f28a0c87d75d850eab9_0-304.npz")
    # ep_flip_x = np.load("home/adam/adam/1_projects/labyrinth/minotaur/logdir/minotaur_briowhiteeasy_29/train_eps/20250722T043838-26cd974c43254f28a0c87d75d850eab9_2-304.npz")
    list(ep_original.keys())
    ep = ep_original
    ep = {k: v[0] for k, v in ep.items()}

    plt.figure()
    plt.plot(ep["position"][:][:, 0], ep["position"][:][:, 1] * -1 + 1, label="position")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect(0.235/0.28)
    plt.legend()

    plt.figure()

    plt.plot(ep["position"][:][:, 0], label="x")
    plt.plot(ep["position"][:][:, 1] * -1 + 1, label="y")
    plt.plot(ep["board_angle"], label="board angle")
    plt.plot(ep["action"], label="action")
    plt.plot(ep["reward"], label="reward")
    plt.legend()

    plt.show()
    plt.close()

    for i in range(0, 64, 1):
        img = ep["image"][i]
        pos = ep["position"][i]
        closest_points = ep["closest_points"][i]
        board_angle = ep["board_angle"][i]
        action = ep["action"][i]
        reward = ep["reward"][i]
        discount = ep["discount"][i]

        print(f"{reward=}")
        print(f"{discount=}")
        plt.figure()
        plt.scatter(pos[0], pos[1] * -1 + 1, label="position")
        plt.scatter(closest_points[::2], closest_points[1::2] * -1 + 1, label="closest points")
        plt.gca().annotate("", xytext=(0.5, 0.5), xy=board_angle * (-1, 1) + 0.5, arrowprops=dict(arrowstyle="->"), label="board angle")
        plt.gca().annotate("", xytext=(0.5, 0.5), xy=action * (0.5, 0.5) + 0.5, arrowprops=dict(arrowstyle="->", color="red"), c=(1, 0, 0), label="action")

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.gca().set_aspect('equal')
        plt.legend()
        plt.ion()
        plt.show()
        cv2.imshow("frame", img)
        cv2.waitKey(0)
        plt.close()