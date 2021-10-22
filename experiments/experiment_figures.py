import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import collections  as mc
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
import random

def draw_figure(improvisations, GRIDWORLD, GRIDWORLD_COSTS):
    fig, ax = plt.subplots(1, 1, tight_layout=True)

    for x in range(len(GRIDWORLD) + 1):
        ax.axhline(x, lw=2, color='k', zorder=0, alpha=0.5)
        ax.axvline(x, lw=2, color='k', zorder=0, alpha=0.5)

    ax.imshow(GRIDWORLD_COSTS, cmap="binary", interpolation='none', extent=[0, len(GRIDWORLD), 0, len(GRIDWORLD)], alpha=0.5)

    impassable = 1
    start = 2
    end = 3
    hard_objectives = [4,5,6,7]
    label_objectives = [8,9,10]

    for x in range(len(GRIDWORLD)):
        for y in range(len(GRIDWORLD)):
            if GRIDWORLD[y][x] == impassable:
                ax.add_patch(PathPatch(TextPath((x + 0.13, len(GRIDWORLD) - 1 - y + 0.11), "X", size=1), color="red", ec="white", alpha=0.5))
            elif GRIDWORLD[y][x] == start:
                ax.add_patch(PathPatch(TextPath((x + 0.2, len(GRIDWORLD) - 1 - y + 0.11), "S", size=1), color="green", ec="white", alpha=0.5))
            elif GRIDWORLD[y][x] == end:
                ax.add_patch(PathPatch(TextPath((x + 0.17, len(GRIDWORLD) - 1 - y + 0.11), "E", size=1), color="green", ec="white", alpha=0.5))
            elif GRIDWORLD[y][x] in hard_objectives:
                ax.add_patch(PathPatch(TextPath((x + 0.11, len(GRIDWORLD) - 1 - y + 0.11), "O", size=1), color="blue", ec="white", alpha=0.5))
            elif GRIDWORLD[y][x] in label_objectives:
                ax.add_patch(PathPatch(TextPath((x + 0.13, len(GRIDWORLD) - 1 - y + 0.11), "C", size=1), color="orange", ec="white", alpha=0.5))


    start_loc = np.where(np.array(GRIDWORLD) == 2)
    start_loc = (start_loc[1][0], start_loc[0][0])

    colors = ["c", "m", "y"]

    # Draw paths
    for i in range(len(improvisations)):
        improvisation = improvisations[i]

        point_a = None
        point_b = (start_loc[0] + 0.25*(i+1),  len(GRIDWORLD) - 1 - start_loc[1] + 0.25*(i+1))

        lines = []

        print(improvisation)

        for direction in improvisation:
            print(direction)
            point_a = point_b

            if direction is None:
                break

            if direction == "North":
                movement = [0,1]
            elif direction == "East":
                movement = [1,0]
            elif direction == "South":
                movement = [0,-1]
            elif direction == "West":
                movement = [-1,0]
            else:
                assert False

            point_b = (point_a[0] + movement[0], point_b[1] + movement[1])

            lines.append([point_a, point_b])

        lc = mc.LineCollection(lines, color=colors[i], ls="-", linewidths=5)
        ax.add_collection(lc)

    plt.axis('off')

    plt.savefig("ExampleTraces.png", bbox_inches="tight", pad_inches=0, dpi=1024)

if __name__ == '__main__':
    improvisations = random.choices(pickle.load(open("qci_fig_samples.pickle", 'rb')), k=3)

    GRIDWORLD =         (
                        (8, 0, 3, 0, 5,  0),
                        (0, 0, 0, 0, 0,  0),
                        (0, 1, 0, 1, 0,  1),
                        (7, 0, 0, 0, 0, 10),
                        (0, 9, 0, 4, 0,  0),
                        (6, 0, 2, 0, 0,  0)
                        )

    GRIDWORLD_COSTS =   (
                        (3, 2, 0, 2, 0, 2),
                        (2, 1, 0, 1, 1, 1),
                        (3, 0, 0, 0, 3, 0),
                        (0, 1, 0, 1, 2, 1),
                        (2, 0, 0, 0, 1, 2),
                        (0, 1, 0, 1, 1, 1)
                        )

    draw_figure(improvisations, GRIDWORLD, GRIDWORLD_COSTS)
