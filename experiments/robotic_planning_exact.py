import itertools
import numpy as np
import time
import os.path
import pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import collections  as mc

from citoolkit.specifications.dfa import Dfa

from citoolkit.improvisers.labelled_quantitative_ci import MaxEntropyLabelledQuantitativeCI

from fractions import Fraction

from citoolkit.labellingfunctions.labelling_dfa import LabellingDfa
from citoolkit.costfunctions.accumulated_cost_dfa import AccumulatedCostDfa

# Top left corner is (0,0)
# 0 denotes normal passable terrain
# 1 denotes impassable terrain
# 2 denotes the start point
# 3 denotes the end point
# 4 denotes label objectives. One label objective must be visited
# 5 denotes hard objective 1. All hard objectives must be visited
# 6 denotes hard objective 2. All hard objectives must be visited
# 7 denotes hard objective 3. All hard objectives must be visited
# 8 denotes hard objective 4. All hard objectives must be visited
GRIDWORLD =         (
                    (0, 0, 0, 0, 3, 0, 0, 0),
                    (0, 0, 0, 4, 0, 0, 6, 0),
                    (0, 0, 0, 0, 0, 0, 0, 0),
                    (1, 1, 0, 1, 0, 1, 1, 0),
                    (8, 4, 0, 0, 0, 0, 4, 0),
                    (0, 0, 0, 0, 0, 5, 0, 0),
                    (0, 0, 0, 0, 0, 0, 0, 0),
                    (7, 0, 0, 0, 2, 0, 0, 0)
                    )

GRIDWORLD_COSTS =   [
                    [2, 2, 2, 2, 0, 2, 2, 2],
                    [2, 3, 2, 0, 1, 2, 2, 2],
                    [2, 2, 2, 2, 1, 2, 2, 2],
                    [0, 0, 4, 0, 1, 0, 0, 4],
                    [2, 0, 2, 3, 1, 3, 3, 0],
                    [2, 2, 2, 2, 1, 0, 3, 2],
                    [2, 2, 2, 2, 1, 3, 3, 2],
                    [0, 2, 2, 2, 0, 2, 2, 2]
                    ]

alphabet = {"North", "East", "South", "West"}

length_bounds = (0,40)
epsilon = 1
prob_bounds = (0, 1)


def run():
    print("Starting Exact LQCI Robotic Planning Experiment...")
    print()
    start = time.time()

    if os.path.isfile("hard_constraint.pickle"):
        print("Loading hard constraint from pickle...\n")
        hard_constraint = pickle.load(open("hard_constraint.pickle", 'rb'))
    else:
        print("Creating hard constraint...\n")
        hard_constraint = create_hard_constraint()
        pickle.dump(hard_constraint, open("hard_constraint.pickle", "wb"))

    if os.path.isfile("label_function.pickle"):
        print("Loading label function from pickle...\n")
        label_function = pickle.load(open("label_function.pickle", 'rb'))
    else:
        print("Creating label function...\n")
        label_function = create_label_function()
        label_function.decompose()
        pickle.dump(label_function, open("label_function.pickle", "wb"))

    if os.path.isfile("cost_function.pickle"):
        print("Loading cost function from pickle...\n")
        cost_function = pickle.load(open("cost_function.pickle", 'rb'))
    else:
        print("Creating cost function...\n")
        cost_function = create_cost_function()
        cost_function.decompose()
        pickle.dump(cost_function, open("cost_function.pickle", "wb"))

    if os.path.isfile("me_improviser.pickle"):
        print("Loading Max Entropy improviser from pickle...\n")
        me_improviser = pickle.load(open("me_improviser.pickle", 'rb'))
    else:
        print("Creating Max Entropy improviser...\n")
        me_improviser = MaxEntropyLabelledQuantitativeCI(hard_constraint, cost_function, label_function, length_bounds, 30, (Fraction(1,6), Fraction(1,2)))
        pickle.dump(cost_function, open("me_improviser.pickle", "wb"))

    print("Done. Total time: " + str(time.time() - start))

def create_hard_constraint():
    max_y = len(GRIDWORLD) - 1
    max_x = len(GRIDWORLD[0]) - 1

    states = set()
    states.add("Sink")

    # Maps (x,y, hc_1, hc_2, hc_3, hc_4) to a state where
    # hc_# indicates that that hard constraint has been visited
    state_map = dict()

    for y in range(len(GRIDWORLD)):
        for x in range(len(GRIDWORLD[0])):
            for hc_objectives in itertools.product([0,1], repeat=4):
                new_state = "State_(" + str(x) + "," + str(y) + ")_" + str(hc_objectives)
                states.add(new_state)
                state_map[(x, y, hc_objectives)] = new_state

    start_loc = np.where(np.array(GRIDWORLD) == 2)
    end_loc = np.where(np.array(GRIDWORLD) == 3)

    start_loc = (start_loc[1][0], start_loc[0][0])
    end_loc = (end_loc[1][0], end_loc[0][0])

    accepting_states = {state_map[(end_loc[0], end_loc[1], (1,1,1,1))]}
    start_state = state_map[(start_loc[0], start_loc[1], (0,0,0,0))]

    transitions = {}

    for symbol in alphabet:
        transitions[("Sink", symbol)] = "Sink"

    hc_1_loc = np.where(np.array(GRIDWORLD) == 5)
    hc_1_loc = (hc_1_loc[1][0], hc_1_loc[0][0])

    hc_2_loc = np.where(np.array(GRIDWORLD) == 6)
    hc_2_loc = (hc_2_loc[1][0], hc_2_loc[0][0])

    hc_3_loc = np.where(np.array(GRIDWORLD) == 7)
    hc_3_loc = (hc_3_loc[1][0], hc_3_loc[0][0])

    hc_4_loc = np.where(np.array(GRIDWORLD) == 8)
    hc_4_loc = (hc_4_loc[1][0], hc_4_loc[0][0])

    for y in range(len(GRIDWORLD)):
        for x in range(len(GRIDWORLD[0])):
            for hc_objectives in itertools.product([0,1], repeat=4):
                for symbol in alphabet:
                    hc_1, hc_2, hc_3, hc_4 = hc_objectives

                    origin_state = state_map[(x, y, hc_objectives)]

                    if symbol == "North":
                        dest_x = x
                        dest_y = y - 1
                    elif symbol == "East":
                        dest_x = x + 1
                        dest_y = y
                    elif symbol == "South":
                        dest_x = x
                        dest_y = y + 1
                    elif symbol == "West":
                        dest_x = x - 1
                        dest_y = y

                    dest_coords = (dest_x, dest_y)

                    # First check map boundaries and impassable areas.
                    if dest_x < 0 or dest_x > max_x or dest_y < 0 or dest_y > max_y:
                        # Off the edge of the map
                        destination_state = "Sink"
                    elif GRIDWORLD[y][x] == 1:
                        # In impassable area
                        destination_state = "Sink"
                    elif GRIDWORLD[dest_y][dest_x] == 1:
                        # Into impassable area
                        destination_state = "Sink"
                    else:
                        # Second check if at hard constraint objective.
                        if (dest_coords) == hc_1_loc:
                            destination_state = state_map[(dest_x, dest_y, (1, hc_2, hc_3, hc_4))]
                        elif (dest_coords) == hc_2_loc:
                            destination_state = state_map[(dest_x, dest_y, (hc_1, 1, hc_3, hc_4))]
                        elif (dest_coords) == hc_3_loc:
                            destination_state = state_map[(dest_x, dest_y, (hc_1, hc_2, 1, hc_4))]
                        elif (dest_coords) == hc_4_loc:
                            destination_state = state_map[(dest_x, dest_y, (hc_1, hc_2, hc_3, 1))]
                        else:
                            # Not at a hard constraint, hc vals stay the same
                            destination_state = state_map[(dest_x, dest_y, (hc_1, hc_2, hc_3, hc_4))]

                    transitions[(origin_state, symbol)] = destination_state

    hard_constraint = Dfa(alphabet, states, accepting_states, start_state, transitions).minimize()

    return hard_constraint

def create_label_function():
    max_y = len(GRIDWORLD) - 1
    max_x = len(GRIDWORLD[0]) - 1

    states = set()
    states.add("Sink")

    state_map = dict()

    for y in range(len(GRIDWORLD)):
        for x in range(len(GRIDWORLD[0])):
            new_state = "State_(" + str(x) + "," + str(y) + ")"
            states.add(new_state)
            state_map[(x, y)] = new_state

    start_loc = np.where(np.array(GRIDWORLD) == 2)
    start_loc = (start_loc[1][0], start_loc[0][0])

    lo_loc = np.where(np.array(GRIDWORLD) == 4)
    lo_locs = []

    for i in range(len(lo_loc[0])):
        lo_locs.append((lo_loc[1][i], lo_loc[0][i]))

    accepting_states = {state_map[coords[0], coords[1]] for coords in lo_locs}
    start_state = state_map[(start_loc[0], start_loc[1])]

    transitions = {}

    for symbol in alphabet:
        transitions[("Sink", symbol)] = "Sink"

    for y in range(len(GRIDWORLD)):
        for x in range(len(GRIDWORLD[0])):
            for symbol in alphabet:
                origin_state = state_map[(x, y)]

                if symbol == "North":
                    dest_x = x
                    dest_y = y - 1
                elif symbol == "East":
                    dest_x = x + 1
                    dest_y = y
                elif symbol == "South":
                    dest_x = x
                    dest_y = y + 1
                elif symbol == "West":
                    dest_x = x - 1
                    dest_y = y

                dest_coords = (dest_x, dest_y)

                # First check map boundaries and impassable areas.
                if dest_x < 0 or dest_x > max_x or dest_y < 0 or dest_y > max_y:
                    # Off the edge of the map
                    destination_state = "Sink"
                elif (x,y) in lo_locs:
                    destination_state = origin_state
                else:
                    destination_state = state_map[(dest_x, dest_y)]

                transitions[(origin_state, symbol)] = destination_state

    label_dfa = Dfa(alphabet, states, accepting_states, start_state, transitions)

    label_map = dict()

    for label_iter, label_coords in enumerate(lo_locs):
        l_x, l_y = label_coords
        label_map[state_map[l_x, l_y]] = "Label" + str(label_iter)

    label_func = LabellingDfa(label_dfa, label_map)

    return label_func

def create_cost_function():
    max_y = len(GRIDWORLD) - 1
    max_x = len(GRIDWORLD[0]) - 1

    states = set()

    state_map = dict()

    for y in range(len(GRIDWORLD)):
        for x in range(len(GRIDWORLD[0])):
            new_state = "State_(" + str(x) + "," + str(y) + ")"
            states.add(new_state)
            state_map[(x, y)] = new_state

    start_loc = np.where(np.array(GRIDWORLD) == 2)
    start_loc = (start_loc[1][0], start_loc[0][0])

    accepting_states = frozenset(states)
    states.add("Sink")
    start_state = state_map[(start_loc[0], start_loc[1])]

    transitions = {}

    for symbol in alphabet:
        transitions[("Sink", symbol)] = "Sink"

    for y in range(len(GRIDWORLD)):
        for x in range(len(GRIDWORLD[0])):
            for symbol in alphabet:
                origin_state = state_map[(x, y)]

                if symbol == "North":
                    dest_x = x
                    dest_y = y - 1
                elif symbol == "East":
                    dest_x = x + 1
                    dest_y = y
                elif symbol == "South":
                    dest_x = x
                    dest_y = y + 1
                elif symbol == "West":
                    dest_x = x - 1
                    dest_y = y

                dest_coords = (dest_x, dest_y)

                # First check map boundaries and impassable areas.
                if dest_x < 0 or dest_x > max_x or dest_y < 0 or dest_y > max_y:
                    # Off the edge of the map
                    destination_state = "Sink"
                else:
                    destination_state = state_map[(dest_x, dest_y)]

                transitions[(origin_state, symbol)] = destination_state

    cost_dfa = Dfa(alphabet, states, accepting_states, start_state, transitions)

    cost_map = dict()

    for y in range(len(GRIDWORLD)):
        for x in range(len(GRIDWORLD[0])):
            cost_map[state_map[(x, y)]] = GRIDWORLD_COSTS[y][x]

    cost_map["Sink"] = 9

    cost_func = AccumulatedCostDfa(cost_dfa, cost_map, max_word_length=length_bounds[1])

    return cost_func

def draw_improvisation(improvisation):
    fig, ax = plt.subplots(1, 1, tight_layout=True)

    for x in range(len(GRIDWORLD) + 1):
        ax.axhline(x, lw=2, color='k', zorder=5)
        ax.axvline(x, lw=2, color='k', zorder=5)

    cmap = colors.ListedColormap(['white', '#000000','grey', 'grey', 'orange', 'darkblue'])
    boundaries = [0, 1, 2, 3, 4, 5, 9]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    ax.imshow(GRIDWORLD, interpolation='none', extent=[0, len(GRIDWORLD), 0, len(GRIDWORLD)], zorder=0, cmap=cmap, norm=norm)

    #ax.axis('off')

    start_loc = np.where(np.array(GRIDWORLD) == 2)

    point_a = None
    point_b = (start_loc[1][0] + 0.5, len(GRIDWORLD) - 1 - start_loc[0][0] + 0.5)

    lines = []

    for symbol in improvisation:
        point_a = point_b

        if symbol == "North":
            point_b = (point_a[0], point_a[1] + 1)
        elif symbol == "East":
            point_b = (point_a[0] + 1, point_a[1])
        elif symbol == "South":
            point_b = (point_a[0], point_a[1] - 1)
        elif symbol == "West":
            point_b = (point_a[0] - 1, point_a[1])

        lines.append([point_a, point_b])

    lc = mc.LineCollection(lines, color="red", linewidths=2)
    ax.add_collection(lc)

    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    run()
    #draw_improvisation(["North", "East", "North", "West", "North"])
