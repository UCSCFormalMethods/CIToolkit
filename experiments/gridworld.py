import itertools
import numpy as np
import time
import os.path
import pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import collections  as mc
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch

import multiprocess

from citoolkit.specifications.dfa import Dfa

from citoolkit.improvisers.labelled_quantitative_ci import LabelledQuantitativeCI, MaxEntropyLabelledQuantitativeCI
from citoolkit.improvisers.quantitative_ci import QuantitativeCI

from fractions import Fraction

from citoolkit.labellingfunctions.labelling_dfa import LabellingDfa
from citoolkit.costfunctions.accumulated_cost_dfa import AccumulatedCostDfa

# Top left corner is (0,0)
# 0 denotes normal passable terrain
# 1 denotes impassable terrain
# 2 denotes the start point
# 3 denotes the end point
# 4 denotes hard objective 1. All hard objectives must be visited
# 5 denotes hard objective 2. All hard objectives must be visited
# 6 denotes hard objective 3. All hard objectives must be visited
# 7 denotes hard objective 4. All hard objectives must be visited
# 8 denotes label objective 0. The selected label objective must be visited first.
# 9 denotes label objective 1. The selected label objective must be visited first.
# 10 denotes label objective 2. The selected label objective must be visited first.
LARGE_GRIDWORLD =       (
                        (8, 0, 0, 3, 0, 0,  0),
                        (0, 0, 0, 0, 0, 5,  0),
                        (0, 0, 0, 0, 0, 0,  0),
                        (1, 0, 1, 0, 1, 0,  1),
                        (7, 0, 0, 0, 0, 0, 10),
                        (0, 0, 9, 0, 0, 4,  0),
                        (6, 0, 0, 2, 0, 0,  0)
                        )

LARGE_GRIDWORLD_COSTS = (
                        ( 9,  6,  3,  0,  4,  4,  4),
                        ( 6,  3,  2,  1,  4,  2,  4),
                        ( 3,  2,  2,  1,  2,  2,  2),
                        ( 0, 10,  0,  1,  0, 10,  0),
                        ( 2,  2,  2,  1,  4,  4,  2),
                        ( 2,  4,  1,  1,  2,  4,  4),
                        ( 2,  2,  2,  0,  2,  2,  2)
                        )

LARGE_LENGTH_BOUNDS = (1,30)

SMALL_GRIDWORLD =       (
                        (8, 0, 3, 0, 5,  0),
                        (0, 0, 0, 0, 0,  0),
                        (0, 1, 0, 1, 0,  1),
                        (7, 0, 0, 0, 0, 10),
                        (0, 9, 0, 4, 0,  0),
                        (6, 0, 2, 0, 0,  0)
                        )

SMALL_GRIDWORLD_COSTS = (
                        (3, 2, 0, 2, 0, 2),
                        (2, 1, 0, 1, 1, 1),
                        (3, 0, 0, 0, 3, 0),
                        (0, 1, 0, 1, 2, 1),
                        (2, 0, 0, 0, 1, 2),
                        (0, 1, 0, 1, 1, 1)
                        )

SMALL_LENGTH_BOUNDS = (1,25)

def get_gridworld_max_cost(gridworld, gridworld_costs, length_bounds):
    alphabet = ["North", "East", "South", "West"]
    max_y = len(gridworld) - 1
    max_x = len(gridworld[0]) - 1

    states = set()

    state_map = dict()

    for y in range(len(gridworld)):
        for x in range(len(gridworld[0])):
            new_state = "State_(" + str(x) + "," + str(y) + ")"
            states.add(new_state)
            state_map[(x, y)] = new_state

    start_loc = np.where(np.array(gridworld) == 2)
    end_loc = np.where(np.array(gridworld) == 3)

    start_loc = (start_loc[1][0], start_loc[0][0])
    end_loc = (end_loc[1][0], end_loc[0][0])

    states.add("Sink")
    accepting_states = {state_map[(end_loc[0], end_loc[1])]}
    start_state = state_map[(start_loc[0], start_loc[1])]
    transitions = {}

    for symbol in alphabet:
        transitions[("Sink", symbol)] = "Sink"

    for y in range(len(gridworld)):
        for x in range(len(gridworld[0])):
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

    for y in range(len(gridworld)):
        for x in range(len(gridworld[0])):
            cost_map[state_map[(x, y)]] = gridworld[y][x]

    cost_map["Sink"] = 0

    cost_func = AccumulatedCostDfa(cost_dfa, cost_map, max_word_length=length_bounds[1])

    return max(cost_func.costs)


def gridworld_to_dfa(gridworld, gridworld_costs, length_bounds):
    alphabet = ["North", "East", "South", "West", "Charge"]

    max_y = len(gridworld) - 1
    max_x = len(gridworld[0]) - 1

    max_cost = get_gridworld_max_cost(gridworld, gridworld_costs, length_bounds)


    start_loc = np.where(np.array(gridworld) == 2)
    end_loc = np.where(np.array(gridworld) == 3)

    start_loc = (start_loc[1][0], start_loc[0][0])
    end_loc = (end_loc[1][0], end_loc[0][0])


    hc_1_loc = np.where(np.array(gridworld) == 4)
    hc_1_loc = (hc_1_loc[1][0], hc_1_loc[0][0])

    hc_2_loc = np.where(np.array(gridworld) == 5)
    hc_2_loc = (hc_2_loc[1][0], hc_2_loc[0][0])

    hc_3_loc = np.where(np.array(gridworld) == 6)
    hc_3_loc = (hc_3_loc[1][0], hc_3_loc[0][0])

    hc_4_loc = np.where(np.array(gridworld) == 7)
    hc_4_loc = (hc_4_loc[1][0], hc_4_loc[0][0])


    lc_1_loc = np.where(np.array(gridworld) == 8)
    lc_1_loc = (lc_1_loc[1][0], lc_1_loc[0][0])

    lc_2_loc = np.where(np.array(gridworld) == 9)
    lc_2_loc = (lc_2_loc[1][0], lc_2_loc[0][0])

    lc_3_loc = np.where(np.array(gridworld) == 10)
    lc_3_loc = (lc_3_loc[1][0], lc_3_loc[0][0])

    lo_locs = [lc_1_loc, lc_2_loc, lc_3_loc]

    ## STATES CREATION ##

    states = set()
    states.add("Sink")

    # Maps (x, y, cost, (hc_1, hc_2, hc_3, hc_4), l_#) to a state where
    # cost is the current accumulated cost.
    # hc_# indicates that that hard constraint has been visited.
    # l_# == 0 indicated the robot has not charged yet.
    # l_# in (1,2,3) indicates that the robot has charged at charging point #.
    state_map = dict()

    # for y in range(len(gridworld)):
    #     for x in range(len(gridworld[0])):
    #         for cost in range(max_cost+1):
    #             for hc_objectives in itertools.product([0,1], repeat=4):
    #                 for label_num in range(4):
    #                     new_state = "State_(" + str(x) + "," + str(y) + ")_" + str(cost) + "_" + str(hc_objectives) + "_" + str(label_num)
    #                     states.add(new_state)
    #                     state_map[(x, y, cost, hc_objectives, label_num)] = new_state

    def get_state(key):
        if key in state_map:
            return state_map[key]
        else:
            x, y, cost, hc_objectives, label_num = key
            new_state = "State_(" + str(x) + "," + str(y) + ")_" + str(cost) + "_" + str(hc_objectives) + "_" + str(label_num)
            state_map[key] = new_state
            return state_map[key]

    start_state = get_state((start_loc[0], start_loc[1], 0, (0,0,0,0), 0))

    ## TRANSITION MAP CREATION ##

    transitions = {}

    # Make Sink trap
    for symbol in alphabet:
        transitions[("Sink", symbol)] = "Sink"

    # Add movement transitions
    for y in range(len(gridworld)):
        for x in range(len(gridworld[0])):
            for cost in range(max_cost+1):
                for hc_objectives in itertools.product([0,1], repeat=4):
                    for label_num in range(4):
                        for symbol in ["North", "East", "South", "West"]:
                            hc_1, hc_2, hc_3, hc_4 = hc_objectives

                            origin_state = get_state((x, y, cost, hc_objectives, label_num))

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


                            # Check map boundaries, impassable areas, and exceeded cost.
                            if dest_x < 0 or dest_x > max_x or dest_y < 0 or dest_y > max_y:
                                # Off the edge of the map
                                destination_state = "Sink"
                            elif gridworld[y][x] == 1:
                                # In impassable area
                                destination_state = "Sink"
                            elif gridworld[dest_y][dest_x] == 1:
                                # Into impassable area
                                destination_state = "Sink"
                            else:
                                # Get new cost
                                new_cost = cost + gridworld_costs[dest_y][dest_x]

                                if new_cost > max_cost:
                                    # Cost impossible
                                    destination_state = "Sink"
                                else:
                                    # Get new hc status
                                    if (dest_coords) == hc_1_loc:
                                        new_hc = (1, hc_2, hc_3, hc_4)
                                    elif (dest_coords) == hc_2_loc:
                                        new_hc = (hc_1, 1, hc_3, hc_4)
                                    elif (dest_coords) == hc_3_loc:
                                        new_hc = (hc_1, hc_2, 1, hc_4)
                                    elif (dest_coords) == hc_4_loc:
                                        new_hc = (hc_1, hc_2, hc_3, 1)
                                    else:
                                        # Not at a hard constraint, hc vals stay the same
                                        new_hc = (hc_1, hc_2, hc_3, hc_4)

                                    destination_state = get_state((dest_x, dest_y, new_cost, new_hc, label_num))

                            transitions[(origin_state, symbol)] = destination_state

    # Add Charge Transitions
    for y in range(len(gridworld)):
        for x in range(len(gridworld[0])):
            for cost in range(max_cost+1):
                for hc_objectives in itertools.product([0,1], repeat=4):
                    for label_num in range(4):
                        origin_state = get_state((x, y, cost, hc_objectives, label_num))

                        if label_num != 0:
                            # Check if already charged
                            destination_state = "Sink"
                        else:
                            # Check if at charging point
                            if (dest_coords) == lc_1_loc:
                                destination_state = get_state((x, y, cost, hc_objectives, 1))
                            elif (dest_coords) == lc_2_loc:
                                destination_state = get_state((x, y, cost, hc_objectives, 2))
                            elif (dest_coords) == lc_3_loc:
                                destination_state = get_state((x, y, cost, hc_objectives, 3))
                            else:
                                # Can't charge if not at a charging point.
                                destination_state = "Sink"

                        transitions[(origin_state, "Charge")] = destination_state

    ## DFA CREATION AND MINIMIZATION ##
    print(list(state_map.values())[0])
    states = states | set(state_map.values())

    print("Total States:", len(states))

    class_keys = [(label_num, cost) for label_num in [1,2,3] for cost in range(max_cost+1)]

    def make_dfa_wrapper(class_key):
        label_num, cost = class_key

        accepting_states = {state_map[(end_loc[0], end_loc[1], cost, (1,1,1,1), label_num)]}

        new_dfa = Dfa(alphabet, states, accepting_states, start_state, transitions).minimize()

        print(("Label" + str(label_num), cost), "States:", len(new_dfa.states))

        return (("Label" + str(label_num), cost), new_dfa)


    with multiprocess.Pool(multiprocess.cpu_count() - 2) as p:
        pool_output = p.map(make_dfa_wrapper, class_keys)

    direct_dfas = {class_key:dfa for class_key, dfa in pool_output}

    return direct_dfas

if __name__ == '__main__':
    direct_dfas = gridworld_to_dfa(SMALL_GRIDWORLD, SMALL_GRIDWORLD_COSTS, SMALL_LENGTH_BOUNDS)

