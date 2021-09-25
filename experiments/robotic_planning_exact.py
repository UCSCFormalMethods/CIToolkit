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


from citoolkit.specifications.dfa import Dfa

from citoolkit.improvisers.labelled_quantitative_ci import MaxEntropyLabelledQuantitativeCI
from citoolkit.improvisers.quantitative_ci import QuantitativeCI

from fractions import Fraction

from citoolkit.labellingfunctions.labelling_dfa import LabellingDfa
from citoolkit.costfunctions.accumulated_cost_dfa import AccumulatedCostDfa

def run_exact_experiments(LARGE_MAP):
    BASE_DIRECTORY = "exact_data/"

    if LARGE_MAP:
        BASE_DIRECTORY += "large_map/"
    else:
        BASE_DIRECTORY += "small_map/"

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
    if LARGE_MAP:
        GRIDWORLD =         (
                            (8, 0, 0, 3, 0, 0,  0),
                            (0, 0, 0, 0, 0, 5,  0),
                            (0, 0, 0, 0, 0, 0,  0),
                            (1, 0, 1, 0, 1, 0,  1),
                            (7, 0, 0, 0, 0, 0, 10),
                            (0, 0, 9, 0, 0, 4,  0),
                            (6, 0, 0, 2, 0, 0,  0)
                            )

        GRIDWORLD_COSTS =   (
                            ( 9,  6,  3,  0,  4,  4,  4),
                            ( 6,  3,  2,  1,  4,  2,  4),
                            ( 3,  2,  2,  1,  2,  2,  2),
                            ( 0, 10,  0,  1,  0, 10,  0),
                            ( 2,  2,  2,  1,  4,  4,  2),
                            ( 2,  4,  1,  1,  2,  4,  4),
                            ( 2,  2,  2,  0,  2,  2,  2)
                            )

        length_bounds = (1,30)
        COST_BOUND = 50
        ALPHA_LIST = [0,0,0]
        BETA_LIST = [1e-6,1e-6,1e-6]

    else:
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

        length_bounds = (1,25)
        COST_BOUND = 30
        ALPHA_LIST = [0,0,0]
        BETA_LIST = [1e-5,1e-5,1e-5]

    alphabet = {"North", "East", "South", "West"}

    length_bounds = (1,25)

    NUM_SAMPLES = 1000000

    word_prob_bounds = (0, 1/3e5)
    cost_bound = 30
    label_prob_bounds = (Fraction(1,6), Fraction(1,2))

    print("\n")
    print("Starting Exact LQCI Robotic Planning Experiment...")
    if LARGE_MAP:
        print("Using Large Map...")
    else:
        print("Using Small Map...")
    print()

    if not os.path.exists(BASE_DIRECTORY):
        os.makedirs(BASE_DIRECTORY)

    print("Trace Length Bounds:", length_bounds)
    print("Word Prob Bounds:", word_prob_bounds)
    print("Label Prob Bounds:", label_prob_bounds)
    print("Cost Bound", cost_bound)

    # start = time.time()

    # if os.path.isfile(BASE_DIRECTORY + "hard_constraint.pickle"):
    #     print("Loading hard constraint from pickle...\n")
    #     hard_constraint = pickle.load(open(BASE_DIRECTORY + "hard_constraint.pickle", 'rb'))
    # else:
    #     print("Creating hard constraint...\n")
    #     hard_constraint = create_hard_constraint()
    #     pickle.dump(hard_constraint, open(BASE_DIRECTORY + "hard_constraint.pickle", "wb"))
    #     print("Done creating Hard Constraint. Total time taken: " + str(time.time() - start))

    # print("Hard Constraint States:", len(hard_constraint.states))

    hard_constraint = Dfa.min_length_dfa(alphabet, 0)

    start = time.time()

    if os.path.isfile(BASE_DIRECTORY + "label_function.pickle"):
        print("Loading label function from pickle...\n")
        label_function = pickle.load(open(BASE_DIRECTORY + "label_function.pickle", 'rb'))
    else:
        print("Creating label function...\n")
        label_function = create_label_function(GRIDWORLD, alphabet)
        label_function.decompose()
        pickle.dump(label_function, open(BASE_DIRECTORY + "label_function.pickle", "wb"))
        print("Done creating Label Function. Total time taken: " + str(time.time() - start))

    print("Label Function States:", len(label_function.dfa.states))


    start = time.time()

    if os.path.isfile(BASE_DIRECTORY + "cost_function.pickle"):
        print("Loading cost function from pickle...\n")
        cost_function = pickle.load(open(BASE_DIRECTORY + "cost_function.pickle", 'rb'))
    else:
        print("Creating cost function...\n")
        cost_function = create_combo_hard_cost_constraint(GRIDWORLD, GRIDWORLD_COSTS, alphabet, length_bounds)
        cost_function.decompose()
        pickle.dump(cost_function, open(BASE_DIRECTORY + "cost_function.pickle", "wb"))
        print("Done creating Cost Function. Total time taken: " + str(time.time() - start))

    print("Cost Function States:", len(cost_function.dfa.states))

    start = time.time()

    if os.path.isfile(BASE_DIRECTORY + "qci_improviser.pickle"):
        print("Loading Quantitative CI improviser from pickle...\n")
        qci_improviser = pickle.load(open(BASE_DIRECTORY + "qci_improviser.pickle", 'rb'))
    else:
        print("Creating Quantitative CI improviser...\n")
        qci_improviser = QuantitativeCI((hard_constraint & label_function.dfa).explicit(), cost_function, length_bounds, cost_bound, word_prob_bounds)
        pickle.dump(qci_improviser, open(BASE_DIRECTORY + "qci_improviser.pickle", "wb"))
        print("Done creating Quantitative CI Improviser. Total time taken: " + str(time.time() - start))


    start = time.time()

    if os.path.isfile(BASE_DIRECTORY + "melqci_improviser.pickle"):
        print("Loading Max Entropy LQCI improviser from pickle...\n")
        melqci_improviser = pickle.load(open(BASE_DIRECTORY + "melqci_improviser.pickle", 'rb'))
    else:
        print("Creating Max Entropy LQCI improviser...\n")
        melqci_improviser = MaxEntropyLabelledQuantitativeCI(hard_constraint, cost_function, label_function, length_bounds, cost_bound, label_prob_bounds)
        pickle.dump(melqci_improviser, open(BASE_DIRECTORY + "melqci_improviser.pickle", "wb"))
        print("Done creating ME LQCI Improviser. Total time taken: " + str(time.time() - start))

    print("Num Samples:", NUM_SAMPLES)

    start = time.time()

    if os.path.isfile(BASE_DIRECTORY + "qci_samples.pickle"):
        print("Loading " + str(NUM_SAMPLES) + " samples for Quantitative CI Improviser for pickle...\n")
        qci_samples = pickle.load(open(BASE_DIRECTORY + "qci_samples.pickle", 'rb'))
    else:
        print("Sampling " + str(NUM_SAMPLES) + " from Quantitative CI Improviser...\n")
        qci_samples = [qci_improviser.improvise() for _ in range(NUM_SAMPLES)]
        pickle.dump(qci_samples, open(BASE_DIRECTORY + "qci_samples.pickle", "wb"))
        print("Done sampling Quantitative CI Improviser. Total time taken: " + str(time.time() - start))

    start = time.time()

    if os.path.isfile(BASE_DIRECTORY + "melqci_samples.pickle"):
        print("Loading " + str(NUM_SAMPLES) + " samples for Max Entropy LQCI Improviser for pickle...\n")
        melqci_samples = pickle.load(open(BASE_DIRECTORY + "melqci_samples.pickle", 'rb'))
    else:
        print("Sampling " + str(NUM_SAMPLES) + " from Max Entropy LQCI Improviser...\n")
        melqci_samples = [melqci_improviser.improvise() for _ in range(NUM_SAMPLES)]
        pickle.dump(melqci_samples, open(BASE_DIRECTORY + "melqci_samples.pickle", "wb"))
        print("Done sampling Max Entropy LQCI Improviser. Total time taken: " + str(time.time() - start))

    qci_label_counts = [0,0,0]
    qci_sum_cost = 0

    for sample in qci_samples:
        qci_sum_cost += cost_function.cost(sample)

        sample_label = label_function.label(sample)

        if sample_label == "Label0":
            qci_label_counts[0] += 1
        elif sample_label == "Label1":
            qci_label_counts[1] += 1
        elif sample_label == "Label2":
            qci_label_counts[2] += 1
        else:
            assert False

    print("Quantitative CI Samples Average Cost:", qci_sum_cost/NUM_SAMPLES)
    print("Quantitative CI Label Probabilities:", [count/NUM_SAMPLES for count in qci_label_counts])

    melqci_label_counts = [0,0,0]
    melqci_sum_cost = 0

    for sample in melqci_samples:
        melqci_sum_cost += cost_function.cost(sample)

        sample_label = label_function.label(sample)

        if sample_label == "Label0":
            melqci_label_counts[0] += 1
        elif sample_label == "Label1":
            melqci_label_counts[1] += 1
        elif sample_label == "Label2":
            melqci_label_counts[2] += 1
        else:
            assert False

    print("Max Entropy LQCI Samples Average Cost:", melqci_sum_cost/NUM_SAMPLES)
    print("Max Entropy LQCI Label Probabilities:", [count/NUM_SAMPLES for count in melqci_label_counts])

def create_hard_constraint(GRIDWORLD, alphabet):
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

    hc_1_loc = np.where(np.array(GRIDWORLD) == 4)
    hc_1_loc = (hc_1_loc[1][0], hc_1_loc[0][0])

    hc_2_loc = np.where(np.array(GRIDWORLD) == 5)
    hc_2_loc = (hc_2_loc[1][0], hc_2_loc[0][0])

    hc_3_loc = np.where(np.array(GRIDWORLD) == 6)
    hc_3_loc = (hc_3_loc[1][0], hc_3_loc[0][0])

    hc_4_loc = np.where(np.array(GRIDWORLD) == 7)
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

def create_label_function(GRIDWORLD, alphabet):
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

    lc_1_loc = np.where(np.array(GRIDWORLD) == 8)
    lc_1_loc = (lc_1_loc[1][0], lc_1_loc[0][0])

    lc_2_loc = np.where(np.array(GRIDWORLD) == 9)
    lc_2_loc = (lc_2_loc[1][0], lc_2_loc[0][0])

    lc_3_loc = np.where(np.array(GRIDWORLD) == 10)
    lc_3_loc = (lc_3_loc[1][0], lc_3_loc[0][0])

    lo_locs = [lc_1_loc, lc_2_loc, lc_3_loc]

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

def create_cost_function(GRIDWORLD, GRIDWORLD_COSTS, alphabet, length_bounds):
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
    end_loc = np.where(np.array(GRIDWORLD) == 3)

    start_loc = (start_loc[1][0], start_loc[0][0])
    end_loc = (end_loc[1][0], end_loc[0][0])

    states.add("Sink")
    accepting_states = {state_map[(end_loc[0], end_loc[1])]}
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

def create_combo_hard_cost_constraint(GRIDWORLD, GRIDWORLD_COSTS, alphabet, length_bounds):
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

    hc_1_loc = np.where(np.array(GRIDWORLD) == 4)
    hc_1_loc = (hc_1_loc[1][0], hc_1_loc[0][0])

    hc_2_loc = np.where(np.array(GRIDWORLD) == 5)
    hc_2_loc = (hc_2_loc[1][0], hc_2_loc[0][0])

    hc_3_loc = np.where(np.array(GRIDWORLD) == 6)
    hc_3_loc = (hc_3_loc[1][0], hc_3_loc[0][0])

    hc_4_loc = np.where(np.array(GRIDWORLD) == 7)
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

    cost_dfa = Dfa(alphabet, states, accepting_states, start_state, transitions)

    cost_map = dict()

    for y in range(len(GRIDWORLD)):
        for x in range(len(GRIDWORLD[0])):
            for hc_objectives in itertools.product([0,1], repeat=4):
                cost_map[state_map[(x, y, hc_objectives)]] = GRIDWORLD_COSTS[y][x]

    cost_map["Sink"] = 9

    cost_func = AccumulatedCostDfa(cost_dfa, cost_map, max_word_length=length_bounds[1])

    return cost_func

def draw_improvisation(improvisation, GRIDWORLD, GRIDWORLD_COSTS):
    fig, ax = plt.subplots(1, 1, tight_layout=True)

    for x in range(len(GRIDWORLD) + 1):
        ax.axhline(x, lw=2, color='k', zorder=5)
        ax.axvline(x, lw=2, color='k', zorder=5)

    ax.imshow(GRIDWORLD_COSTS, cmap="binary", interpolation='none', extent=[0, len(GRIDWORLD), 0, len(GRIDWORLD)], zorder=0)

    impassable = 1
    start = 2
    end = 3
    hard_objectives = [4,5,6,7]
    label_objectives = [8,9,10]

    for x in range(len(GRIDWORLD)):
        for y in range(len(GRIDWORLD)):
            if GRIDWORLD[y][x] == impassable:
                ax.add_patch(PathPatch(TextPath((x + 0.13, len(GRIDWORLD) - 1 - y + 0.11), "X", size=1), color="red", ec="white"))
            elif GRIDWORLD[y][x] == start:
                ax.add_patch(PathPatch(TextPath((x + 0.2, len(GRIDWORLD) - 1 - y + 0.11), "S", size=1), color="green", ec="white"))
            elif GRIDWORLD[y][x] == end:
                ax.add_patch(PathPatch(TextPath((x + 0.17, len(GRIDWORLD) - 1 - y + 0.11), "E", size=1), color="green", ec="white"))
            elif GRIDWORLD[y][x] in hard_objectives:
                ax.add_patch(PathPatch(TextPath((x + 0.11, len(GRIDWORLD) - 1 - y + 0.11), "O", size=1), color="blue", ec="white"))
            elif GRIDWORLD[y][x] in label_objectives:
                ax.add_patch(PathPatch(TextPath((x + 0.13, len(GRIDWORLD) - 1 - y + 0.11), "C", size=1), color="orange", ec="white"))

    point_a = None
    point_b = (improvisation[0][0] + 0.5,  len(GRIDWORLD) - 1 - improvisation[0][1] + 0.5)

    lines = []

    print(improvisation)

    for coords in improvisation[1:]:
        point_a = point_b

        if coords is None:
            break

        point_b = (coords[0] + 0.5,  len(GRIDWORLD) - 1 - coords[1] + 0.5)

        lines.append([point_a, point_b])

    lc = mc.LineCollection(lines, color="red", linewidths=2)
    ax.add_collection(lc)

    plt.axis('off')

    plt.show()
if __name__ == '__main__':
    run_exact_experiments(LARGE_MAP = True)
