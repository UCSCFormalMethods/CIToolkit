import math
import itertools
from pyeda.boolalg.expr import expr, exprvar, expr2dimacscnf
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import collections  as mc

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

length_bounds = (1,25)

# Create a mapping from a cell in gridworld to a variable assignment. 
num_cell_vars = math.ceil(math.log(len(GRIDWORLD) * len(GRIDWORLD[0]),2))

cell_id_map = {}
id_cell_map = {}

var_assignments = itertools.product([0,1], repeat=num_cell_vars)

for y in range(len(GRIDWORLD)):
    for x in range(len(GRIDWORLD[0])):
        assignment = next(var_assignments)
        cell_id_map[(x, y)] = assignment
        id_cell_map[assignment] = (x,y)


def run():
    hc = create_hard_constraint().simplify()

    cnf_hc = hc.tseitin()

    print("Starting Sat solver...")

    assignment = {v:val for v, val in cnf_hc.satisfy_one().items() if v.name != 'aux'}

    coords = []

    for var_iter in range(length_bounds[0], length_bounds[1]+1):
        var = "Cell_" + str(var_iter)
        if var not in {name.names[0] for name in assignment.keys()}:
            break
        else:
            cell_id = []

            for pos in range(num_cell_vars):
                cell_id.append(assignment[exprvar(var, pos)])

            print(var + " : " + str(cell_id))

            coords.append(id_cell_map[tuple(cell_id)])

    print(coords)

    #print({v:val for v, val in cnf_hc.satisfy_one().items() if v.name != 'aux'})
    #print([v for v, val in cnf_hc.satisfy_one().items() if v.name != 'aux' and v.indices[0] == 1])

def create_hard_constraint():
    hc = expr(False)

    for length in range(length_bounds[0], length_bounds[1]):
        hc = hc | create_exact_length_hard_constraint(length)

    return hc

def create_exact_length_hard_constraint(length):
    # Make constraint accept any accepting path.
    start_loc = np.where(np.array(GRIDWORLD) == 2)
    end_loc = np.where(np.array(GRIDWORLD) == 3)

    start_loc = (start_loc[1][0], start_loc[0][0])
    end_loc = (end_loc[1][0], end_loc[0][0])

    start_var = "Cell_1"
    end_var = "Cell_" + str(length)

    start_cell_valid = get_var_equal_cell_formula(start_var, cell_id_map[(start_loc)])
    end_cell_valid = get_var_equal_cell_formula(end_var, cell_id_map[(end_loc)])

    hc = start_cell_valid

    last_var = start_var

    for var_iter in range(2, length+1):
        next_var = "Cell_" + str(var_iter)

        hc = hc & get_feasible_transitions_formula(last_var, next_var)

        last_var = next_var

    hc = hc & end_cell_valid

    # Make constraint visit all hard objectives.
    hc_1_loc = np.where(np.array(GRIDWORLD) == 5)
    hc_1_loc = (hc_1_loc[1][0], hc_1_loc[0][0])

    hc_2_loc = np.where(np.array(GRIDWORLD) == 6)
    hc_2_loc = (hc_2_loc[1][0], hc_2_loc[0][0])

    hc_3_loc = np.where(np.array(GRIDWORLD) == 7)
    hc_3_loc = (hc_3_loc[1][0], hc_3_loc[0][0])

    hc_4_loc = np.where(np.array(GRIDWORLD) == 8)
    hc_4_loc = (hc_4_loc[1][0], hc_4_loc[0][0])

    ho_locs = [hc_1_loc, hc_2_loc, hc_3_loc, hc_4_loc]

    for ho_loc in ho_locs:
        ho_cell = cell_id_map[ho_loc]
        ho_formula = expr(False)

        for var_iter in range(1, length+1):
            target_var = "Cell_" + str(var_iter)

            ho_formula = ho_formula | get_var_equal_cell_formula(target_var, ho_cell)

        hc = hc & ho_formula

    return hc

def get_feasible_transitions_formula(var_1, var_2):
    # Create a set of all possible transitions
    # (origin state id, destination state id)
    transitions = set()

    max_y = len(GRIDWORLD) - 1
    max_x = len(GRIDWORLD[0]) - 1

    for y in range(len(GRIDWORLD)):
        for x in range(len(GRIDWORLD[0])):
            for symbol in alphabet:
                origin_cell_id = cell_id_map[(x, y)]

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

                # First check map boundaries and impassable areas.
                if dest_x < 0 or dest_x > max_x or dest_y < 0 or dest_y > max_y:
                    # Off the edge of the map, no valid transition
                    continue
                else:

                    destination_cell_id = cell_id_map[(dest_x, dest_y)]

                transitions.add((origin_cell_id, destination_cell_id))

    # Create a boolean formula that accepts only if var_1 to var_2 is a valid transition.
    transition_formula = expr(False)

    for transition in transitions:
        orig_cell, dest_cell = transition

        orig_cell_valid = get_var_equal_cell_formula(var_1, orig_cell)

        dest_cell_valid = get_var_equal_cell_formula(var_2, dest_cell)

        transition_formula = transition_formula | (orig_cell_valid & dest_cell_valid)

    return transition_formula

def get_var_equal_cell_formula(var, cell):
    # Returns a formula that is true if var is set to cell
    cell_valid = expr(True)

    for pos in range(num_cell_vars):
        if cell[pos] == 1:
            cell_valid = cell_valid & exprvar(var, pos)
        else:
            cell_valid = cell_valid & ~exprvar(var, pos)

    return cell_valid

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
    point_b = (improvisation[0][0] + 0.5, improvisation[0][1] + 0.5)

    lines = []

    for coords in improvisation[1:]:
        point_a = point_b

        point_b = (coords[0] + 0.5, coords[1] + 0.5)

        lines.append([point_a, point_b])

    lc = mc.LineCollection(lines, color="red", linewidths=2)
    ax.add_collection(lc)

    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    run()
