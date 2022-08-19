import math
import itertools
import time
import pickle
import numpy as np
from z3 import *
import subprocess
import multiprocessing
import glob
import random
from functools import lru_cache

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import collections  as mc
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch


###################################################################################################
# Main Experiment
###################################################################################################

def run_approximate_experiments(LARGE_MAP, GAMMA,  SMALL_MAP=False):
    global APPROX_BASE_DIRECTORY
    APPROX_BASE_DIRECTORY = "approx_data/"

    if LARGE_MAP:
        APPROX_BASE_DIRECTORY += "large_map_"
    else:
        APPROX_BASE_DIRECTORY += "small_map_"

    APPROX_BASE_DIRECTORY += str(int(GAMMA)) + "/"

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
    if SMALL_MAP:
        GRIDWORLD =         (
                            (8, 0, 3, 0, 5),
                            (0, 1, 0, 1, 0),
                            (7, 0, 0, 0, 10),
                            (0, 9, 0, 4,  0),
                            (6, 0, 2, 0,  0)
                            )

        GRIDWORLD_COSTS =   (
                            (3, 2, 0, 2, 0),
                            (3, 0, 0, 0, 3),
                            (0, 1, 0, 2, 1),
                            (2, 0, 0, 0, 2),
                            (0, 1, 0, 1, 1)
                            )

        length_bounds = (1,25)
        COST_BOUND = 50
        ALPHA_LIST = [0,0,0]
        BETA_LIST = [1e-4,1e-4,1e-4]

    elif LARGE_MAP:
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
        BETA_LIST = [1e-5,1e-5,1e-5]

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

    # Create a mapping from a cell in gridworld to a variable assignment.
    num_cell_vars = math.ceil(math.log(len(GRIDWORLD) * len(GRIDWORLD[0]) + 1, 2))

    cell_id_map = {}
    id_cell_map = {}

    var_assignments = itertools.product([0,1], repeat=num_cell_vars)

    null_cell_id = next(var_assignments)

    assert list(null_cell_id) == [0]*num_cell_vars

    cell_id_map[None] = null_cell_id
    id_cell_map[null_cell_id] = None

    for y in range(len(GRIDWORLD)):
        for x in range(len(GRIDWORLD[0])):
            assignment = next(var_assignments)
            cell_id_map[(x, y)] = assignment
            id_cell_map[assignment] = (x,y)


    num_cost_vars = 10

    lc_1_loc = np.where(np.array(GRIDWORLD) == 8)
    lc_1_loc = (lc_1_loc[1][0], lc_1_loc[0][0])

    lc_2_loc = np.where(np.array(GRIDWORLD) == 9)
    lc_2_loc = (lc_2_loc[1][0], lc_2_loc[0][0])

    lc_3_loc = np.where(np.array(GRIDWORLD) == 10)
    lc_3_loc = (lc_3_loc[1][0], lc_3_loc[0][0])

    lo_locs = [lc_1_loc, lc_2_loc, lc_3_loc]

    num_label_vars = 2

    R = 1.2

    max_cost = 2**num_cost_vars
    max_r = math.ceil(math.log(max_cost, R))-1

    num_buckets = len(lo_locs) * max_r

    OUT_DELTA = 0.2
    OUT_GAMMA = GAMMA
    DELTA = None # 1 - math.pow((1- OUT_DELTA), 1/num_buckets)
    EPSILON = math.pow((1 + OUT_GAMMA)/17, 1/2) - 1
    LAMBDA = .3
    RHO = .4
    NUM_SAMPLES = 500

    print()
    print("------------------------------------------------------------------------------------------")
    print("Starting Approximate LQCI Robotic Planning Experiment...")
    if SMALL_MAP:
        print("Using Extra Large Map...")
    elif LARGE_MAP:
        print("Using Large Map...")
    else:
        print("Using Small Map...")
    print("GAMMA: " + str(GAMMA))
    print()


    start = time.time()
    print("Assembling DIMACS Fomulas...")

    p1 = (num_cost_vars, num_label_vars, lo_locs)
    p2 = (GRIDWORLD, GRIDWORLD_COSTS, length_bounds, tuple(cell_id_map.items()), frozenset(alphabet), num_cell_vars, num_cost_vars, num_label_vars, tuple(lo_locs))

    formula_var_map, feasibility_map = compute_dimacs_formulas(lo_locs, max_r, R, length_bounds, num_cell_vars, p1, p2)

    print("Assembled DIMACS Formulas in " + str(time.time() - start))

    num_nontriv_buckets = sum(map(int, feasibility_map.values()))

    #print(feasibility_map)

    DELTA = 1 - math.pow((1- OUT_DELTA), 1/num_nontriv_buckets)

    # print("R Value:", R)
    # print("Delta:", OUT_DELTA)
    # print("Gamma:", OUT_GAMMA)
    # print("Counting/Sampling Delta:", DELTA)
    # print("Counting/Sampling Epsilon:", EPSILON)

    start = time.time()
    print("Counting solutions for DIMACS Fomulas...")

    class_count_map = compute_class_counts(lo_locs, max_r, R, EPSILON, DELTA, feasibility_map)

    print("Counting solutions for DIMACS Formulas in " + str(time.time() - start))
    print()

    class_size_map = {key:val[0] for key,val in class_count_map.items()}

    # print("Counts:", class_count_map)
    # print("Sizes:", class_size_map)
    # print()

    for label_iter in range(len(lo_locs)):
        label_count = 0
        label_total_lo_cost = 0

        curr_lo_cost = 1

        for cost_iter in range(max_r):
            label_count += class_size_map[(label_iter, cost_iter)]
            label_total_lo_cost += class_size_map[(label_iter, cost_iter)]*curr_lo_cost

            curr_lo_cost = math.ceil(curr_lo_cost * R)

        # print("Label#" + str(label_iter) + "  Size: " + str(label_count))
        # print("Label#" + str(label_iter) + "  Mean Lo Cost: " + str(label_total_lo_cost/label_count))

    # Initialize alpha probabilities
    conditional_weights = {(label_iter, cost_iter):class_size_map[(label_iter, cost_iter)]*(ALPHA_LIST[label_iter]/(1+EPSILON)) for label_iter, cost_iter, _, _ in get_formula_data_list(lo_locs, max_r, R)}

    label_sum_prob = {label_iter:sum([prob for ((l, c), prob) in conditional_weights.items() if l == label_iter]) for label_iter in range(len(lo_locs))}

    # Check if we've already broken probability bounds.
    for label_iter in range(len(lo_locs)):
        assert label_sum_prob[label_iter] <= 1

    # Add beta probabilities
    for label_iter in range(len(lo_locs)):
        for cost_iter in range(max_r):
            new_cost = min((1 + EPSILON) * BETA_LIST[label_iter] * class_size_map[(label_iter, cost_iter)], 1 - label_sum_prob[label_iter])

            conditional_weights[(label_iter, cost_iter)] = new_cost

            label_sum_prob[label_iter] = sum([prob for ((l, c), prob) in conditional_weights.items() if l == label_iter])

    # Check if we've now broken probability bounds
    for label_iter in range(len(lo_locs)):
        assert label_sum_prob[label_iter] == 1

    # Calculate conditional exptected costs
    conditional_costs = {label_iter:sum([conditional_weights[(l,c)]*Lo for l, c, Lo, _ in get_formula_data_list(lo_locs, max_r, R) if l == label_iter]) for label_iter in range(len(lo_locs))}

    # print("Conditional Weights:", conditional_weights)
    # print("Conditional Costs", conditional_costs)

    # Now calculate marginal costs.
    marginal_weights = []

    u = math.floor((1 - len(lo_locs)*LAMBDA)/(RHO - LAMBDA))

    for label_iter in sorted(range(len(lo_locs)), key=lambda x: conditional_costs[label_iter]):
        if label_iter < u:
            marginal_weights.append(RHO)
        elif label_iter == u:
            marginal_weights.append(1 - RHO*u - LAMBDA*(len(lo_locs) - u - 1))
        else:
            marginal_weights.append(LAMBDA)

    assert sum(marginal_weights) == 1

    expected_cost = sum([marginal_weights[label_iter] * conditional_costs[label_iter] for label_iter in range(len(lo_locs))])

    # print("Marginal Weights:", marginal_weights)
    # print("Expected Cost:", expected_cost)
    # print()

    sorted_label_weights = marginal_weights
    sorted_labels = range(len(lo_locs))

    sorted_cost_weights = {label_iter:[conditional_weights[(label_iter, cost_iter)] for cost_iter in range(max_r)] for label_iter in range(len(lo_locs))}
    sorted_costs = range(max_r)

    # print("Sorted Label Weights:", sorted_label_weights)
    # print("Sorted Cost Weights:", sorted_cost_weights)

    start = time.time()
    if os.path.isfile(APPROX_BASE_DIRECTORY + "samples.pickle"):
        print("Loading samples from pickle...\n")
        samples = pickle.load(open(APPROX_BASE_DIRECTORY + "samples.pickle", 'rb'))
    else:
        print("Sampling improviser...\n")
        samples = []

        for sample_iter in range(NUM_SAMPLES):
            #print("Sample#", sample_iter)
            label_choice = random.choices(population=sorted_labels, weights=sorted_label_weights, k=1)[0]
            cost_choice = random.choices(population=sorted_costs, weights=sorted_cost_weights[label_choice], k=1)[0]

            coords = sample_improviser(label_choice, cost_choice, formula_var_map, class_count_map, length_bounds, num_cell_vars, id_cell_map)

            #draw_improvisation(coords, GRIDWORLD, GRIDWORLD_COSTS)

            samples.append(coords)

        pickle.dump(samples, open(APPROX_BASE_DIRECTORY + "samples.pickle", "wb"))
        print("Done sampling. Total time taken: " + str(time.time() - start))

def sample_improviser(label_choice, cost_choice, formula_var_map, class_count_map, length_bounds, num_cell_vars, id_cell_map):
    target_formula = APPROX_BASE_DIRECTORY + "formulas/RP_Label_" + str(label_choice) + "_Cost_" + str(cost_choice) + ".cnf"
    target_cell_count = class_count_map[(label_choice, cost_choice)][1]
    target_hash_count = class_count_map[(label_choice, cost_choice)][2]

    #print(target_formula)
    #print("Cell Count:", target_cell_count, "    Hash Count:", target_hash_count)

    start = time.time()

    var_nums = sample_dimacs_formula(target_formula, target_cell_count, target_hash_count)

    #print("Sampled in " + str(time.time() - start))

    solution = {}
    var_num_map = formula_var_map[(label_choice, cost_choice)]
    num_var_map = {num:var for (var, num) in var_num_map.items()}

    for num in var_nums:

        var = abs(num)

        if num == 0:
            break
        elif num > 0:
            solution[num_var_map[abs(num)]] = True
        else:
            solution[num_var_map[abs(num)]] = False

    coords = []

    for var_iter in range(length_bounds[0], length_bounds[1]+1):
        var = "Cell_" + str(var_iter)
        cell_id = []

        for pos in range(num_cell_vars):
            var_pos = var + "[" + str(pos) + "]"

            if solution[var_pos]:
                cell_id.append(1)
            else:
                cell_id.append(0)

        coords.append(id_cell_map[tuple(cell_id)])

    return coords

###################################################################################################
# Experiment Funcs
###################################################################################################
def compute_class_counts(lo_locs, max_r, R, EPSILON, DELTA, feasibility_map):
    if os.path.exists(APPROX_BASE_DIRECTORY + "/class_counts.pickle"):
        print("Loading class sizes dictionary from pickle...\n")
        class_counts = pickle.load(open(APPROX_BASE_DIRECTORY + "/class_counts.pickle", 'rb'))
        return class_counts

    formula_files = glob.glob(APPROX_BASE_DIRECTORY + "formulas/*.cnf")

    formula_data = get_formula_data_list(lo_locs, max_r, R)

    formula_data = [(x, EPSILON, DELTA, feasibility_map) for x in formula_data]

    with multiprocessing.Pool(multiprocessing.cpu_count() - 5) as p:
        formula_count_data = p.map(count_dimacs_wrapper, formula_data, chunksize=1)

        p.close()
        p.join()

    sum_time = sum([time for _,_, time in formula_count_data])
    print("Total CPU Time:", sum_time)

    formula_count_map = {formula_name:count for formula_name, count, _ in formula_count_data}

    pickle.dump(formula_count_map, open(APPROX_BASE_DIRECTORY + "/class_counts.pickle", "wb"))

    return formula_count_map

def count_dimacs_wrapper(x):
    start_time = time.time()
    x, EPSILON, DELTA, feasibility_map = x
    label_num, curr_r, left_cost, right_cost = x

    if not feasibility_map[(label_num, curr_r)]:
        return ((label_num, curr_r), (0, 0, 0), 0)

    formula_name = "RP_Label_" + str(label_num) + "_Cost_" + str(curr_r) + ".cnf"
    count = count_dimacs_formula(APPROX_BASE_DIRECTORY + "formulas/" + formula_name, EPSILON, DELTA)
    return ((label_num, curr_r), count, time.time() - start_time)

def compute_dimacs_formulas(lo_locs, max_r, R, length_bounds, num_cell_vars, p1, p2):
    if not os.path.exists(APPROX_BASE_DIRECTORY + "formulas"):
        os.makedirs(APPROX_BASE_DIRECTORY + "formulas")
    
    if os.path.exists(APPROX_BASE_DIRECTORY + "/formula_metadata.pickle"):
        print("Loading variable mappings from pickle...\n")
        formula_metadata = pickle.load(open(APPROX_BASE_DIRECTORY + "/formula_metadata.pickle", 'rb'))
        return formula_metadata

    formula_data = get_formula_data_list(lo_locs, max_r, R)
    formula_data = [(x, length_bounds, num_cell_vars) for x in formula_data]
    formula_data = [(x, p1, p2) for x in formula_data]

    with multiprocessing.Pool(multiprocessing.cpu_count() - 5) as p:
        var_map_list = p.map(convert_dimacs_wrapper, formula_data, chunksize=1)

        p.close()
        p.join()

    sum_time = sum([time for _,_,_, time in var_map_list])
    print("Total CPU Time:", sum_time)

    var_maps = {key: val for key, val, _, _ in var_map_list}
    feasibility_map = {key: feasible for key, _, feasible,_ in var_map_list}

    formula_metadata = (var_maps, feasibility_map)

    pickle.dump(formula_metadata, open(APPROX_BASE_DIRECTORY + "/formula_metadata.pickle", "wb"))

    return formula_metadata

def convert_dimacs_wrapper(x):
    start_time = time.process_time()
    x_prime, p1, p2 = x
    formula_data, length_bounds, num_cell_vars = x_prime
    label_num, curr_r, left_cost, right_cost = formula_data

    formula_problem =  get_symbolic_problem_instance(p1, p2, min_cost=left_cost, max_cost=right_cost, target_label=label_num)

    solver = Solver()
    solver.add(formula_problem)
    if str(solver.check()) == "sat":
        formula_feasible = True
    else:
        formula_feasible = False

    formula_name = "RP_Label_" + str(label_num) + "_Cost_" + str(curr_r) + ".cnf"
    var_map = convert_dimacs_problem_instance(formula_problem, APPROX_BASE_DIRECTORY + "formulas/" + formula_name, length_bounds, num_cell_vars)
    return ((label_num, curr_r), var_map, formula_feasible, time.process_time() - start_time)

def get_formula_data_list(lo_locs, max_r, R):
    # Create name and parameter tuples.
    formula_data = []

    label_nums = range(len(lo_locs))

    for label_num in label_nums:
        left_cost = 1

        for curr_r in range(max_r):
            right_cost = math.ceil(left_cost * R)

            formula_data.append((label_num, curr_r, left_cost, right_cost))

            left_cost = right_cost

    return formula_data

def convert_dimacs_problem_instance(problem, out_file_path, length_bounds, num_cell_vars):
    # Convert z3 expression to CNF form.
    tactic = Then('simplify', 'bit-blast', 'tseitin-cnf')
    goal = tactic(problem)

    assert len(goal) == 1

    clauses = goal[0]

    s = Solver()
    s.add(clauses)
    if s.check() != sat:
        clauses = [Bool("Infeasible"), Not(Bool("Infeasible"))]

    context = (1, {})

    f = open(out_file_path, "w")

    # Count and construct map for variables
    path_variables = []
    for cell_iter in range(1, length_bounds[1]+1):
        for pos in range(num_cell_vars):
            context = add_dimacs_variable("Cell_" + str(cell_iter) + "[" + str(pos) + "]", context)

    for clause in clauses:
        if is_or(clause):   # not a unit clause
            for literal in clause.children():
                if is_not(literal):   # literal is negated
                    context = add_dimacs_variable(str(literal.children()[0]), context)
                else:
                    context = add_dimacs_variable(str(literal), context)
        elif is_not(clause):   # negative unit clause
            context = add_dimacs_variable(str(clause.children()[0]), context)
        else:   # positive unit clause
            context = add_dimacs_variable(str(clause), context)

    dimac_var_map = context[1]

    f.write('p cnf ' + str(len(dimac_var_map)) + ' ' + str(len(clauses)) + '\n')

    # print(list(filter(lambda x: "Cell_" in str(x[0]), dimac_var_map.items())))

    # Add ind annotations
    path_variables = []
    for cell_iter in range(1, length_bounds[1]+1):
        for pos in range(num_cell_vars):
            path_variables.append(dimac_var_map["Cell_" + str(cell_iter) + "[" + str(pos) + "]"])

    it = iter(path_variables)
    while True:
        nextBatch = itertools.islice(it, 10)
        s = ''
        for var in nextBatch:
            s += str(var)+' '
        if s == '':
            break
        else:
            f.write('c ind '+s+'0\n')

    # Add clauses
    for clause in clauses:
        clause_encoding = ""
        if is_or(clause):   # not a unit clause
            for literal in clause.children():
                if is_not(literal):   # literal is negated
                    varNumber = dimac_var_map[str(literal.children()[0])]
                    clause_encoding += ('-' + str(varNumber) + ' ')
                else:
                    varNumber = dimac_var_map[str(literal)]
                    clause_encoding += str(varNumber) + ' '
        elif is_not(clause):   # negative unit clause
            varNumber = dimac_var_map[str(clause.children()[0])]
            clause_encoding += ('-' + str(varNumber) + ' ')
        else:   # positive unit clause
            varNumber = dimac_var_map[str(clause)]
            clause_encoding += (str(varNumber) + ' ')
        clause_encoding += ('0\n')
        f.write(clause_encoding)

    return dimac_var_map

def add_dimacs_variable(var, context):
    next_var, mapping = context

    if var in mapping:
        return (next_var, mapping)
    else:
        mapping[var] = next_var
        next_var += 1
        return (next_var, mapping)

def get_symbolic_problem_instance(p1, p2, min_cost = None, max_cost = None, target_label = None):
    # NOTE: MAX COST IS NOT INCLUSIVE
    BASE_HARD_CONSTRAINT = create_hard_constraint(p2)
    BASE_COST_FUNCTION = create_cost_function(p2)
    BASE_LABEL_FUNCTION = create_label_function(p2)

    num_cost_vars, num_label_vars, lo_locs = p1

    problem = And(BASE_HARD_CONSTRAINT, BASE_COST_FUNCTION, BASE_LABEL_FUNCTION)

    max_cost = min(max_cost, 2**num_cost_vars - 1)
    min_cost = max(min_cost, 0)

    if min_cost is not None:
        problem = And(problem,  BitVec("CostSum", num_cost_vars) >= min_cost)

    if max_cost is not None:
        problem = And(problem,  BitVec("CostSum", num_cost_vars) < max_cost)

    if target_label is not None:
        assert target_label <= len(lo_locs) - 1
        problem = And(problem, BitVec("LabelChoice", num_label_vars) == BitVecVal(target_label, num_label_vars))

    return problem

def count_dimacs_formula(file_path, EPSILON, DELTA):
    # Returns (total_count, cell_count, hash_count)
    arguments = ["./approxmc", "--verb", "0", "--epsilon", str(EPSILON), "--delta", str(DELTA), "--seed", str(random.randint(1,1000000000)), file_path]

    process = subprocess.run(args=arguments, capture_output=True)

    output = process.stdout.decode("utf-8")

    for line in output.split("\n"):
        if line[:34] == 'c [appmc] Number of solutions is: ':
            stripped_line = line[34:]
            split_line = stripped_line.split("*")
            cell_count = int(split_line[0])
            hash_count = int(split_line[3])
            count = cell_count * (2**hash_count)
            break

    return (count, cell_count, hash_count)

def sample_dimacs_formula(file_path, cell_count, hash_count):
    arguments = ["./unigen", "--verb", "0", "--multisample", "0", "--seed", str(random.randint(1,1000000000)) , \
    "--samples", "1", "--param_hash_count", str(hash_count), "--param_cell_count", str(cell_count), file_path]

    process = subprocess.run(args=arguments, capture_output=True)

    output = process.stdout.decode("utf-8")
    #print(output)

    line = output.split("\n")[0]

    var_nums = map(int, [item.strip() for item in line.split()])

    return var_nums

###################################################################################################
# Hard Constraint Funcs
###################################################################################################
def create_hard_constraint(p):
    GRIDWORLD, GRIDWORLD_COSTS, length_bounds, _, alphabet, num_cell_vars, num_cost_vars, num_label_vars, lo_locs = p

    hc = False

    for length in range(length_bounds[0], length_bounds[1]+1):
        hc = Or(hc, create_exact_length_hard_constraint(length, length_bounds[1], p))

    return hc

def create_exact_length_hard_constraint(length, max_length, p):
    GRIDWORLD, GRIDWORLD_COSTS, length_bounds, cell_id_map_items, alphabet, num_cell_vars, num_cost_vars, num_label_vars, lo_locs = p

    cell_id_map = {key:val for (key,val) in cell_id_map_items}

    # Make constraint accept any accepting path.
    start_loc = np.where(np.array(GRIDWORLD) == 2)
    end_loc = np.where(np.array(GRIDWORLD) == 3)

    start_loc = (start_loc[1][0], start_loc[0][0])
    end_loc = (end_loc[1][0], end_loc[0][0])

    start_var = "Cell_1"
    end_var = "Cell_" + str(length)

    start_cell_valid = get_var_equal_cell_formula(start_var, cell_id_map[(start_loc)], p)
    end_cell_valid = get_var_equal_cell_formula(end_var, cell_id_map[(end_loc)], p)

    # Start cell is valid
    hc = start_cell_valid

    last_var = start_var

    # Path is valid
    for var_iter in range(2, length+1):
        next_var = "Cell_" + str(var_iter)

        hc = And(hc, get_feasible_transitions_formula(last_var, next_var, p))

        last_var = next_var

    # End cell is valid
    hc =And(hc, end_cell_valid)

    # All other cells are set to null cell id
    for var_iter in range(length+1, max_length+1):
        ignored_var = "Cell_" + str(var_iter)

        hc = And(hc, get_var_equal_cell_formula(ignored_var, cell_id_map[None], p))

    # Make constraint visit all hard objectives.
    hc_1_loc = np.where(np.array(GRIDWORLD) == 4)
    hc_1_loc = (hc_1_loc[1][0], hc_1_loc[0][0])

    hc_2_loc = np.where(np.array(GRIDWORLD) == 5)
    hc_2_loc = (hc_2_loc[1][0], hc_2_loc[0][0])

    hc_3_loc = np.where(np.array(GRIDWORLD) == 6)
    hc_3_loc = (hc_3_loc[1][0], hc_3_loc[0][0])

    hc_4_loc = np.where(np.array(GRIDWORLD) == 7)
    hc_4_loc = (hc_4_loc[1][0], hc_4_loc[0][0])

    ho_locs = [hc_1_loc, hc_2_loc, hc_3_loc, hc_4_loc]

    for ho_loc in ho_locs:
        ho_cell = cell_id_map[ho_loc]
        ho_formula = False

        for var_iter in range(1, length+1):
            target_var = "Cell_" + str(var_iter)

            ho_formula = Or(ho_formula, get_var_equal_cell_formula(target_var, ho_cell, p))

        hc = And(hc, ho_formula)

    return hc

@lru_cache(maxsize=None)
def get_feasible_transitions_formula(var_1, var_2, p):
    GRIDWORLD, GRIDWORLD_COSTS, length_bounds, cell_id_map_items, alphabet, num_cell_vars, num_cost_vars, num_label_vars, lo_locs = p

    cell_id_map = {key:val for (key,val) in cell_id_map_items}

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
                elif GRIDWORLD[y][x] == 1:
                    # In impassable area
                    continue
                elif GRIDWORLD[dest_y][dest_x] == 1:
                    # Into impassable area
                    continue
                else:
                    destination_cell_id = cell_id_map[(dest_x, dest_y)]

                transitions.add((origin_cell_id, destination_cell_id))

    # Create a boolean formula that accepts only if var_1 to var_2 is a valid transition.
    transition_formula = False

    for transition in transitions:
        orig_cell, dest_cell = transition

        orig_cell_valid = get_var_equal_cell_formula(var_1, orig_cell, p)

        dest_cell_valid = get_var_equal_cell_formula(var_2, dest_cell, p)

        transition_formula = Or(transition_formula, And(orig_cell_valid, dest_cell_valid))

    return transition_formula

###################################################################################################
# CostFunction Funcs
###################################################################################################
def create_cost_function(p):
    GRIDWORLD, GRIDWORLD_COSTS, length_bounds, _, alphabet, num_cell_vars, num_cost_vars, num_label_vars, lo_locs = p
    cf = True

    # Create a function that fixes the cost of each cell_cost_var to
    # the cost of that cell. If cell is NONE, then cost is 0.
    for var_iter in range(1, length_bounds[1]+1):
        target_var = "Cell_" + str(var_iter)

        cf = And(cf, get_var_cost_function(target_var, p))

    # Create a function that fixes the variable "CostSum" to be the sum of
    # all cost variables
    # cost_sum = bfarray.exprzeros(num_cost_vars)
    # for var_iter in range(1, length_bounds[1]):
    #     print("Var_iter:", var_iter)
    #     target_var = "Cell_" + str(var_iter)

    #     var_array = bfarray.farray([exprvar(target_var + "_Cost", pos) for pos in range(num_cost_vars)])

    #     output = ripple_carry_add(var_array, cost_sum)

    #     cost_sum =  bfarray.farray([expr.simplify() for expr in output[0]])

    # cost_sum =  bfarray.farray([expr.simplify() for expr in output[0]])

    # for pos in range(num_cost_vars):
    #     print("Starting bit #", pos)
    #     cost_bit_formula = (cost_sum[pos] & exprvar("CostSum", pos)) | (~cost_sum[pos] & ~exprvar("CostSum", pos)) # ITE(cost_sum[pos], ~exprvar("CostSum", pos), exprvar("CostSum", pos))

    #     cf = (cf & cost_bit_formula).simplify()

    cost_sum = BitVec("CostSum", num_cost_vars) == sum([BitVec("Cell_" + str(var_iter) + "_Cost", num_cost_vars) for var_iter in range(1, length_bounds[1])])

    cf = And(cf, cost_sum)

    return cf

def get_var_cost_function(var, p):
    GRIDWORLD, GRIDWORLD_COSTS, length_bounds, cell_id_map_items, alphabet, num_cell_vars, num_cost_vars, num_label_vars, lo_locs = p

    cell_id_map = {key:val for (key,val) in cell_id_map_items}

    # GRIDWORLD_COSTS
    cell_cost_var = var + str("_Cost")

    # Create formula which is a disjunction of clauses
    # that each contain a conjunction of a cell assignment
    # along with an assignment of the appropriate cost to
    # the corresponding cell_cost variable.

    # Fixes the null cost to zero

    cell_valid_formula = get_var_equal_cell_formula(var, cell_id_map[None], p)
    cost_valid_formula = get_var_equal_cost_formula(cell_cost_var, 0, p)

    cell_and_cost_formula = Implies(cell_valid_formula, cost_valid_formula)

    var_cf = cell_and_cost_formula

    # Fixes each cell_cost_variable
    for y in range(len(GRIDWORLD)):
        for x in range(len(GRIDWORLD[0])):
            cell_cost = GRIDWORLD_COSTS[y][x]

            cell_valid_formula = get_var_equal_cell_formula(var, cell_id_map[(x,y)], p)
            cost_valid_formula = get_var_equal_cost_formula(cell_cost_var, cell_cost, p)

            cell_and_cost_formula = Implies(cell_valid_formula, cost_valid_formula)

            var_cf = And(var_cf, cell_and_cost_formula)

    return var_cf

###################################################################################################
# LabelFunction Funcs
###################################################################################################

def create_label_function(p):
    GRIDWORLD, GRIDWORLD_COSTS, length_bounds, cell_id_map_items, alphabet, num_cell_vars, num_cost_vars, num_label_vars, lo_locs = p

    cell_id_map = {key:val for (key,val) in cell_id_map_items}

    lf = True

    for lo_iter, lo_loc in enumerate(lo_locs):
        lo_id = cell_id_map[lo_loc]

        lo_indicator = create_label_indicator(lo_id, length_bounds[1], p)

        lo_implication = Implies(BitVec("LabelChoice", num_label_vars) == BitVecVal(lo_iter, num_label_vars), lo_indicator)

        lf = And(lf, lo_implication)

    return lf

def create_label_indicator(lo_id, length, p):
    # Creates a func that is true if and only if lo_id has aready been the first label cell encountered
    # or no other lo_id has been encountered and the current cell is lo_id
    if length == 1:
        return get_var_equal_cell_formula("Cell_1", lo_id, p)
    else:
        return Or(create_label_indicator(lo_id, length-1, p), And(create_no_prev_lo_function(length-1, p), get_var_equal_cell_formula("Cell_" + str(length), lo_id, p)))

def create_no_prev_lo_function(length, p):
    GRIDWORLD, GRIDWORLD_COSTS, length_bounds, cell_id_map_items, alphabet, num_cell_vars, num_cost_vars, num_label_vars, lo_locs = p

    cell_id_map = {key:val for (key,val) in cell_id_map_items}

    if length == 1:
        func = True
    else:
        func = create_no_prev_lo_function(length-1, p)

    for lo_loc in lo_locs:
        lo_id = cell_id_map[lo_loc]
        func = And(func, Not(get_var_equal_cell_formula("Cell_" + str(length), lo_id, p)))

    return func


###################################################################################################
# General Utility Funcs
###################################################################################################
@lru_cache(maxsize=None)
def get_var_equal_cell_formula(var, cell, p):
    GRIDWORLD, GRIDWORLD_COSTS, length_bounds, _, alphabet, num_cell_vars, num_cost_vars, num_label_vars, lo_locs = p

    # Returns a formula that is true if var is set to cell
    cell_valid = True

    for pos in range(num_cell_vars):
        if cell[pos] == 1:
            cell_valid = And(cell_valid, Bool(var + "[" + str(pos) + "]"))
        else:
            cell_valid = And(cell_valid, Not(Bool(var + "[" + str(pos) + "]")))

    return cell_valid

def get_var_equal_cost_formula(var, cost, p):
    GRIDWORLD, GRIDWORLD_COSTS, length_bounds, _, alphabet, num_cell_vars, num_cost_vars, num_label_vars, lo_locs = p
    # Returns a formula that is true if var is set to cell
    # cost_valid = expr(True)

    # for pos in range(num_cost_vars):
    #     if cost[pos] == 1:
    #         cost_valid = cost_valid & exprvar(var, pos)
    #     else:
    #         cost_valid = cost_valid & ~exprvar(var, pos)

    return BitVec(var, num_cost_vars) == BitVecVal(cost, num_cost_vars)


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

    start_loc = np.where(np.array(GRIDWORLD) == 2)

    point_a = None
    if len(improvisation) > 0:    
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

    plt.savefig("RP_ExtraLargeMap.png", bbox_inches="tight", pad_inches=0, dpi=1024)
    plt.show()


if __name__ == '__main__':
    GRIDWORLD =         (
                        (8, 0, 3, 0, 5),
                        (0, 1, 0, 1, 0),
                        (7, 0, 0, 0, 10),
                        (0, 9, 0, 4,  0),
                        (6, 0, 2, 0,  0)
                        )

    GRIDWORLD_COSTS =   (
                        (3, 2, 0, 2, 0),
                        (3, 0, 0, 0, 3),
                        (0, 1, 0, 2, 1),
                        (2, 0, 0, 0, 2),
                        (0, 1, 0, 1, 1)
                        )

    draw_improvisation([], GRIDWORLD, GRIDWORLD_COSTS)
