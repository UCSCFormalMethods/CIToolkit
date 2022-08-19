import time

import z3

from citoolkit.improvisers.alqci import ALQCI

from citoolkit.specifications.z3_formula import Z3Formula
from citoolkit.labellingfunctions.labelling_z3_formula import Z3LabelFormula
from citoolkit.costfunctions.cost_z3_formula import Z3CostFormula

import dill
import random

def main():
    start_time = time.time()

    # Create hard constraint, cost function, and label function.
    x = z3.BitVec("x", 24)
    y = z3.BitVec("y", 24)
    z = z3.BitVec("z", 24)

    label_var = z3.BitVec("LabelVar", 2)
    cost_var = z3.BitVec("CostVar", 24)

    h_formula = True
    h_formula = z3.And(h_formula, z3.ULT(x + y + z, 512))

    h_formula = z3.And(h_formula, z3.ULT(x, 512))
    h_formula = z3.And(h_formula, z3.ULT(y, 512))
    h_formula = z3.And(h_formula, z3.ULT(z, 512))
    h_formula = z3.And(h_formula, (z3.UGT(x, 300) == (label_var == 0)))
    h_formula = z3.And(h_formula, (z3.UGT(y, 400) == (label_var == 1)))
    h_formula = z3.And(h_formula, (z3.UGT(z, 500) == (label_var == 2)))
    h_formula = z3.And(h_formula, ((x + y + z) == cost_var))

    h_variables = [x,y,z]

    hard_constraint = Z3Formula(h_formula, h_variables)

    cost_function = Z3CostFormula(cost_var)

    label_map = {"x_big": 0, "y_big": 1, "z_big": 2, "all_small": 3}
    label_function = Z3LabelFormula(label_map, label_var)

    # Fix other improviser parameters
    bucket_ratio, counting_tol, sampling_tol, conf = 2, 0.8, 15, 0.2

    cost_bound = 400
    label_prob_bounds = (0.15, 0.35)
    word_prob_bounds = {"x_big": (0, .01), "y_big": (0, .01), "z_big": (0, .01), "all_small": (0, .01)}

    for _ in range(2):
        # Create improviser
        random.seed(42)

        improviser = ALQCI(hard_constraint, cost_function, label_function, bucket_ratio, counting_tol, sampling_tol, conf, num_threads=1, lazy=False)
        improviser.parameterize(cost_bound, label_prob_bounds, word_prob_bounds, num_threads=1)

        # for _ in range(5):
        #     improviser.improvise()

        print()

if __name__ == '__main__':
    main()
