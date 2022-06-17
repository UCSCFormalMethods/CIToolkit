""" Tests for the Approximate LQCI Improviser class"""

import pytest

import z3

from citoolkit.improvisers.alqci import ALQCI

from citoolkit.specifications.z3_formula import Z3Formula
from citoolkit.labellingfunctions.labelling_z3_formula import Z3LabelFormula
from citoolkit.costfunctions.cost_z3_formula import Z3CostFormula


def test_alqci_basic():
    # Create a simple Z3 formula.
    x = z3.BitVec("x", 12)
    y = z3.BitVec("y", 12)
    z = z3.BitVec("z", 12)

    label_var = z3.BitVec("LabelVar", 2)
    cost_var = z3.BitVec("CostVar", 12)

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

    hc = Z3Formula(h_formula, h_variables)

    label_map = {"x_big": 0, "y_big": 1, "z_big": 2, "all_small": 3}

    lf = Z3LabelFormula(label_map, label_var)

    cf = Z3CostFormula(cost_var)

    improviser = ALQCI(hc, cf, lf, 400, (0.15, 0.35), \
                 {"x_big": (0, .01), "y_big": (0, .01), "z_big": (0, .01), "all_small": (0, .01)}, \
                 1.2, 0.8, 15, 0.2)

    for sample_iter in range(100):
        improviser.improvise()
