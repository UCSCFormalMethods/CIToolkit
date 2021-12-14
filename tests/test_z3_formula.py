""" Tests for the Z3Formula class"""

import pytest

import z3

from citoolkit.specifications.z3_formula import Z3Formula

def test_z3_formula_basic():
    # Create a simple Z3 formula.
    x = z3.BitVec("x", 8)
    y = z3.BitVec("y", 8)
    z = z3.BitVec("z", 8)

    x_big = z3.Bool("x_big")
    y_big = z3.Bool("y_big")
    z_big = z3.Bool("z_big")

    formula = True

    formula = z3.And(formula, (x + y + z == 200))
    formula = z3.And(formula, (z3.UGT(x, 64) == x_big))
    formula = z3.And(formula, (z3.UGT(y, 64) == y_big))
    formula = z3.And(formula, (z3.UGT(z, 64) == z_big))
    formula = z3.And(formula, (z == 63))

    main_variables = []
    main_variables.append(("x", "BitVec", 8))
    main_variables.append(("y", "BitVec", 8))
    main_variables.append(("z", "BitVec", 8))
    main_variables.append(("x_big", "Bool", 1))
    main_variables.append(("y_big", "Bool", 1))
    main_variables.append(("z_big", "Bool", 1))

    spec = Z3Formula(formula, main_variables)

    spec.language_size()

    for i in range(1000):
        print(i)
        sample = spec.sample(seed=i)
        print(sample)

        assert (sample["x"] > 64) == sample["x_big"]
        assert (sample["y"] > 64) == sample["y_big"]
        assert (sample["z"] > 64) == sample["z_big"]
        assert (sample["x"] + sample["y"] + sample["z"])%256 == 200

if __name__ == '__main__':
    test_z3_formula_basic()
