""" Tests for the Z3Formula class"""

import pytest

import z3
import dill as pickle

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

    main_variables = [x,y,z,x_big,y_big,z_big]

    spec = Z3Formula(formula, main_variables)

    spec.language_size()

    #TODO Unigen bug, fix and set num iterations back to 1000
    for i in range(1000):
        sample = spec.sample(seed=i)

        assert (sample["x"] > 64) == sample["x_big"]
        assert (sample["y"] > 64) == sample["y_big"]
        assert (sample["z"] > 64) == sample["z_big"]
        assert (sample["x"] + sample["y"] + sample["z"])%256 == 200

def test_z3_formula_pickle():
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

    main_variables = [x,y,z,x_big,y_big,z_big]

    spec = Z3Formula(formula, main_variables)
    trivial_spec = Z3Formula(x == x, [x])

    spec = trivial_spec & spec

    p_spec = pickle.loads(pickle.dumps(spec))
    p_spec.explicit()
    p_spec = pickle.loads(pickle.dumps(p_spec))

    assert spec.explicit().formula.sexpr() == p_spec.explicit().formula.sexpr()

    p_spec.language_size()

    #TODO Unigen bug, fix and set num iterations back to 1000
    for i in range(1000):
        sample = p_spec.sample(seed=i)

        assert (sample["x"] > 64) == sample["x_big"]
        assert (sample["y"] > 64) == sample["y_big"]
        assert (sample["z"] > 64) == sample["z_big"]
        assert (sample["x"] + sample["y"] + sample["z"])%256 == 200

def test_z3_formula_operations():
    # Create 2 simple Z3 formulas.
    #TODO Fix this test and the accepts function
    return

    x = z3.BitVec("x", 8)
    y = z3.BitVec("y", 8)

    big_x = x == 200
    big_y = y == 200
    small_x = x == 5
    small_y = y == 5
    not_small_x = x == 10
    not_small_y = y == 11

    main_variables = [x,y]

    big_vars_formula = Z3Formula(z3.And(z3.UGT(x, 100), z3.UGT(y, 100)), main_variables=main_variables)
    small_vars_formula = Z3Formula(z3.And(z3.ULT(x, 10), z3.ULT(y, 10)), main_variables=main_variables)
    big_or_small_vars_formula = big_vars_formula | small_vars_formula
    big_and_small_vars_formula = big_vars_formula & small_vars_formula
    not_small_vars_formula = ~small_vars_formula

    assert not big_vars_formula.accepts(big_x)
    assert not big_vars_formula.accepts(big_y)
    assert big_vars_formula.accepts(z3.And(big_x, big_y))

    assert not small_vars_formula.accepts(small_x)
    assert not small_vars_formula.accepts(small_y)
    assert small_vars_formula.accepts(z3.And(small_x, small_y))

    assert big_or_small_vars_formula.accepts(z3.And(small_x, big_y))
    assert big_or_small_vars_formula.accepts(z3.And(big_x, small_y))

    assert not big_and_small_vars_formula.accepts(z3.And(small_x, small_y))
    assert not big_and_small_vars_formula.accepts(z3.And(big_x, big_y))

    assert not not_small_vars_formula.accepts(small_x)
    assert not not_small_vars_formula.accepts(z3.And(small_x, small_y))
    assert not_small_vars_formula.accepts(z3.And(not_small_x, not_small_y))
    assert not_small_vars_formula.accepts(z3.And(big_x, big_y))

if __name__ == '__main__':
    test_z3_formula_basic()
