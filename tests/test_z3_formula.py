""" Tests for the Z3Formula class"""

import pytest

import z3
import dill as pickle

from citoolkit.specifications.z3_formula import Z3Formula

def test_z3_formula_basic():
    """ Test basic functionality of the Z3Formula class """
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

    for i in range(1000):
        raw_sample = spec.sample(seed=i)

        sample = {str(var) :val for var,val in raw_sample.items()}

        assert (sample["x"] > 64) == sample["x_big"]
        assert (sample["y"] > 64) == sample["y_big"]
        assert (sample["z"] > 64) == sample["z_big"]
        assert (sample["x"] + sample["y"] + sample["z"])%256 == 200

def test_z3_formula_operations():
    """ Tests that binary operations work correctly on the Z3Formulas"""
    # Create 2 simple Z3 formulas.

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

    assert big_vars_formula.accepts({x: 200})
    assert big_vars_formula.accepts({y: 200})
    assert big_vars_formula.accepts({x: 200, y: 200})

    assert small_vars_formula.accepts({x: 5})
    assert small_vars_formula.accepts({y: 5})
    assert small_vars_formula.accepts({x: 5, y: 5})

    assert not big_or_small_vars_formula.accepts({x: 5, y: 200})
    assert not big_or_small_vars_formula.accepts({x: 200, y: 5})
    assert big_or_small_vars_formula.accepts({x: 200, y: 200})
    assert big_or_small_vars_formula.accepts({x: 5, y: 5})

    assert not big_and_small_vars_formula.accepts({x: 5, y: 200})
    assert not big_and_small_vars_formula.accepts({x: 200, y: 5})
    assert not big_and_small_vars_formula.accepts({x: 5, y: 5})
    assert not big_and_small_vars_formula.accepts({x: 200, y: 200})

    assert not not_small_vars_formula.accepts({x: 5})
    assert not not_small_vars_formula.accepts({x: 5, y: 5})
    assert not_small_vars_formula.accepts({x: 10, y: 11})
    assert not_small_vars_formula.accepts({x: 200, y: 200})
