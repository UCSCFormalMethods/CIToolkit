""" Tests for the Approximate LQCI Improviser class"""

import random

import z3

import pytest
from hypothesis import given, settings
from hypothesis.strategies import booleans, integers

from citoolkit.improvisers.alqci import ALQCI

from citoolkit.specifications.z3_formula import Z3Formula
from citoolkit.labellingfunctions.labelling_z3_formula import LabellingZ3Formula
from citoolkit.costfunctions.cost_z3_formula import CostZ3Formula

@given(num_threads=integers(1,2), lazy=booleans())
@settings(deadline=None)
def test_alqci_improvise(num_threads, lazy):
    """ Tests a simple Approximate Labelled Quantitative CI Instance """
    # Create hard constraint, cost function, and label function.
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

    hard_constraint = Z3Formula(h_formula, h_variables)

    cost_function = CostZ3Formula(cost_var)

    label_map = {"x_big": 0, "y_big": 1, "z_big": 2, "all_small": 3}
    label_function = LabellingZ3Formula(label_map, label_var)

    # Fix other improviser parameters
    bucket_ratio, counting_tol, sampling_tol, conf = 2, 0.8, 15, 0.2

    cost_bound = 400
    label_prob_bounds = (0.15, 0.35)
    word_prob_bounds = {"x_big": (0, .01), "y_big": (0, .01), "z_big": (0, .01), "all_small": (0, .01)}


    # Create improviser
    improviser = ALQCI(hard_constraint, cost_function, label_function, bucket_ratio, counting_tol, sampling_tol, conf, num_threads=num_threads, lazy=lazy)
    improviser.parameterize(cost_bound, label_prob_bounds, word_prob_bounds, num_threads=num_threads)

    for _ in range(100):
        # Sample word
        word = improviser.improvise()

        # Check word validity
        assert hard_constraint.accepts(word)

@pytest.mark.skip(reason="Fails due to Z3Py Nondeterminism")
@given(num_threads=integers(1,2), lazy=booleans())
@settings(deadline=None)
def test_alqci_reproducible(num_threads, lazy):
    """ Tests a simple Approximate Labelled Quantitative CI instance gives reproducible results
    modulo the same random state
    """
    # Store the initial random state and initialize a variable to store sampled words for
    # comparison across runs.
    state = random.getstate()
    sampled_words = None
    global_context = z3.Context()

    for _ in range(3):
        print()
        # Reset the random state
        random.setstate(state)

        # Create hard constraint, cost function, and label function.
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

        hard_constraint = Z3Formula(h_formula, h_variables)

        cost_function = CostZ3Formula(cost_var)

        label_map = {"x_big": 0, "y_big": 1, "z_big": 2, "all_small": 3}
        label_function = LabellingZ3Formula(label_map, label_var)

        # Fix other improviser parameters
        bucket_ratio, counting_tol, sampling_tol, conf = 1.2, 0.8, 15, 0.2

        cost_bound = 400
        label_prob_bounds = (0.15, 0.35)
        word_prob_bounds = {"x_big": (0, .01), "y_big": (0, .01), "z_big": (0, .01), "all_small": (0, .01)}

        # Create improviser
        improviser = ALQCI(hard_constraint, cost_function, label_function, bucket_ratio, counting_tol, sampling_tol, conf, num_threads=num_threads, lazy=lazy)
        improviser.parameterize(cost_bound, label_prob_bounds, word_prob_bounds, num_threads=num_threads)

        # Sample 10 words from the improviser
        new_words = []
        for _ in range(10):
            raw_words = improviser.improvise()
            new_words.append({var.translate(global_context): val for var, val in raw_words.items()})

        # Check that these 10 words are consistent across all runs, which they should be since we have the same random state.
        if sampled_words is None:
            sampled_words = new_words
        else:
            assert new_words == sampled_words
