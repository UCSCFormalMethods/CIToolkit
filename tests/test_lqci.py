""" Tests for the LQCI and MELQCI classes"""

import random
from fractions import Fraction
from cvxpy.error import SolverError

import pytest
from hypothesis import given, settings
from hypothesis.strategies import booleans, integers, fractions, tuples, shared

from citoolkit.improvisers.improviser import InfeasibleImproviserError, InfeasibleCostError,\
    InfeasibleLabelRandomnessError, InfeasibleWordRandomnessError
from citoolkit.improvisers.lqci import LQCI, MELQCI
from citoolkit.specifications.dfa import Dfa
from citoolkit.costfunctions.static_cost_dfa import StaticCostDfa
from citoolkit.labellingfunctions.labelling_dfa import LabellingDfa

from .test_utils import *

###################################################################################################
# Basic Tests
###################################################################################################

## LQCI Tests ##

@given(num_threads=integers(1,2), lazy=booleans())
@settings(deadline=None)
def test_lqci_improvise(num_threads, lazy):
    """ Test a simple Labelled Quantitative CI instance. """
    # Create a hard constraint Dfa that accepts all words start with "0"
    alphabet = {"0", "1", "2"}
    h_states = {"Start", "Accept", "Reject"}
    h_accepting_states = {"Accept"}
    h_start_state = "Start"

    h_transitions = {}
    h_transitions[("Start", "0")] = "Accept"
    h_transitions[("Start", "1")] = "Reject"
    h_transitions[("Start", "2")] = "Reject"
    h_transitions[("Accept", "0")] = "Accept"
    h_transitions[("Accept", "1")] = "Accept"
    h_transitions[("Accept", "2")] = "Accept"
    h_transitions[("Reject", "0")] = "Reject"
    h_transitions[("Reject", "1")] = "Reject"
    h_transitions[("Reject", "2")] = "Reject"

    hard_constraint = Dfa(alphabet, h_states, h_accepting_states, h_start_state, h_transitions)

    # Create a cost function that assigns a cost based off of how many "1" symbols there are,
    # up to 3.
    k_states = {"Seen0", "Seen1", "Seen2", "Seen3"}
    k_accepting_states = {"Seen0", "Seen1", "Seen2", "Seen3"}
    k_start_state = "Seen0"

    k_transitions = {}
    k_transitions[("Seen0", "0")] = "Seen0"
    k_transitions[("Seen0", "1")] = "Seen1"
    k_transitions[("Seen0", "2")] = "Seen0"
    k_transitions[("Seen1", "0")] = "Seen1"
    k_transitions[("Seen1", "1")] = "Seen2"
    k_transitions[("Seen1", "2")] = "Seen1"
    k_transitions[("Seen2", "0")] = "Seen2"
    k_transitions[("Seen2", "1")] = "Seen3"
    k_transitions[("Seen2", "2")] = "Seen2"
    k_transitions[("Seen3", "0")] = "Seen3"
    k_transitions[("Seen3", "1")] = "Seen3"
    k_transitions[("Seen3", "2")] = "Seen3"

    cost_dfa = Dfa(alphabet, k_states, k_accepting_states, k_start_state, k_transitions)

    cost_map = {}
    cost_map["Seen0"] = 0
    cost_map["Seen1"] = 1
    cost_map["Seen2"] = 2
    cost_map["Seen3"] = 3

    cost_func = StaticCostDfa(cost_dfa, cost_map)

    # Create a label function that assigns a label based off of when the first "2" symbol
    # is seen. Does not assign a label to strings with no 2 symbols.

    l_states = {"Seen0", "Seen1", "Seen2", "Seen3", "Seen2_Locked", "Seen3_Locked", "Seen4_Locked", "Sink"}
    l_accepting_states = {"Seen2_Locked", "Seen3_Locked", "Seen4_Locked"}
    l_start_state = "Seen0"

    l_transitions = {}
    l_transitions[("Seen0", "0")] = "Seen1"
    l_transitions[("Seen0", "1")] = "Seen1"
    l_transitions[("Seen0", "2")] = "Seen1"
    l_transitions[("Seen1", "0")] = "Seen2"
    l_transitions[("Seen1", "1")] = "Seen2"
    l_transitions[("Seen1", "2")] = "Seen2_Locked"
    l_transitions[("Seen2", "0")] = "Seen3"
    l_transitions[("Seen2", "1")] = "Seen3"
    l_transitions[("Seen2", "2")] = "Seen3_Locked"
    l_transitions[("Seen3", "0")] = "Sink"
    l_transitions[("Seen3", "1")] = "Sink"
    l_transitions[("Seen3", "2")] = "Seen4_Locked"
    l_transitions[("Seen2_Locked", "0")] = "Seen2_Locked"
    l_transitions[("Seen2_Locked", "1")] = "Seen2_Locked"
    l_transitions[("Seen2_Locked", "2")] = "Seen2_Locked"
    l_transitions[("Seen3_Locked", "0")] = "Seen3_Locked"
    l_transitions[("Seen3_Locked", "1")] = "Seen3_Locked"
    l_transitions[("Seen3_Locked", "2")] = "Seen3_Locked"
    l_transitions[("Seen4_Locked", "0")] = "Seen4_Locked"
    l_transitions[("Seen4_Locked", "1")] = "Seen4_Locked"
    l_transitions[("Seen4_Locked", "2")] = "Seen4_Locked"
    l_transitions[("Sink", "0")] = "Sink"
    l_transitions[("Sink", "1")] = "Sink"
    l_transitions[("Sink", "2")] = "Sink"

    label_dfa = Dfa(alphabet, l_states, l_accepting_states, l_start_state, l_transitions)

    label_map = {}
    label_map["Seen2_Locked"] = "Label_Pos2"
    label_map["Seen3_Locked"] = "Label_Pos3"
    label_map["Seen4_Locked"] = "Label_Pos4"

    label_func = LabellingDfa(label_dfa, label_map)

    # Fix remaining improviser parameters
    length_bounds = (1,4)
    cost_bound = 0.5
    label_prob_bounds = (Fraction(1,5), Fraction(1,2))
    word_prob_bounds = {"Label_Pos2":(Fraction(1,13), Fraction(1,13)), "Label_Pos3":(Fraction(1,20),Fraction(1,4)), "Label_Pos4":(Fraction(1,6),Fraction(5,12))}

    improviser = LQCI(hard_constraint, cost_func, label_func, length_bounds, num_threads=num_threads, lazy=lazy)
    improviser.parameterize(cost_bound, label_prob_bounds, word_prob_bounds, num_threads=num_threads)

    # Sample a collection of words from the improviser.
    improvisation_count = {}
    label_count = {label:0 for label in label_func.labels}
    cost_accumulator = 0

    label_improvisation_map = {label:set() for label in label_func.labels}

    for _ in range(100000):
        word = improviser.improvise()

        assert hard_constraint.accepts(word)
        assert label_func.label(word) is not None
        assert cost_func.cost(word) is not None

        if word not in improvisation_count:
            improvisation_count[word] = 1
        else:
            improvisation_count[word] += 1

        label_count[label_func.label(word)] += 1
        cost_accumulator += cost_func.cost(word)

        label_improvisation_map[label_func.label(word)].add(word)

    # Check that sampled word probabilities and average cost are valid.
    for label in label_func.labels:
        label_sampled_prob = label_count[label]/100000
        assert label_prob_bounds[0]-.01 <= label_sampled_prob <= label_prob_bounds[1]+.01

        for word in label_improvisation_map[label]:
            cond_word_sampled_prob = (improvisation_count[word]/100000)/label_sampled_prob
            assert word_prob_bounds[label][0]-0.1 <= cond_word_sampled_prob <= word_prob_bounds[label][1]+0.1

    assert cost_accumulator/100000 < cost_bound*1.01

@given(num_threads=integers(1,2), lazy=booleans())
@settings(deadline=None)
def test_lqci_reproducible(num_threads, lazy):
    """ Test that a simple Labelled Quantitative CI instance gives reproducible results
    modulo the same random state
    """
    # Store the initial random state and initialize a variable to store sampled words for
    # comparison across runs.
    state = random.getstate()
    sampled_words = None

    for _ in range(3):
        # Reset the random state
        random.setstate(state)

        # Create a hard constraint Dfa that accepts all words start with "0"
        alphabet = {"0", "1", "2"}
        h_states = {"Start", "Accept", "Reject"}
        h_accepting_states = {"Accept"}
        h_start_state = "Start"

        h_transitions = {}
        h_transitions[("Start", "0")] = "Accept"
        h_transitions[("Start", "1")] = "Reject"
        h_transitions[("Start", "2")] = "Reject"
        h_transitions[("Accept", "0")] = "Accept"
        h_transitions[("Accept", "1")] = "Accept"
        h_transitions[("Accept", "2")] = "Accept"
        h_transitions[("Reject", "0")] = "Reject"
        h_transitions[("Reject", "1")] = "Reject"
        h_transitions[("Reject", "2")] = "Reject"

        hard_constraint = Dfa(alphabet, h_states, h_accepting_states, h_start_state, h_transitions)

        # Create a cost function that assigns a cost based off of how many "1" symbols there are,
        # up to 3.
        k_states = {"Seen0", "Seen1", "Seen2", "Seen3"}
        k_accepting_states = {"Seen0", "Seen1", "Seen2", "Seen3"}
        k_start_state = "Seen0"

        k_transitions = {}
        k_transitions[("Seen0", "0")] = "Seen0"
        k_transitions[("Seen0", "1")] = "Seen1"
        k_transitions[("Seen0", "2")] = "Seen0"
        k_transitions[("Seen1", "0")] = "Seen1"
        k_transitions[("Seen1", "1")] = "Seen2"
        k_transitions[("Seen1", "2")] = "Seen1"
        k_transitions[("Seen2", "0")] = "Seen2"
        k_transitions[("Seen2", "1")] = "Seen3"
        k_transitions[("Seen2", "2")] = "Seen2"
        k_transitions[("Seen3", "0")] = "Seen3"
        k_transitions[("Seen3", "1")] = "Seen3"
        k_transitions[("Seen3", "2")] = "Seen3"

        cost_dfa = Dfa(alphabet, k_states, k_accepting_states, k_start_state, k_transitions)

        cost_map = {}
        cost_map["Seen0"] = 0
        cost_map["Seen1"] = 1
        cost_map["Seen2"] = 2
        cost_map["Seen3"] = 3

        cost_func = StaticCostDfa(cost_dfa, cost_map)

        # Create a label function that assigns a label based off of when the first "2" symbol
        # is seen. Does not assign a label to strings with no 2 symbols.

        l_states = {"Seen0", "Seen1", "Seen2", "Seen3", "Seen2_Locked", "Seen3_Locked", "Seen4_Locked", "Sink"}
        l_accepting_states = {"Seen2_Locked", "Seen3_Locked", "Seen4_Locked"}
        l_start_state = "Seen0"

        l_transitions = {}
        l_transitions[("Seen0", "0")] = "Seen1"
        l_transitions[("Seen0", "1")] = "Seen1"
        l_transitions[("Seen0", "2")] = "Seen1"
        l_transitions[("Seen1", "0")] = "Seen2"
        l_transitions[("Seen1", "1")] = "Seen2"
        l_transitions[("Seen1", "2")] = "Seen2_Locked"
        l_transitions[("Seen2", "0")] = "Seen3"
        l_transitions[("Seen2", "1")] = "Seen3"
        l_transitions[("Seen2", "2")] = "Seen3_Locked"
        l_transitions[("Seen3", "0")] = "Sink"
        l_transitions[("Seen3", "1")] = "Sink"
        l_transitions[("Seen3", "2")] = "Seen4_Locked"
        l_transitions[("Seen2_Locked", "0")] = "Seen2_Locked"
        l_transitions[("Seen2_Locked", "1")] = "Seen2_Locked"
        l_transitions[("Seen2_Locked", "2")] = "Seen2_Locked"
        l_transitions[("Seen3_Locked", "0")] = "Seen3_Locked"
        l_transitions[("Seen3_Locked", "1")] = "Seen3_Locked"
        l_transitions[("Seen3_Locked", "2")] = "Seen3_Locked"
        l_transitions[("Seen4_Locked", "0")] = "Seen4_Locked"
        l_transitions[("Seen4_Locked", "1")] = "Seen4_Locked"
        l_transitions[("Seen4_Locked", "2")] = "Seen4_Locked"
        l_transitions[("Sink", "0")] = "Sink"
        l_transitions[("Sink", "1")] = "Sink"
        l_transitions[("Sink", "2")] = "Sink"

        label_dfa = Dfa(alphabet, l_states, l_accepting_states, l_start_state, l_transitions)

        label_map = {}
        label_map["Seen2_Locked"] = "Label_Pos2"
        label_map["Seen3_Locked"] = "Label_Pos3"
        label_map["Seen4_Locked"] = "Label_Pos4"

        label_func = LabellingDfa(label_dfa, label_map)

        # Fix remaining improviser parameters
        length_bounds = (1,4)
        cost_bound = 0.5
        label_prob_bounds = (Fraction(1,5), Fraction(1,2))
        word_prob_bounds = {"Label_Pos2":(Fraction(1,13), Fraction(1,13)), "Label_Pos3":(Fraction(1,20),Fraction(1,4)), "Label_Pos4":(Fraction(1,6),Fraction(5,12))}

        improviser = LQCI(hard_constraint, cost_func, label_func, length_bounds, num_threads=num_threads, lazy=lazy)
        improviser.parameterize(cost_bound, label_prob_bounds, word_prob_bounds, num_threads=num_threads)

        # Sample 10 words from the improviser
        new_words = []
        for _ in range(10):
            new_words.append(improviser.improvise())

        # Check that these 10 words are consistent across all runs, which they should be since we have the same random state.
        if sampled_words is None:
            sampled_words = new_words
        else:
            assert new_words == sampled_words

@given(num_threads=integers(1,2), lazy=booleans())
@settings(deadline=None)
def test_lqci_infeasible(num_threads, lazy):
    """ Test that different infeasible Labelled Quantitative CI
    problems correctly raise an exception.
    """
    # Create a hard constraint Dfa that accepts all words start with "0"
    alphabet = {"0", "1", "2"}
    h_states = {"Start", "Accept", "Reject"}
    h_accepting_states = {"Accept"}
    h_start_state = "Start"

    h_transitions = {}
    h_transitions[("Start", "0")] = "Accept"
    h_transitions[("Start", "1")] = "Reject"
    h_transitions[("Start", "2")] = "Reject"
    h_transitions[("Accept", "0")] = "Accept"
    h_transitions[("Accept", "1")] = "Accept"
    h_transitions[("Accept", "2")] = "Accept"
    h_transitions[("Reject", "0")] = "Reject"
    h_transitions[("Reject", "1")] = "Reject"
    h_transitions[("Reject", "2")] = "Reject"

    hard_constraint = Dfa(alphabet, h_states, h_accepting_states, h_start_state, h_transitions)

    # Create a cost function that assigns a cost based off of how many "1" symbols there are,
    # up to 3.
    k_states = {"Seen0", "Seen1", "Seen2", "Seen3"}
    k_accepting_states = {"Seen0", "Seen1", "Seen2", "Seen3"}
    k_start_state = "Seen0"

    k_transitions = {}
    k_transitions[("Seen0", "0")] = "Seen0"
    k_transitions[("Seen0", "1")] = "Seen1"
    k_transitions[("Seen0", "2")] = "Seen0"
    k_transitions[("Seen1", "0")] = "Seen1"
    k_transitions[("Seen1", "1")] = "Seen2"
    k_transitions[("Seen1", "2")] = "Seen1"
    k_transitions[("Seen2", "0")] = "Seen2"
    k_transitions[("Seen2", "1")] = "Seen3"
    k_transitions[("Seen2", "2")] = "Seen2"
    k_transitions[("Seen3", "0")] = "Seen3"
    k_transitions[("Seen3", "1")] = "Seen3"
    k_transitions[("Seen3", "2")] = "Seen3"

    cost_dfa = Dfa(alphabet, k_states, k_accepting_states, k_start_state, k_transitions)

    cost_map = {}
    cost_map["Seen0"] = 0
    cost_map["Seen1"] = 1
    cost_map["Seen2"] = 2
    cost_map["Seen3"] = 3

    cost_func = StaticCostDfa(cost_dfa, cost_map)

    # Create a label function that assigns a label based off of when the first "2" symbol
    # is seen. Does not assign a label to strings with no 2 symbols.

    l_states = {"Seen0", "Seen1", "Seen2", "Seen3", "Seen2_Locked", "Seen3_Locked", "Seen4_Locked", "Sink"}
    l_accepting_states = {"Seen2_Locked", "Seen3_Locked", "Seen4_Locked"}
    l_start_state = "Seen0"

    l_transitions = {}
    l_transitions[("Seen0", "0")] = "Seen1"
    l_transitions[("Seen0", "1")] = "Seen1"
    l_transitions[("Seen0", "2")] = "Seen1"
    l_transitions[("Seen1", "0")] = "Seen2"
    l_transitions[("Seen1", "1")] = "Seen2"
    l_transitions[("Seen1", "2")] = "Seen2_Locked"
    l_transitions[("Seen2", "0")] = "Seen3"
    l_transitions[("Seen2", "1")] = "Seen3"
    l_transitions[("Seen2", "2")] = "Seen3_Locked"
    l_transitions[("Seen3", "0")] = "Sink"
    l_transitions[("Seen3", "1")] = "Sink"
    l_transitions[("Seen3", "2")] = "Seen4_Locked"
    l_transitions[("Seen2_Locked", "0")] = "Seen2_Locked"
    l_transitions[("Seen2_Locked", "1")] = "Seen2_Locked"
    l_transitions[("Seen2_Locked", "2")] = "Seen2_Locked"
    l_transitions[("Seen3_Locked", "0")] = "Seen3_Locked"
    l_transitions[("Seen3_Locked", "1")] = "Seen3_Locked"
    l_transitions[("Seen3_Locked", "2")] = "Seen3_Locked"
    l_transitions[("Seen4_Locked", "0")] = "Seen4_Locked"
    l_transitions[("Seen4_Locked", "1")] = "Seen4_Locked"
    l_transitions[("Seen4_Locked", "2")] = "Seen4_Locked"
    l_transitions[("Sink", "0")] = "Sink"
    l_transitions[("Sink", "1")] = "Sink"
    l_transitions[("Sink", "2")] = "Sink"

    label_dfa = Dfa(alphabet, l_states, l_accepting_states, l_start_state, l_transitions)

    label_map = {}
    label_map["Seen2_Locked"] = "Label_Pos2"
    label_map["Seen3_Locked"] = "Label_Pos3"
    label_map["Seen4_Locked"] = "Label_Pos4"

    label_func = LabellingDfa(label_dfa, label_map)

    # Fix default improviser parameters
    length_bounds = (1,4)
    cost_bound = 0.5
    label_prob_bounds = (Fraction(1,5), Fraction(1,2))
    word_prob_bounds = {"Label_Pos2":(Fraction(1,13), Fraction(1,13)), "Label_Pos3":(Fraction(1,20),Fraction(1,4)), "Label_Pos4":(Fraction(1,6),Fraction(5,12))}

    # Ensure the base LQCI problem is feasible
    improviser = LQCI(hard_constraint, cost_func, label_func, length_bounds)
    improviser.parameterize(cost_bound, label_prob_bounds, word_prob_bounds)

    # Check that various parameter tweaks that render the
    # problem infeasible are identified by the improviser.
    with pytest.raises(InfeasibleCostError):
        improviser = LQCI(hard_constraint, cost_func, label_func, length_bounds, num_threads=num_threads, lazy=lazy)
        improviser.parameterize(0.3, label_prob_bounds, word_prob_bounds, num_threads=num_threads)

    with pytest.raises(InfeasibleCostError):
        improviser = LQCI(hard_constraint, cost_func, label_func, length_bounds, num_threads=num_threads, lazy=lazy)
        improviser.parameterize(cost_bound, (Fraction(1,3), Fraction(1,3)), word_prob_bounds, num_threads=num_threads)

    with pytest.raises(InfeasibleLabelRandomnessError):
        improviser = LQCI(hard_constraint, cost_func, label_func, length_bounds, num_threads=num_threads, lazy=lazy)
        improviser.parameterize(cost_bound, (Fraction(1,4), Fraction(1,4)), word_prob_bounds, num_threads=num_threads)

    with pytest.raises(InfeasibleCostError):
        strict_word_prob_bounds = {"Label_Pos2":(Fraction(1,13), Fraction(1,13)), "Label_Pos3":(Fraction(1,20),Fraction(1,4)), "Label_Pos4":(Fraction(1,4),Fraction(1,4))}
        improviser = LQCI(hard_constraint, cost_func, label_func, length_bounds, num_threads=num_threads, lazy=lazy)
        improviser.parameterize(cost_bound, label_prob_bounds, strict_word_prob_bounds, num_threads=num_threads)

    with pytest.raises(InfeasibleWordRandomnessError):
        infeasible_word_prob_bounds = {"Label_Pos2":(Fraction(1,12), Fraction(1,12)), "Label_Pos3":(Fraction(1,20),Fraction(1,4)), "Label_Pos4":(Fraction(1,6),Fraction(5,12))}
        improviser = LQCI(hard_constraint, cost_func, label_func, length_bounds, num_threads=num_threads, lazy=lazy)
        improviser.parameterize(cost_bound, label_prob_bounds, infeasible_word_prob_bounds, num_threads=num_threads)

## MELQCI Tests ##
@given(num_threads=integers(1,2))
@settings(deadline=None)
def test_melqci_improvise(num_threads):
    """ Test a simple Labelled Quantitative CI instance. """
    # Create a hard constraint Dfa that accepts all words start with "0"
    alphabet = {"0", "1", "2"}
    h_states = {"Start", "Accept", "Reject"}
    h_accepting_states = {"Accept"}
    h_start_state = "Start"

    h_transitions = {}
    h_transitions[("Start", "0")] = "Accept"
    h_transitions[("Start", "1")] = "Reject"
    h_transitions[("Start", "2")] = "Reject"
    h_transitions[("Accept", "0")] = "Accept"
    h_transitions[("Accept", "1")] = "Accept"
    h_transitions[("Accept", "2")] = "Accept"
    h_transitions[("Reject", "0")] = "Reject"
    h_transitions[("Reject", "1")] = "Reject"
    h_transitions[("Reject", "2")] = "Reject"

    hard_constraint = Dfa(alphabet, h_states, h_accepting_states, h_start_state, h_transitions)

    # Create a cost function that assigns a cost based off of how many "1" symbols there are,
    # up to 3.
    k_states = {"Seen0", "Seen1", "Seen2", "Seen3"}
    k_accepting_states = {"Seen0", "Seen1", "Seen2", "Seen3"}
    k_start_state = "Seen0"

    k_transitions = {}
    k_transitions[("Seen0", "0")] = "Seen0"
    k_transitions[("Seen0", "1")] = "Seen1"
    k_transitions[("Seen0", "2")] = "Seen0"
    k_transitions[("Seen1", "0")] = "Seen1"
    k_transitions[("Seen1", "1")] = "Seen2"
    k_transitions[("Seen1", "2")] = "Seen1"
    k_transitions[("Seen2", "0")] = "Seen2"
    k_transitions[("Seen2", "1")] = "Seen3"
    k_transitions[("Seen2", "2")] = "Seen2"
    k_transitions[("Seen3", "0")] = "Seen3"
    k_transitions[("Seen3", "1")] = "Seen3"
    k_transitions[("Seen3", "2")] = "Seen3"

    cost_dfa = Dfa(alphabet, k_states, k_accepting_states, k_start_state, k_transitions)

    cost_map = {}
    cost_map["Seen0"] = 0
    cost_map["Seen1"] = 1
    cost_map["Seen2"] = 2
    cost_map["Seen3"] = 3

    cost_func = StaticCostDfa(cost_dfa, cost_map)

    # Create a label function that assigns a label based off of when the first "2" symbol
    # is seen. Does not assign a label to strings with no 2 symbols.

    l_states = {"Seen0", "Seen1", "Seen2", "Seen3", "Seen2_Locked", "Seen3_Locked", "Seen4_Locked", "Sink"}
    l_accepting_states = {"Seen2_Locked", "Seen3_Locked", "Seen4_Locked"}
    l_start_state = "Seen0"

    l_transitions = {}
    l_transitions[("Seen0", "0")] = "Seen1"
    l_transitions[("Seen0", "1")] = "Seen1"
    l_transitions[("Seen0", "2")] = "Seen1"
    l_transitions[("Seen1", "0")] = "Seen2"
    l_transitions[("Seen1", "1")] = "Seen2"
    l_transitions[("Seen1", "2")] = "Seen2_Locked"
    l_transitions[("Seen2", "0")] = "Seen3"
    l_transitions[("Seen2", "1")] = "Seen3"
    l_transitions[("Seen2", "2")] = "Seen3_Locked"
    l_transitions[("Seen3", "0")] = "Sink"
    l_transitions[("Seen3", "1")] = "Sink"
    l_transitions[("Seen3", "2")] = "Seen4_Locked"
    l_transitions[("Seen2_Locked", "0")] = "Seen2_Locked"
    l_transitions[("Seen2_Locked", "1")] = "Seen2_Locked"
    l_transitions[("Seen2_Locked", "2")] = "Seen2_Locked"
    l_transitions[("Seen3_Locked", "0")] = "Seen3_Locked"
    l_transitions[("Seen3_Locked", "1")] = "Seen3_Locked"
    l_transitions[("Seen3_Locked", "2")] = "Seen3_Locked"
    l_transitions[("Seen4_Locked", "0")] = "Seen4_Locked"
    l_transitions[("Seen4_Locked", "1")] = "Seen4_Locked"
    l_transitions[("Seen4_Locked", "2")] = "Seen4_Locked"
    l_transitions[("Sink", "0")] = "Sink"
    l_transitions[("Sink", "1")] = "Sink"
    l_transitions[("Sink", "2")] = "Sink"

    label_dfa = Dfa(alphabet, l_states, l_accepting_states, l_start_state, l_transitions)

    label_map = {}
    label_map["Seen2_Locked"] = "Label_Pos2"
    label_map["Seen3_Locked"] = "Label_Pos3"
    label_map["Seen4_Locked"] = "Label_Pos4"

    label_func = LabellingDfa(label_dfa, label_map)

    # Fix remaining improviser parameters
    length_bounds = (1,4)
    cost_bound = 0.5
    label_prob_bounds = (Fraction(1,5), Fraction(1,2))

    improviser = MELQCI(hard_constraint, cost_func, label_func, length_bounds, num_threads=num_threads)
    improviser.parameterize(cost_bound, label_prob_bounds, num_threads=num_threads)

    # Sample a collection of words from the improviser.
    improvisation_count = {}
    label_count = {label:0 for label in label_func.labels}
    cost_accumulator = 0

    label_improvisation_map = {label:set() for label in label_func.labels}

    for _ in range(100000):
        word = improviser.improvise()

        assert hard_constraint.accepts(word)
        assert label_func.label(word) is not None
        assert cost_func.cost(word) is not None

        if word not in improvisation_count:
            improvisation_count[word] = 1
        else:
            improvisation_count[word] += 1

        label_count[label_func.label(word)] += 1
        cost_accumulator += cost_func.cost(word)

        label_improvisation_map[label_func.label(word)].add(word)

    # Check that sampled word probabilities and average cost are valid.
    for label in label_func.labels:
        label_sampled_prob = label_count[label]/100000
        assert label_prob_bounds[0]-.01 <= label_sampled_prob <= label_prob_bounds[1]+.01

    assert cost_accumulator/100000 <= cost_bound*1.01

@given(num_threads=integers(1,2))
@settings(deadline=None)
def test_melqci_infeasible(num_threads):
    """ Test a simple Labelled Quantitative CI instance. """
    # Create a hard constraint Dfa that accepts all words start with "0"
    alphabet = {"0", "1", "2"}
    h_states = {"Start", "Accept", "Reject"}
    h_accepting_states = {"Accept"}
    h_start_state = "Start"

    h_transitions = {}
    h_transitions[("Start", "0")] = "Accept"
    h_transitions[("Start", "1")] = "Reject"
    h_transitions[("Start", "2")] = "Reject"
    h_transitions[("Accept", "0")] = "Accept"
    h_transitions[("Accept", "1")] = "Accept"
    h_transitions[("Accept", "2")] = "Accept"
    h_transitions[("Reject", "0")] = "Reject"
    h_transitions[("Reject", "1")] = "Reject"
    h_transitions[("Reject", "2")] = "Reject"

    hard_constraint = Dfa(alphabet, h_states, h_accepting_states, h_start_state, h_transitions)

    # Create a cost function that assigns a cost based off of how many "1" symbols there are,
    # up to 3.
    k_states = {"Seen0", "Seen1", "Seen2", "Seen3"}
    k_accepting_states = {"Seen0", "Seen1", "Seen2", "Seen3"}
    k_start_state = "Seen0"

    k_transitions = {}
    k_transitions[("Seen0", "0")] = "Seen0"
    k_transitions[("Seen0", "1")] = "Seen1"
    k_transitions[("Seen0", "2")] = "Seen0"
    k_transitions[("Seen1", "0")] = "Seen1"
    k_transitions[("Seen1", "1")] = "Seen2"
    k_transitions[("Seen1", "2")] = "Seen1"
    k_transitions[("Seen2", "0")] = "Seen2"
    k_transitions[("Seen2", "1")] = "Seen3"
    k_transitions[("Seen2", "2")] = "Seen2"
    k_transitions[("Seen3", "0")] = "Seen3"
    k_transitions[("Seen3", "1")] = "Seen3"
    k_transitions[("Seen3", "2")] = "Seen3"

    cost_dfa = Dfa(alphabet, k_states, k_accepting_states, k_start_state, k_transitions)

    cost_map = {}
    cost_map["Seen0"] = 1
    cost_map["Seen1"] = 2
    cost_map["Seen2"] = 3
    cost_map["Seen3"] = 4

    cost_func = StaticCostDfa(cost_dfa, cost_map)

    # Create a label function that assigns a label based off of when the first "2" symbol
    # is seen. Does not assign a label to strings with no 2 symbols.

    l_states = {"Seen0", "Seen1", "Seen2", "Seen3", "Seen2_Locked", "Seen3_Locked", "Seen4_Locked", "Sink"}
    l_accepting_states = {"Seen2_Locked", "Seen3_Locked", "Seen4_Locked"}
    l_start_state = "Seen0"

    l_transitions = {}
    l_transitions[("Seen0", "0")] = "Seen1"
    l_transitions[("Seen0", "1")] = "Seen1"
    l_transitions[("Seen0", "2")] = "Seen1"
    l_transitions[("Seen1", "0")] = "Seen2"
    l_transitions[("Seen1", "1")] = "Seen2"
    l_transitions[("Seen1", "2")] = "Seen2_Locked"
    l_transitions[("Seen2", "0")] = "Seen3"
    l_transitions[("Seen2", "1")] = "Seen3"
    l_transitions[("Seen2", "2")] = "Seen3_Locked"
    l_transitions[("Seen3", "0")] = "Sink"
    l_transitions[("Seen3", "1")] = "Sink"
    l_transitions[("Seen3", "2")] = "Seen4_Locked"
    l_transitions[("Seen2_Locked", "0")] = "Seen2_Locked"
    l_transitions[("Seen2_Locked", "1")] = "Seen2_Locked"
    l_transitions[("Seen2_Locked", "2")] = "Seen2_Locked"
    l_transitions[("Seen3_Locked", "0")] = "Seen3_Locked"
    l_transitions[("Seen3_Locked", "1")] = "Seen3_Locked"
    l_transitions[("Seen3_Locked", "2")] = "Seen3_Locked"
    l_transitions[("Seen4_Locked", "0")] = "Seen4_Locked"
    l_transitions[("Seen4_Locked", "1")] = "Seen4_Locked"
    l_transitions[("Seen4_Locked", "2")] = "Seen4_Locked"
    l_transitions[("Sink", "0")] = "Sink"
    l_transitions[("Sink", "1")] = "Sink"
    l_transitions[("Sink", "2")] = "Sink"

    label_dfa = Dfa(alphabet, l_states, l_accepting_states, l_start_state, l_transitions)

    label_map = {}
    label_map["Seen2_Locked"] = "Label_Pos2"
    label_map["Seen3_Locked"] = "Label_Pos3"
    label_map["Seen4_Locked"] = "Label_Pos4"

    label_func = LabellingDfa(label_dfa, label_map)

    # Fix default improviser parameters
    length_bounds = (1,4)
    cost_bound = 1.5
    label_prob_bounds = (Fraction(1,5), Fraction(1,2))

    # Ensure the base LQCI problem is feasible
    improviser = MELQCI(hard_constraint, cost_func, label_func, length_bounds, num_threads=num_threads)
    improviser.parameterize(cost_bound, label_prob_bounds, num_threads=num_threads)

    # Check that various parameter tweaks that render the
    # problem infeasible are identified by the improviser.
    with pytest.raises(InfeasibleImproviserError):
        improviser = MELQCI(hard_constraint, cost_func, label_func, length_bounds)
        improviser.parameterize(0.5, label_prob_bounds)

    with pytest.raises(InfeasibleImproviserError):
        improviser = MELQCI(hard_constraint, cost_func, label_func, length_bounds)
        improviser.parameterize(cost_bound, (Fraction(1,4), Fraction(1,4)))

###################################################################################################
# Random Tests
###################################################################################################

@given( length_bounds=tuples(integers(0,10), integers(0,10)),
        hard_constraint=random_dfa(num_symbols=shared(integers(1,3), key="foo")),
        cost_func=random_static_cost_dfa(num_symbols=shared(integers(1,3), key="foo")),
        label_dfa_bounds=random_label_dfa_bounds(num_symbols=shared(integers(1,3), key="foo")),
        label_prob_bounds=tuples(fractions(0,1), fractions(0,1)),
        cost_bound=fractions(0,100),
        num_threads=integers(1,2), lazy=booleans())
@settings(deadline=None, max_examples=10000)
@pytest.mark.advanced
def test_lqci_fuzz(length_bounds, hard_constraint, cost_func, label_dfa_bounds, label_prob_bounds,
                   cost_bound, num_threads, lazy):
    # Generate completely random LQCI instance.
    label_func, word_prob_bounds = label_dfa_bounds
    label_prob_bounds = tuple(sorted(label_prob_bounds))
    length_bounds = tuple(sorted(length_bounds))

    # Attempt to create an LQCI improviser, catching any InfeasibleImproviserErrors
    try:
        improviser = LQCI(hard_constraint, cost_func, label_func, length_bounds, num_threads=num_threads, lazy=lazy)
        improviser.parameterize(cost_bound, label_prob_bounds, word_prob_bounds, num_threads=num_threads)
    except InfeasibleImproviserError:
        lqci_feasible = False
    else:
        lqci_feasible = True

    # Attempt to create a MELQCI improviser, catching any InfeasibleImproverErrors
    # and suboptimality warnings.
    try:
        improviser = MELQCI(hard_constraint, cost_func, label_func, length_bounds, num_threads=num_threads)
        improviser.parameterize(cost_bound, label_prob_bounds, num_threads=num_threads)
    except InfeasibleImproviserError:
        melqci_feasible = False
    except UserWarning:
        melqci_feasible = None
    except SolverError:
        melqci_feasible = None
    else:
        melqci_feasible = True

    if lqci_feasible:
        assert melqci_feasible is None or melqci_feasible

