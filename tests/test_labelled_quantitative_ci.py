""" Tests for the LabelledQuantitativeCI class"""

import random
from fractions import Fraction

import pytest

from citoolkit.improvisers.improviser import InfeasibleImproviserError
from citoolkit.improvisers.labelled_quantitative_ci import LabelledQuantitativeCI
from citoolkit.specifications.dfa import Dfa
from citoolkit.costfunctions.static_cost_dfa import StaticCostDfa
from citoolkit.labellingfunctions.labelling_dfa import LabellingDfa

from .test_dfa import generate_random_dfa

###################################################################################################
# Basic Tests
###################################################################################################

def test_labelled_quantitative_ci_improvise():
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

    improviser = LabelledQuantitativeCI(hard_constraint, cost_func, label_func, length_bounds, cost_bound, label_prob_bounds, word_prob_bounds)

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

        if word not in improvisation_count.keys():
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

    assert cost_accumulator/sum(label_count.values()) < cost_bound

def test_labelled_quantitative_ci_infeasible():
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

    # Fix remaining improviser parameters
    length_bounds = (1,4)
    cost_bound = 0.5
    label_prob_bounds = (Fraction(1,5), Fraction(1,2))
    word_prob_bounds = {"Label_Pos2":(Fraction(1,13), Fraction(1,13)), "Label_Pos3":(Fraction(1,20),Fraction(1,4)), "Label_Pos4":(Fraction(1,6),Fraction(5,12))}

    # Ensure the base LQCI problem is feasible
    LabelledQuantitativeCI(hard_constraint, cost_func, label_func, length_bounds, cost_bound, label_prob_bounds, word_prob_bounds)

    # Check that various parameter tweaks that render the
    # problem infeasible are identified by the improviser.
    with pytest.raises(InfeasibleImproviserError):
        LabelledQuantitativeCI(hard_constraint, cost_func, label_func, length_bounds, 0.3, label_prob_bounds, word_prob_bounds)

    with pytest.raises(InfeasibleImproviserError):
        LabelledQuantitativeCI(hard_constraint, cost_func, label_func, length_bounds, cost_bound, (Fraction(1,3), Fraction(1,3)), word_prob_bounds)

    with pytest.raises(InfeasibleImproviserError):
        LabelledQuantitativeCI(hard_constraint, cost_func, label_func, length_bounds, cost_bound, (Fraction(1,4), Fraction(1,4)), word_prob_bounds)

    with pytest.raises(InfeasibleImproviserError):
        strict_word_prob_bounds = {"Label_Pos2":(Fraction(1,13), Fraction(1,13)), "Label_Pos3":(Fraction(1,20),Fraction(1,4)), "Label_Pos4":(Fraction(1,4),Fraction(1,4))}
        LabelledQuantitativeCI(hard_constraint, cost_func, label_func, length_bounds, cost_bound, label_prob_bounds, strict_word_prob_bounds)

    with pytest.raises(InfeasibleImproviserError):
        infeasible_word_prob_bounds = {"Label_Pos2":(Fraction(1,12), Fraction(1,12)), "Label_Pos3":(Fraction(1,20),Fraction(1,4)), "Label_Pos4":(Fraction(1,6),Fraction(5,12))}
        LabelledQuantitativeCI(hard_constraint, cost_func, label_func, length_bounds, cost_bound, label_prob_bounds, infeasible_word_prob_bounds)

###################################################################################################
# Random Tests
###################################################################################################
