""" Tests for the LabelledQuantitativeCI class"""

import random
from fractions import Fraction

import pytest

from citoolkit.improvisers.improviser import InfeasibleImproviserError, InfeasibleCostError,\
    InfeasibleLabelRandomnessError, InfeasibleWordRandomnessError
from citoolkit.improvisers.labelled_quantitative_ci import LabelledQuantitativeCI, MaxEntropyLabelledQuantitativeCI
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

    assert cost_accumulator/100000 < cost_bound*1.01

def test_labelled_quantitative_ci_improvise_2():
    """ Test a simple Labelled Quantitative CI instance, but with direct specs and multiple threads """
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

    # Create direct specs.
    direct_specs = dict()

    label_map = label_func.decompose()
    cost_map = cost_func.decompose()

    for label in label_func.labels:
        for cost in cost_func.costs:
            direct_specs[(label, cost)] = (hard_constraint & label_map[label] & cost_map[cost]).explicit()

    # Fix remaining improviser parameters
    length_bounds = (1,4)
    cost_bound = 0.5
    label_prob_bounds = (Fraction(1,5), Fraction(1,2))
    word_prob_bounds = {"Label_Pos2":(Fraction(1,13), Fraction(1,13)), "Label_Pos3":(Fraction(1,20),Fraction(1,4)), "Label_Pos4":(Fraction(1,6),Fraction(5,12))}

    improviser = LabelledQuantitativeCI(hard_constraint, cost_func, label_func, length_bounds, cost_bound, label_prob_bounds, word_prob_bounds,\
        direct_specs=direct_specs, num_threads=3)

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

    assert cost_accumulator/100000 < cost_bound*1.01

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

    # Fix default improviser parameters
    length_bounds = (1,4)
    cost_bound = 0.5
    label_prob_bounds = (Fraction(1,5), Fraction(1,2))
    word_prob_bounds = {"Label_Pos2":(Fraction(1,13), Fraction(1,13)), "Label_Pos3":(Fraction(1,20),Fraction(1,4)), "Label_Pos4":(Fraction(1,6),Fraction(5,12))}

    # Ensure the base LQCI problem is feasible
    LabelledQuantitativeCI(hard_constraint, cost_func, label_func, length_bounds, cost_bound, label_prob_bounds, word_prob_bounds)

    # Check that various parameter tweaks that render the
    # problem infeasible are identified by the improviser.
    with pytest.raises(InfeasibleCostError):
        LabelledQuantitativeCI(hard_constraint, cost_func, label_func, length_bounds, 0.3, label_prob_bounds, word_prob_bounds)

    with pytest.raises(InfeasibleCostError):
        LabelledQuantitativeCI(hard_constraint, cost_func, label_func, length_bounds, cost_bound, (Fraction(1,3), Fraction(1,3)), word_prob_bounds)

    with pytest.raises(InfeasibleLabelRandomnessError):
        LabelledQuantitativeCI(hard_constraint, cost_func, label_func, length_bounds, cost_bound, (Fraction(1,4), Fraction(1,4)), word_prob_bounds)

    with pytest.raises(InfeasibleCostError):
        strict_word_prob_bounds = {"Label_Pos2":(Fraction(1,13), Fraction(1,13)), "Label_Pos3":(Fraction(1,20),Fraction(1,4)), "Label_Pos4":(Fraction(1,4),Fraction(1,4))}
        LabelledQuantitativeCI(hard_constraint, cost_func, label_func, length_bounds, cost_bound, label_prob_bounds, strict_word_prob_bounds)

    with pytest.raises(InfeasibleWordRandomnessError):
        infeasible_word_prob_bounds = {"Label_Pos2":(Fraction(1,12), Fraction(1,12)), "Label_Pos3":(Fraction(1,20),Fraction(1,4)), "Label_Pos4":(Fraction(1,6),Fraction(5,12))}
        LabelledQuantitativeCI(hard_constraint, cost_func, label_func, length_bounds, cost_bound, label_prob_bounds, infeasible_word_prob_bounds)

def test_max_entropy_labelled_quantitative_ci_improvise():
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

    improviser = MaxEntropyLabelledQuantitativeCI(hard_constraint, cost_func, label_func, length_bounds, cost_bound, label_prob_bounds)

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

    assert cost_accumulator/100000 <= cost_bound*1.01

def test_max_entropy_labelled_quantitative_ci_infeasible():
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
    MaxEntropyLabelledQuantitativeCI(hard_constraint, cost_func, label_func, length_bounds, cost_bound, label_prob_bounds)

    # Check that various parameter tweaks that render the
    # problem infeasible are identified by the improviser.
    with pytest.raises(InfeasibleImproviserError):
        MaxEntropyLabelledQuantitativeCI(hard_constraint, cost_func, label_func, length_bounds, 0.5, label_prob_bounds)

    with pytest.raises(InfeasibleImproviserError):
        MaxEntropyLabelledQuantitativeCI(hard_constraint, cost_func, label_func, length_bounds, cost_bound, ((Fraction(1,4), Fraction(1,4))))

###################################################################################################
# Random Tests
###################################################################################################

# Randomized tests default parameters
_RANDOM_LQCI_TEST_NUM_ITERS = 200
_RANDOM_LQCI_TEST_NUM_SAMPLES = 100000
_RANDOM_MELQCI_TEST_NUM_SAMPLES = 100000
_MIN_WORD_LENGTH_BOUND = 0
_MAX_WORD_LENGTH_BOUND = 10
_MAX_COST = 100

@pytest.mark.slow
def test_labelled_quantitative_ci_improvise_random():
    """ Tests generating a mix of fully random and random
    but likely feasible Labelled Quantitative CI improviser
    instances and ensuring that they either are infeasible
    or are feasible and improvise correctly.
    """
    for _ in range(_RANDOM_LQCI_TEST_NUM_ITERS):
        # Generate random set of LabelledCI parameters. 50% chance to
        # generate an instance where each parameter is individually feasible.
        if random.random() < 0.5:
            # Generate completely random LCI instance.
            min_length = random.randint(_MIN_WORD_LENGTH_BOUND, _MAX_WORD_LENGTH_BOUND)
            max_length = random.randint(min_length, _MAX_WORD_LENGTH_BOUND)
            length_bounds = (min_length, max_length)

            hard_constraint = generate_random_dfa(max_states=6)

            lf_dfa = generate_random_dfa(max_states=6, alphabet=hard_constraint.alphabet)

            label_set = ["Label" + str(label_iter) for label_iter in range(len(lf_dfa.accepting_states))]
            label_map = {}

            for accepting_state in lf_dfa.accepting_states:
                label_map[accepting_state] = random.choice(label_set)

            label_func = LabellingDfa(lf_dfa, label_map)

            cf_dfa = generate_random_dfa(max_states=6, alphabet=hard_constraint.alphabet)

            cost_set = [Fraction(random.randint(0, _MAX_COST), random.randint(1, _MAX_COST)) for cost_iter in range(len(cf_dfa.accepting_states))]
            cost_map = {}

            for accepting_state in cf_dfa.accepting_states:
                cost_map[accepting_state] = random.choice(cost_set)

            cost_func = StaticCostDfa(cf_dfa, cost_map)

            cost_bound = random.uniform(0,_MAX_COST)

            label_min_prob = random.uniform(0,1)
            label_max_prob = random.uniform(label_min_prob, 1)
            label_prob_bounds = (label_min_prob, label_max_prob)

            word_min_prob = {label:random.uniform(0,1) for label in label_func.labels}
            word_max_prob = {label:random.uniform(word_min_prob[label], 1) for label in label_func.labels}
            word_prob_bounds = {label:(word_min_prob[label], word_max_prob[label]) for label in label_func.labels}
        else:
            # Generate a random LCI instance while making each of the parameters individually feasible.
            min_length = random.randint(_MIN_WORD_LENGTH_BOUND, _MAX_WORD_LENGTH_BOUND)
            max_length = random.randint(min_length, _MAX_WORD_LENGTH_BOUND)
            length_bounds = (min_length, max_length)

            hard_constraint = generate_random_dfa(max_states=6)
            lf_dfa = generate_random_dfa(max_states=6, alphabet=hard_constraint.alphabet)

            label_set = ["Label" + str(label_iter) for label_iter in range(len(lf_dfa.accepting_states))]
            label_map = {}

            for accepting_state in lf_dfa.accepting_states:
                label_map[accepting_state] = random.choice(label_set)

            label_func = LabellingDfa(lf_dfa, label_map)

            cf_dfa = generate_random_dfa(max_states=6, alphabet=hard_constraint.alphabet)

            while True:
                empty_label_class = False

                if (hard_constraint & lf_dfa & cf_dfa).language_size(*length_bounds) == 0:
                    empty_label_class = True
                else:
                    for label, label_spec in label_func.decompose().items():
                        if (hard_constraint & label_spec & cf_dfa).language_size(*length_bounds) == 0:
                            empty_label_class = True
                            break

                if not empty_label_class:
                    break

                hard_constraint = generate_random_dfa(max_states=6)

                lf_dfa = generate_random_dfa(max_states=6, alphabet=hard_constraint.alphabet)

                label_set = ["Label" + str(label_iter) for label_iter in range(len(lf_dfa.accepting_states))]
                label_map = {}

                for accepting_state in lf_dfa.accepting_states:
                    label_map[accepting_state] = random.choice(label_set)

                label_func = LabellingDfa(lf_dfa, label_map)

                cf_dfa = generate_random_dfa(max_states=6, alphabet=hard_constraint.alphabet)

            cost_set = [Fraction(random.randint(0, _MAX_COST), random.randint(1, _MAX_COST)) for cost_iter in range(len(cf_dfa.accepting_states))]
            cost_map = {}

            for accepting_state in cf_dfa.accepting_states:
                cost_map[accepting_state] = random.choice(cost_set)

            cost_func = StaticCostDfa(cf_dfa, cost_map)

            cost_specs = cost_func.decompose()

            total_cost = sum([cost * (hard_constraint & cost_specs[cost]).language_size(*length_bounds) for cost in cost_func.costs])
            total_words = sum([(hard_constraint & cost_specs[cost]).language_size(*length_bounds) for cost in cost_func.costs])

            cost_bound = Fraction(1.1) * Fraction(total_cost, total_words)

            label_class_sizes = {label:(hard_constraint & label_spec).language_size(*length_bounds) for (label, label_spec) in label_func.decompose().items()}

            label_min_prob = Fraction(1, len(label_func.labels) * random.randint(1, 10))
            label_max_prob = min(1, Fraction(random.randint(1, 10) , len(label_func.labels)))
            label_prob_bounds = (label_min_prob, label_max_prob)

            word_min_prob = {label:Fraction(1, label_class_sizes[label] * random.randint(1, 10)) for label in label_func.labels}
            word_max_prob = {label:min(1, Fraction(random.randint(1, 10) , label_class_sizes[label])) for label in label_func.labels}
            word_prob_bounds = {label:(word_min_prob[label], word_max_prob[label]) for label in label_func.labels}

        # Attempt to create the improviser. If it is a feasible problem,
        # attempt to sample it and ensure that the output distribution
        # is relatively correct.
        try:
            improviser = LabelledQuantitativeCI(hard_constraint, cost_func, label_func, length_bounds, cost_bound, label_prob_bounds, word_prob_bounds)
        except InfeasibleImproviserError:
            continue

        improvisation_count = {}
        accumulated_cost = 0

        # Sample a collection of words from the improviser.
        for _ in range(_RANDOM_LQCI_TEST_NUM_SAMPLES):
            word = improviser.improvise()

            if word not in improvisation_count.keys():
                improvisation_count[word] = 1
            else:
                improvisation_count[word] += 1

            accumulated_cost += cost_func.cost(word)

        # Ensures the sampled distribution is relatively correct within a tolerance.
        for word in improvisation_count:
            assert hard_constraint.accepts(word)

        for label in label_func.labels:
            label_count = 0
            label_spec = label_func.decompose()[label]
            label_words = set()

            for word, count in improvisation_count.items():
                if label_spec.accepts(word):
                    label_count += count
                    label_words.add(word)

            assert label_prob_bounds[0] - 0.02 <= label_count/_RANDOM_LQCI_TEST_NUM_SAMPLES <= label_prob_bounds[1] + 0.02

            for word in label_words:
                count = improvisation_count[word]
                assert word_prob_bounds[label][0] - 0.1 <= count/label_count <= word_prob_bounds[label][1] + 0.1

        assert accumulated_cost/_RANDOM_LQCI_TEST_NUM_SAMPLES <= (cost_bound + .05) * 1.05

@pytest.mark.slow
def test_max_entropy_labelled_quantitative_ci_improvise_random():
    """ Tests generating a mix of fully random and random
    but likely feasible Labelled Quantitative CI improviser
    instances and ensuring that they either are infeasible
    or are feasible and improvise correctly.
    """
    for _ in range(_RANDOM_LQCI_TEST_NUM_ITERS):
        # Generate random set of LabelledQuantitativeCI parameters. 50% chance to
        # generate an instance where each parameter is individually feasible.
        if random.random() < 0.5:
            # Generate completely random LQCI instance.
            min_length = random.randint(_MIN_WORD_LENGTH_BOUND, _MAX_WORD_LENGTH_BOUND)
            max_length = random.randint(min_length, _MAX_WORD_LENGTH_BOUND)
            length_bounds = (min_length, max_length)

            hard_constraint = generate_random_dfa(max_states=6)

            lf_dfa = generate_random_dfa(max_states=6, alphabet=hard_constraint.alphabet)

            label_set = ["Label" + str(label_iter) for label_iter in range(len(lf_dfa.accepting_states))]
            label_map = {}

            for accepting_state in lf_dfa.accepting_states:
                label_map[accepting_state] = random.choice(label_set)

            label_func = LabellingDfa(lf_dfa, label_map)

            cf_dfa = generate_random_dfa(max_states=6, alphabet=hard_constraint.alphabet)

            cost_set = [Fraction(random.randint(0, _MAX_COST), random.randint(1, _MAX_COST)) for cost_iter in range(len(cf_dfa.accepting_states))]
            cost_map = {}

            for accepting_state in cf_dfa.accepting_states:
                cost_map[accepting_state] = random.choice(cost_set)

            cost_func = StaticCostDfa(cf_dfa, cost_map)

            cost_bound = random.uniform(0,_MAX_COST)

            label_min_prob = random.uniform(0,1)
            label_max_prob = random.uniform(label_min_prob, 1)
            label_prob_bounds = (label_min_prob, label_max_prob)

        else:
            # Generate a random LCI instance while making each of the parameters individually feasible.
            min_length = random.randint(_MIN_WORD_LENGTH_BOUND, _MAX_WORD_LENGTH_BOUND)
            max_length = random.randint(min_length, _MAX_WORD_LENGTH_BOUND)
            length_bounds = (min_length, max_length)

            hard_constraint = generate_random_dfa(max_states=6)
            lf_dfa = generate_random_dfa(max_states=6, alphabet=hard_constraint.alphabet)

            label_set = ["Label" + str(label_iter) for label_iter in range(len(lf_dfa.accepting_states))]
            label_map = {}

            for accepting_state in lf_dfa.accepting_states:
                label_map[accepting_state] = random.choice(label_set)

            label_func = LabellingDfa(lf_dfa, label_map)

            cf_dfa = generate_random_dfa(max_states=6, alphabet=hard_constraint.alphabet)

            while True:
                empty_label_class = False

                if (hard_constraint & lf_dfa & cf_dfa).language_size(*length_bounds) == 0:
                    empty_label_class = True
                else:
                    for label, label_spec in label_func.decompose().items():
                        if (hard_constraint & label_spec & cf_dfa).language_size(*length_bounds) == 0:
                            empty_label_class = True
                            break

                if not empty_label_class:
                    break

                hard_constraint = generate_random_dfa(max_states=6)

                lf_dfa = generate_random_dfa(max_states=6, alphabet=hard_constraint.alphabet)

                label_set = ["Label" + str(label_iter) for label_iter in range(len(lf_dfa.accepting_states))]
                label_map = {}

                for accepting_state in lf_dfa.accepting_states:
                    label_map[accepting_state] = random.choice(label_set)

                label_func = LabellingDfa(lf_dfa, label_map)

                cf_dfa = generate_random_dfa(max_states=6, alphabet=hard_constraint.alphabet)

            cost_set = [Fraction(random.randint(0, _MAX_COST), random.randint(1, _MAX_COST)) for cost_iter in range(len(cf_dfa.accepting_states))]
            cost_map = {}

            for accepting_state in cf_dfa.accepting_states:
                cost_map[accepting_state] = random.choice(cost_set)

            cost_func = StaticCostDfa(cf_dfa, cost_map)

            cost_specs = cost_func.decompose()

            total_cost = sum([cost * (hard_constraint & cost_specs[cost]).language_size(*length_bounds) for cost in cost_func.costs])
            total_words = sum([(hard_constraint & cost_specs[cost]).language_size(*length_bounds) for cost in cost_func.costs])

            cost_bound = Fraction(1.1) * Fraction(total_cost, total_words)

            label_min_prob = Fraction(1, len(label_func.labels) * random.randint(1, 10))
            label_max_prob = min(1, Fraction(random.randint(1, 10) , len(label_func.labels)))
            label_prob_bounds = (label_min_prob, label_max_prob)

        # Attempt to create the MELCI improviser. Then check that
        # the associated LCI instance is also feasible/infeasible.
        # If it is a feasible problem, attempt to sample it and
        # ensure that the output distribution is relatively correct.
        try:
            improviser = MaxEntropyLabelledQuantitativeCI(hard_constraint, cost_func, label_func, length_bounds, cost_bound, label_prob_bounds)
        except InfeasibleImproviserError:
            melqci_feasible = False
        except UserWarning:
            continue
        else:
            melqci_feasible = True

        try:
            trivial_word_bounds = {label:(0,1) for label in label_func.labels}
            LabelledQuantitativeCI(hard_constraint, cost_func, label_func, length_bounds, cost_bound, label_prob_bounds, trivial_word_bounds)
        except InfeasibleImproviserError:
            qlci_feasible = False
        else:
            qlci_feasible = True

        assert melqci_feasible == qlci_feasible

        if not melqci_feasible:
            continue

        improvisation_count = {}
        accumulated_cost = 0

        # Sample a collection of words from the improviser.
        for _ in range(_RANDOM_MELQCI_TEST_NUM_SAMPLES):
            word = improviser.improvise()

            if word not in improvisation_count.keys():
                improvisation_count[word] = 1
            else:
                improvisation_count[word] += 1

            accumulated_cost += cost_func.cost(word)

        # Ensures the sampled distribution is relatively correct within a tolerance.
        for word in improvisation_count:
            assert hard_constraint.accepts(word)

        for label in label_func.labels:
            label_count = 0
            label_spec = label_func.decompose()[label]
            label_words = set()

            for word, count in improvisation_count.items():
                if label_spec.accepts(word):
                    label_count += count
                    label_words.add(word)

            assert label_prob_bounds[0] - 0.02 <= label_count/_RANDOM_MELQCI_TEST_NUM_SAMPLES <= label_prob_bounds[1] + 0.02

        assert accumulated_cost/_RANDOM_MELQCI_TEST_NUM_SAMPLES <= (cost_bound + .05) * 1.05
