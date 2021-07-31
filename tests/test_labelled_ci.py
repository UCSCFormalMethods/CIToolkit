""" Tests for the LabelledCI class"""

import pytest

from citoolkit.improvisers.improviser import InfeasibleImproviserError
from citoolkit.improvisers.labelled_ci import LabelledCI
from citoolkit.specifications.dfa import Dfa
from citoolkit.labellingfunctions.labelling_dfa import LabellingDfa

###################################################################################################
# Basic Tests
###################################################################################################

def test_labelled_ci_improvise():
    """ Test a simple Labelled CI instance. """
    # Create a hard constraint Dfa that accepts all words that end with "01"
    alphabet = {"0", "1"}
    h_states = {"Start", "Seen0", "Accept"}
    h_accepting_states = {"Accept"}
    h_start_state = "Start"

    h_transitions = {}
    h_transitions[("Start", "0")] = "Seen0"
    h_transitions[("Start", "1")] = "Start"
    h_transitions[("Seen0", "0")] = "Seen0"
    h_transitions[("Seen0", "1")] = "Accept"
    h_transitions[("Accept", "0")] = "Seen0"
    h_transitions[("Accept", "1")] = "Start"

    hard_constraint = Dfa(alphabet, h_states, h_accepting_states, h_start_state, h_transitions)

    # Create a soft constraint that accepts all words of length 4
    soft_constraint = Dfa.exact_length_dfa(alphabet, 4)

    # Create a labelling function that labels a word depending on how many "1"
    # symbols it has
    lf_states = {"Seen0", "Seen1", "Seen2", "Seen3", "Sink"}
    lf_accepting_states = {"Seen1", "Seen2", "Seen3"}
    lf_start_state = "Seen0"

    lf_transitions = {}
    lf_transitions[("Seen0", "0")] = "Seen0"
    lf_transitions[("Seen0", "1")] = "Seen1"
    lf_transitions[("Seen1", "0")] = "Seen1"
    lf_transitions[("Seen1", "1")] = "Seen2"
    lf_transitions[("Seen2", "0")] = "Seen2"
    lf_transitions[("Seen2", "1")] = "Seen3"
    lf_transitions[("Seen3", "0")] = "Seen3"
    lf_transitions[("Seen3", "1")] = "Sink"
    lf_transitions[("Sink", "0")] = "Sink"
    lf_transitions[("Sink", "1")] = "Sink"

    lf_dfa = Dfa(alphabet, lf_states, lf_accepting_states, lf_start_state, lf_transitions)

    label_map = {}
    label_map["Seen1"] = "Label1"
    label_map["Seen2"] = "Label2"
    label_map["Seen3"] = "Label3"

    label_func = LabellingDfa(lf_dfa, label_map)

    # Layout remaining Labelled CI parameters.
    length_bounds = (1,4)
    epsilon = 0.2
    label_prob_bounds = (.25, .4)

    word_prob_bounds = {}
    word_prob_bounds["Label1"] = (.2,.8)
    word_prob_bounds["Label2"] = (.2,.8)
    word_prob_bounds["Label3"] = (0,1)

    # Create Labelled CI Improviser
    improviser = LabelledCI(hard_constraint, soft_constraint, label_func, length_bounds, epsilon, label_prob_bounds, word_prob_bounds)

    # Check that the calculated probabilities are valid
    assert sum(improviser.sorted_labels_weights) == pytest.approx(1)

    for label_weight in improviser.sorted_labels_weights:
        assert label_prob_bounds[0] <= label_weight <= label_prob_bounds[1]

    for label in label_func.labels:
        if improviser.i_specs[label].language_size(*length_bounds) == 0:
            assert improviser.i_probs[label] == 0
        else:
            assert word_prob_bounds[label][0] <= improviser.i_probs[label] / improviser.i_specs[label].language_size(*length_bounds) <= word_prob_bounds[label][1]

        if improviser.a_specs[label].language_size(*length_bounds) == 0:
            assert improviser.a_probs[label] == 0
        else:
            assert word_prob_bounds[label][0] <= improviser.a_probs[label] / improviser.a_specs[label].language_size(*length_bounds) <= word_prob_bounds[label][1]

        assert improviser.i_probs[label] + improviser.a_probs[label] == pytest.approx(1)

    # Create sampling testing variables
    improvisations = {tuple("01"), tuple("001"), tuple("101"), tuple("0001"), tuple("0101"), tuple("1001"), tuple("1101")}
    improvisation_count = {improvisation:0 for improvisation in improvisations}

    soft_constraint_improvisations = {tuple("0001"), tuple("0101"), tuple("1001"), tuple("1101")}
    soft_constraint_count = 0

    label_count = {label:0 for label in ["Label1", "Label2", "Label3"]}

    label_improvisation_map = {}
    label_improvisation_map["Label1"] = {tuple("01"), tuple("001"), tuple("0001")}
    label_improvisation_map["Label2"] = {tuple("101"), tuple("0101"), tuple("1001")}
    label_improvisation_map["Label3"] = {tuple("1101")}

    improvisation_label_map = {}
    for label, items in label_improvisation_map.items():
        for item in items:
            improvisation_label_map[item] = label

    # Sample a collection of words from the improviser.
    for _ in range(100000):
        word = improviser.improvise()

        assert word in improvisations

        improvisation_count[word] += 1
        label_count[improvisation_label_map[word]] += 1

        if word in soft_constraint_improvisations:
            soft_constraint_count+=1

    # Check that sampled word probabilities are valid
    for label in label_func.labels:
        label_sampled_prob = label_count[label]/100000
        assert label_prob_bounds[0]-.01 <= label_sampled_prob <= label_prob_bounds[1]+.01

        for word in label_improvisation_map[label]:
            cond_word_sampled_prob = (improvisation_count[word]/100000)/label_sampled_prob
            assert word_prob_bounds[label][0]-0.1 <= cond_word_sampled_prob <= word_prob_bounds[label][1]+0.1

    assert soft_constraint_count/100000 >= .99 - epsilon

def test_labelled_ci_infeasible():
    """ Test that different infeasible Labelled CI
    problems correctly raise an exception.
    """
    # Create a hard constraint Dfa that accepts all words that end with "01"
    alphabet = {"0", "1"}
    h_states = {"Start", "Seen0", "Accept"}
    h_accepting_states = {"Accept"}
    h_start_state = "Start"

    h_transitions = {}
    h_transitions[("Start", "0")] = "Seen0"
    h_transitions[("Start", "1")] = "Start"
    h_transitions[("Seen0", "0")] = "Seen0"
    h_transitions[("Seen0", "1")] = "Accept"
    h_transitions[("Accept", "0")] = "Seen0"
    h_transitions[("Accept", "1")] = "Start"

    hard_constraint = Dfa(alphabet, h_states, h_accepting_states, h_start_state, h_transitions)

    # Create a soft constraint that accepts all words of length 4
    soft_constraint = Dfa.exact_length_dfa(alphabet, 4)

    # Create a labelling function that labels a word depending on how many "1"
    # symbols it has
    lf_states = {"Seen0", "Seen1", "Seen2", "Seen3", "Sink"}
    lf_accepting_states = {"Seen1", "Seen2", "Seen3"}
    lf_start_state = "Seen0"

    lf_transitions = {}
    lf_transitions[("Seen0", "0")] = "Seen0"
    lf_transitions[("Seen0", "1")] = "Seen1"
    lf_transitions[("Seen1", "0")] = "Seen1"
    lf_transitions[("Seen1", "1")] = "Seen2"
    lf_transitions[("Seen2", "0")] = "Seen2"
    lf_transitions[("Seen2", "1")] = "Seen3"
    lf_transitions[("Seen3", "0")] = "Seen3"
    lf_transitions[("Seen3", "1")] = "Sink"
    lf_transitions[("Sink", "0")] = "Sink"
    lf_transitions[("Sink", "1")] = "Sink"

    lf_dfa = Dfa(alphabet, lf_states, lf_accepting_states, lf_start_state, lf_transitions)

    label_map = {}
    label_map["Seen1"] = "Label1"
    label_map["Seen2"] = "Label2"
    label_map["Seen3"] = "Label3"

    label_func = LabellingDfa(lf_dfa, label_map)

    # Layout remaining Labelled CI parameters.
    length_bounds = (1,4)
    epsilon = 0.2
    label_prob_bounds = (.25, .4)

    word_prob_bounds = {}
    word_prob_bounds["Label1"] = (.2,.8)
    word_prob_bounds["Label2"] = (.2,.8)
    word_prob_bounds["Label3"] = (0,1)

    # Ensure that the base LCI problem is feasible
    LabelledCI(hard_constraint, soft_constraint, label_func, length_bounds, epsilon, label_prob_bounds, word_prob_bounds)

    # Check that various parameter tweaks that render the
    # problem infeasible are identified by the improviser.

    with pytest.raises(InfeasibleImproviserError):
        LabelledCI(hard_constraint, soft_constraint, label_func, length_bounds, 0.16, label_prob_bounds, word_prob_bounds)

    with pytest.raises(InfeasibleImproviserError):
        LabelledCI(hard_constraint, soft_constraint, label_func, length_bounds, epsilon, (0.33,0.33), word_prob_bounds)

    with pytest.raises(InfeasibleImproviserError):
        LabelledCI(hard_constraint, soft_constraint, label_func, length_bounds, epsilon, label_prob_bounds, {label:(.2,.8) for label in label_func.labels})
