""" Tests for the QCI class"""

from fractions import Fraction

import pytest

from citoolkit.improvisers.improviser import InfeasibleCostError, InfeasibleRandomnessError
from citoolkit.improvisers.qci import QCI
from citoolkit.specifications.dfa import Dfa
from citoolkit.costfunctions.static_cost_dfa import StaticCostDfa

###################################################################################################
# Basic Tests
###################################################################################################

def test_qci_improvise():
    """ Test a simple Quantitative CI instance. """
    # Create a hard constraint Dfa that accepts all words start with "0" and end with "0"
    alphabet = {"0", "1"}
    h_states = {"Start", "Middle", "End", "Sink"}
    h_accepting_states = {"End"}
    h_start_state = "Start"

    h_transitions = {}
    h_transitions[("Start", "0")] = "Middle"
    h_transitions[("Start", "1")] = "Sink"
    h_transitions[("Middle", "0")] = "End"
    h_transitions[("Middle", "1")] = "Middle"
    h_transitions[("End", "0")] = "End"
    h_transitions[("End", "1")] = "Middle"
    h_transitions[("Sink", "0")] = "Sink"
    h_transitions[("Sink", "1")] = "Sink"

    hard_constraint = Dfa(alphabet, h_states, h_accepting_states, h_start_state, h_transitions)

    # Create a cost function assigns an integer cost equal to the number of 1 symbols seen.
    k_states = {"Seen0", "Seen1", "Seen2", "Seen3", "Seen4+"}
    k_accepting_states = {"Seen0", "Seen1", "Seen2", "Seen3"}
    k_start_state = "Seen0"

    k_transitions = {}
    k_transitions[("Seen0", "0")] = "Seen0"
    k_transitions[("Seen0", "1")] = "Seen1"
    k_transitions[("Seen1", "0")] = "Seen1"
    k_transitions[("Seen1", "1")] = "Seen2"
    k_transitions[("Seen2", "0")] = "Seen2"
    k_transitions[("Seen2", "1")] = "Seen3"
    k_transitions[("Seen3", "0")] = "Seen3"
    k_transitions[("Seen3", "1")] = "Seen4+"
    k_transitions[("Seen4+", "0")] = "Seen4+"
    k_transitions[("Seen4+", "1")] = "Seen4+"

    cost_dfa = Dfa(alphabet, k_states, k_accepting_states, k_start_state, k_transitions)

    cost_map = {}
    cost_map["Seen0"] = 1
    cost_map["Seen1"] = 2
    cost_map["Seen2"] = 3
    cost_map["Seen3"] = 4

    cost_function = StaticCostDfa(cost_dfa, cost_map)

    # Layout remaining Quantitative CI parameters.
    length_bounds = (1,5)
    prob_bounds = [Fraction(1,50), Fraction(1,11)]
    cost_bound = 2

    # Create Quantitative CI Improviser.
    improviser = QCI(hard_constraint, cost_function, length_bounds, cost_bound, prob_bounds)

    # Sample the improviser and check that all improvisations are valid and that the probabilities and cost are reasonable.
    improvisations = {tuple("00"), tuple("000"), tuple("010"), tuple("0000"), tuple("0010"), tuple("0100"), tuple("0110"), \
                      tuple("00000"), tuple("00010"), tuple("00100"), tuple("00110"), tuple("01000"), tuple("01010"), tuple("01100"), tuple("01110")}
    improvisation_count = {improvisation:0 for improvisation in improvisations}

    accumulated_cost = 0

    # Sample a collection of words from the improviser.
    for _ in range(100000):
        word = improviser.improvise()

        assert word in improvisations

        improvisation_count[word] += 1
        accumulated_cost += cost_function.cost(word)

    # Check that sampled word probabilities and average cost are valid
    for word in improvisations:
        assert prob_bounds[0]-.01 <= improvisation_count[word]/100000 <= prob_bounds[1]+.01

    assert accumulated_cost/100000 <= cost_bound

def test_qci_infeasible():
    """ Test that different infeasible Quantitative CI
    problems correctly raise an exception.
    """
    # Create a hard constraint Dfa that accepts all words start with "0" and end with "0"
    alphabet = {"0", "1"}
    h_states = {"Start", "Middle", "End", "Sink"}
    h_accepting_states = {"End"}
    h_start_state = "Start"

    h_transitions = {}
    h_transitions[("Start", "0")] = "Middle"
    h_transitions[("Start", "1")] = "Sink"
    h_transitions[("Middle", "0")] = "End"
    h_transitions[("Middle", "1")] = "Middle"
    h_transitions[("End", "0")] = "End"
    h_transitions[("End", "1")] = "Middle"
    h_transitions[("Sink", "0")] = "Sink"
    h_transitions[("Sink", "1")] = "Sink"

    hard_constraint = Dfa(alphabet, h_states, h_accepting_states, h_start_state, h_transitions)

    # Create a cost function assigns an integer cost equal to the number of 1 symbols seen.
    k_states = {"Seen0", "Seen1", "Seen2", "Seen3", "Seen4+"}
    k_accepting_states = {"Seen0", "Seen1", "Seen2", "Seen3"}
    k_start_state = "Seen0"

    k_transitions = {}
    k_transitions[("Seen0", "0")] = "Seen0"
    k_transitions[("Seen0", "1")] = "Seen1"
    k_transitions[("Seen1", "0")] = "Seen1"
    k_transitions[("Seen1", "1")] = "Seen2"
    k_transitions[("Seen2", "0")] = "Seen2"
    k_transitions[("Seen2", "1")] = "Seen3"
    k_transitions[("Seen3", "0")] = "Seen3"
    k_transitions[("Seen3", "1")] = "Seen4+"
    k_transitions[("Seen4+", "0")] = "Seen4+"
    k_transitions[("Seen4+", "1")] = "Seen4+"

    cost_dfa = Dfa(alphabet, k_states, k_accepting_states, k_start_state, k_transitions)

    cost_map = {}
    cost_map["Seen0"] = 1
    cost_map["Seen1"] = 2
    cost_map["Seen2"] = 3
    cost_map["Seen3"] = 4

    cost_function = StaticCostDfa(cost_dfa, cost_map)

    # Layout remaining Quantitative CI parameters.
    length_bounds = (1,5)
    prob_bounds = [Fraction(1,50), Fraction(1,11)]
    cost_bound = 2

    # Ensure that the base QCI problem is feasible
    QCI(hard_constraint, cost_function, length_bounds, cost_bound, prob_bounds)

    # Check that various parameter tweaks that render the
    # problem infeasible are identified by the improviser.
    with pytest.raises(InfeasibleCostError):
        QCI(hard_constraint, cost_function, length_bounds, 1, prob_bounds)

    with pytest.raises(InfeasibleRandomnessError):
        QCI(hard_constraint, cost_function, length_bounds, cost_bound, [Fraction(1,10), Fraction(1,10)])
