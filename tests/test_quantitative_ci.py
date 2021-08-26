""" Tests for the QuantitativeCI class"""

import random
from fractions import Fraction

import pytest

from citoolkit.improvisers.improviser import InfeasibleImproviserError
from citoolkit.improvisers.quantitative_ci import QuantitativeCI
from citoolkit.specifications.dfa import Dfa
from citoolkit.costfunctions.static_cost_dfa import StaticCostDfa

from .test_dfa import generate_random_dfa

###################################################################################################
# Basic Tests
###################################################################################################

def test_quantitative_ci_improvise():
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
    min_cost = 1
    epsilon = 2

    # Create Quantitative CI Improviser.
    improviser = QuantitativeCI(hard_constraint, cost_function, length_bounds, epsilon, prob_bounds)

    # Check that the calculated probabilities and expected cost of the improviser are valid.
    assert sum(improviser.sorted_costs_weights) == pytest.approx(1)
    assert improviser.expected_cost <= min_cost * epsilon

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

    assert accumulated_cost/100000 <= min_cost * epsilon

def test_quantitative_ci_infeasible():
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
    epsilon = 2

    # Ensure that the base QCI problem is feasible
    QuantitativeCI(hard_constraint, cost_function, length_bounds, epsilon, prob_bounds)

    # Check that various parameter tweaks that render the
    # problem infeasible are identified by the improviser.
    with pytest.raises(InfeasibleImproviserError):
        QuantitativeCI(hard_constraint, cost_function, length_bounds, 1, prob_bounds)

    with pytest.raises(InfeasibleImproviserError):
        QuantitativeCI(hard_constraint, cost_function, length_bounds, epsilon, [Fraction(1,15), Fraction(1,15)])

###################################################################################################
# Randomized Tests
###################################################################################################

# Randomized tests default parameters
_RANDOM_QCI_TEST_NUM_ITERS = 1000
_RANDOM_QCI_TEST_NUM_SAMPLES = 25000
_MIN_WORD_LENGTH_BOUND = 0
_MAX_WORD_LENGTH_BOUND = 10
_MAX_COST = 100

@pytest.mark.slow
def test_quantitative_ci_improvise_random():
    """ Tests generating a mix of fully random and random
    but feasible Quantitative CI improviser instances and
    ensuring that they either are infeasible or are
    feasible and improvise correctly.
    """
    for _ in range(_RANDOM_QCI_TEST_NUM_ITERS):
        # Generate random set of QuantitativeCI parameters. 50% chance to
        # generate an instance that is guaranteed satisfiable.
        if random.random() < 0.5:
            # Generate completely random QCI instance.
            guaranteed_feasible = False

            min_length = random.randint(_MIN_WORD_LENGTH_BOUND, _MAX_WORD_LENGTH_BOUND)
            max_length = random.randint(min_length, _MAX_WORD_LENGTH_BOUND)
            length_bounds = (min_length, max_length)

            hard_constraint = generate_random_dfa(max_states=8)

            cf_dfa = generate_random_dfa(max_states=8, alphabet=hard_constraint.alphabet)

            cost_set = [Fraction(random.randint(1, _MAX_COST), random.randint(1, _MAX_COST)) for cost_iter in range(len(cf_dfa.accepting_states))]
            cost_map = {}

            for accepting_state in cf_dfa.accepting_states:
                cost_map[accepting_state] = random.choice(cost_set)

            cost_func = StaticCostDfa(cf_dfa, cost_map)

            epsilon = random.uniform(1,_MAX_COST)

            min_prob = random.uniform(0,1)
            max_prob = random.uniform(min_prob, 1)
            prob_bounds = (min_prob, max_prob)
        else:
            # Generate a somewhat random QCI instance that is feasible.
            guaranteed_feasible = True

            min_length = random.randint(_MIN_WORD_LENGTH_BOUND, _MAX_WORD_LENGTH_BOUND)
            max_length = random.randint(min_length, _MAX_WORD_LENGTH_BOUND)
            length_bounds = (min_length, max_length)

            hard_constraint = generate_random_dfa(max_states=8)
            cf_dfa = generate_random_dfa(max_states=8, alphabet=hard_constraint.alphabet)

            while (hard_constraint & cf_dfa).language_size(*length_bounds) == 0:
                hard_constraint = generate_random_dfa()
                cf_dfa = generate_random_dfa(max_states=8, alphabet=hard_constraint.alphabet)

            cost_set = [Fraction(random.randint(1, _MAX_COST), random.randint(1, _MAX_COST)) for cost_iter in range(len(cf_dfa.accepting_states))]
            cost_map = {}

            for accepting_state in cf_dfa.accepting_states:
                cost_map[accepting_state] = random.choice(cost_set)

            cost_func = StaticCostDfa(cf_dfa, cost_map)

            cost_specs = cost_func.decompose()

            total_cost = sum([cost * (hard_constraint & cost_specs[cost]).language_size(*length_bounds) for cost in cost_func.costs])
            total_words = sum([(hard_constraint & cost_specs[cost]).language_size(*length_bounds) for cost in cost_func.costs])

            epsilon = Fraction(1.1) * Fraction(total_cost, total_words)/min(cost_func.costs)

            min_prob = Fraction(1, total_words * random.randint(1, 10))
            max_prob = min(1, Fraction(random.randint(1, 10) , total_words))
            prob_bounds = (min_prob, max_prob)


        # Attempt to create the improviser. If it is a feasible problem,
        # attempt to sample it and ensure that the output distribution
        # is relatively correct.
        try:
            improviser = QuantitativeCI(hard_constraint, cost_func, length_bounds, epsilon, prob_bounds)
        except InfeasibleImproviserError:
            assert not guaranteed_feasible
            continue

        improvisation_count = {}
        accumulated_cost = 0

        # Sample a collection of words from the improviser.
        for _ in range(_RANDOM_QCI_TEST_NUM_SAMPLES):
            word = improviser.improvise()

            if word not in improvisation_count.keys():
                improvisation_count[word] = 1
            else:
                improvisation_count[word] += 1

            accumulated_cost += cost_func.cost(word)

        # Check that sampled word probabilities and average cost are valid
        for word in improvisation_count:
            assert hard_constraint.accepts(word)

            assert prob_bounds[0]-.01 <= improvisation_count[word]/_RANDOM_QCI_TEST_NUM_SAMPLES <= prob_bounds[1]+.01

        assert accumulated_cost/_RANDOM_QCI_TEST_NUM_SAMPLES <= improviser.min_cost * epsilon
