""" Tests for the AccumulatedCostDfa class"""

from fractions import Fraction
import itertools
import random

import pytest

from citoolkit.specifications.dfa import Dfa, State
from citoolkit.costfunctions.accumulated_cost_dfa import AccumulatedCostDfa

from .test_dfa import generate_random_dfa


###################################################################################################
# Basic Tests
###################################################################################################

def test_accumulated_cost_dfa_complete():
    """ Creates a simple AccumulatedCostDfa and ensures that costs
    are calculated correctly.
    """
    # Create a simple DFA and AccumulatedCostDfa
    alphabet = {"0", "1"}
    states = {State("State1"), "State2", State("State3"), "State4", "Sink"}
    accepting_states = {"State4"}
    start_state = "State1"

    transitions = {}

    transitions[("State1", "0")] = State("State2")
    transitions[("State1", "1")] = "State3"

    transitions[("State2", "0")] = "State1"
    transitions[("State2", "1")] = "State4"

    transitions[("State3", "0")] = "Sink"
    transitions[("State3", "1")] = "State2"

    transitions[(State("State4"), "0")] = "State4"
    transitions[("State4", "1")] = "State4"

    transitions[("Sink", "0")] = State("Sink")
    transitions[("Sink", "1")] = "Sink"

    dfa = Dfa(alphabet, states, accepting_states, start_state, transitions)

    cost_map = {}
    cost_map[State("State1")] = 1
    cost_map["State2"] = 2
    cost_map["State3"] = 3
    cost_map[State("State4")] = 4
    cost_map["Sink"] = 999999

    accumulated_cost_dfa = AccumulatedCostDfa(dfa, cost_map, max_word_length=6)

    # Check that select words have the correct cost.
    assert accumulated_cost_dfa.cost(tuple("0001")) == 10
    assert accumulated_cost_dfa.cost(tuple("0000")) is None
    assert accumulated_cost_dfa.cost(tuple("110111")) == 16
    assert accumulated_cost_dfa.cost(tuple("11001")) == 13

def test_static_cost_dfa_incomplete():
    """ Creates a simple incomplete AccumulatedCostDfa and ensures
    this raises an error.
    """

    # Create a DFA that accepts all strings with at least one "1"
    # symbol, but tracks up to 4 seen.
    alphabet = {"0", "1", "2"}
    states = {"0_Seen", "1_Seen", "2_Seen", "3_Seen", "4+_Seen"}
    accepting_states = {"1_Seen", "2_Seen", "3_Seen", "4+_Seen"}
    start_state = "0_Seen"

    transitions = {}

    transitions[("0_Seen", "0")] = "0_Seen"
    transitions[("0_Seen", "1")] = "1_Seen"
    transitions[("0_Seen", "2")] = "0_Seen"

    transitions[("1_Seen", "0")] = "1_Seen"
    transitions[("1_Seen", "1")] = "2_Seen"
    transitions[("1_Seen", "2")] = "1_Seen"

    transitions[("2_Seen", "0")] = "2_Seen"
    transitions[("2_Seen", "1")] = "3_Seen"
    transitions[("2_Seen", "2")] = "2_Seen"

    transitions[("3_Seen", "0")] = "3_Seen"
    transitions[("3_Seen", "1")] = "4+_Seen"
    transitions[("3_Seen", "2")] = "3_Seen"

    transitions[("4+_Seen", "0")] = "4+_Seen"
    transitions[("4+_Seen", "1")] = "4+_Seen"
    transitions[("4+_Seen", "2")] = "4+_Seen"

    # Create the DFA, which should not raise an exception.
    dfa = Dfa(alphabet, states, accepting_states, start_state, transitions)

    # Construct an Accumulated Cost Dfa with a missing cost for
    # the state "2_Seen"
    cost_map = {}
    cost_map["0_Seen"] =  0
    cost_map["1_Seen"] =  1
    cost_map["3_Seen"] =  3
    cost_map["4+_Seen"] = Fraction(4.5)

    # Creates an Accumulated Cost DFA, which should raise an exception.
    with pytest.raises(ValueError):
        AccumulatedCostDfa(dfa, cost_map, 10)

def test_accumulated_cost_dfa_irrational():
    """ Creates a simple complete AccumulatedCostDfa with a cost
    that is of a type not descending from Rational and ensures
    this raises an error.
    """

    # Create a DFA that accepts all strings with at least one "1"
    # symbol, but tracks up to 4 seen.
    alphabet = {"0", "1", "2"}
    states = {"0_Seen", "1_Seen", "2_Seen", "3_Seen", "4+_Seen"}
    accepting_states = {"1_Seen", "2_Seen", "3_Seen", "4+_Seen"}
    start_state = "0_Seen"

    transitions = {}

    transitions[("0_Seen", "0")] = "0_Seen"
    transitions[("0_Seen", "1")] = "1_Seen"
    transitions[("0_Seen", "2")] = "0_Seen"

    transitions[("1_Seen", "0")] = "1_Seen"
    transitions[("1_Seen", "1")] = "2_Seen"
    transitions[("1_Seen", "2")] = "1_Seen"

    transitions[("2_Seen", "0")] = "2_Seen"
    transitions[("2_Seen", "1")] = "3_Seen"
    transitions[("2_Seen", "2")] = "2_Seen"

    transitions[("3_Seen", "0")] = "3_Seen"
    transitions[("3_Seen", "1")] = "4+_Seen"
    transitions[("3_Seen", "2")] = "3_Seen"

    transitions[("4+_Seen", "0")] = "4+_Seen"
    transitions[("4+_Seen", "1")] = "4+_Seen"
    transitions[("4+_Seen", "2")] = "4+_Seen"

    # Create the DFA, which should not raise an exception.
    dfa = Dfa(alphabet, states, accepting_states, start_state, transitions)

    # Construct a Accumulated Cost Dfa which is complete.
    # However, 4.5 is actually of type Real, not Rational.
    cost_map = {}
    cost_map["0_Seen"] =  0
    cost_map["1_Seen"] =  1
    cost_map["2_Seen"] =  2
    cost_map["3_Seen"] =  3
    cost_map["4+_Seen"] = 4.5

    # Creates an Accumulated Cost DFA, which should raise an exception.
    with pytest.raises(ValueError):
        AccumulatedCostDfa(dfa, cost_map, 10)

def test_accumulated_cost_dfa_no_costs():
    """ Creates a simple complete AccumulatedCostDfa with no costs
    and ensures this does not raise an error.
    """

    # Create a DFA that accepts all strings with at least one "1"
    # symbol, but tracks up to 4 seen.
    alphabet = {"0", "1", "2"}
    states = {"0_Seen", "1_Seen", "2_Seen", "3_Seen", "4+_Seen"}
    accepting_states = {}
    start_state = "0_Seen"

    transitions = {}

    transitions[("0_Seen", "0")] = "0_Seen"
    transitions[("0_Seen", "1")] = "1_Seen"
    transitions[("0_Seen", "2")] = "0_Seen"

    transitions[("1_Seen", "0")] = "1_Seen"
    transitions[("1_Seen", "1")] = "2_Seen"
    transitions[("1_Seen", "2")] = "1_Seen"

    transitions[("2_Seen", "0")] = "2_Seen"
    transitions[("2_Seen", "1")] = "3_Seen"
    transitions[("2_Seen", "2")] = "2_Seen"

    transitions[("3_Seen", "0")] = "3_Seen"
    transitions[("3_Seen", "1")] = "4+_Seen"
    transitions[("3_Seen", "2")] = "3_Seen"

    transitions[("4+_Seen", "0")] = "4+_Seen"
    transitions[("4+_Seen", "1")] = "4+_Seen"
    transitions[("4+_Seen", "2")] = "4+_Seen"

    # Create the DFA, which should not raise an exception.
    dfa = Dfa(alphabet, states, accepting_states, start_state, transitions)

    cost_map = {}
    cost_map["0_Seen"] = 0
    cost_map["1_Seen"] =  1
    cost_map["2_Seen"] =  2
    cost_map["3_Seen"] =  3
    cost_map["4+_Seen"] = 4

    # Creates an AccumulatedCostDfa, which should not raise an exception.
    static_cost_dfa = AccumulatedCostDfa(dfa, cost_map, 10)

    # Checks select strings against the Accumulated Cost DFA
    assert static_cost_dfa.decompose() == dict()

    assert static_cost_dfa.cost(list("")) is None
    assert static_cost_dfa.cost(list("0")) is None
    assert static_cost_dfa.cost(list("1")) is None
    assert static_cost_dfa.cost(list("2")) is None

    assert static_cost_dfa.cost(list("00")) is None
    assert static_cost_dfa.cost(list("01")) is None
    assert static_cost_dfa.cost(list("11")) is None
    assert static_cost_dfa.cost(list("12")) is None
    assert static_cost_dfa.cost(list("22")) is None
    assert static_cost_dfa.cost(list("02")) is None

    assert static_cost_dfa.cost(list("011")) is None
    assert static_cost_dfa.cost(list("112")) is None
    assert static_cost_dfa.cost(list("012")) is None
    assert static_cost_dfa.cost(list("111")) is None

    assert static_cost_dfa.cost(list("01112")) is None
    assert static_cost_dfa.cost(list("11111")) is None

    assert static_cost_dfa.cost(list("1111111111111111111111111111111")) is None
    assert static_cost_dfa.cost(list("0010021200001100011002020011022")) is None
    assert static_cost_dfa.cost(list("0000000022020020202000222220000")) is None
    assert static_cost_dfa.cost(list("0000000002220202020202020022001")) is None

def test_accumulated_cost_dfa_cost_set():
    """ Creates a simple AccumulatedCostDfa and ensures that the feasible cost
    set is calculated correctly.
    """
    # Create a simple DFA and AccumulatedCostDfa
    alphabet = {"0", "1"}
    states = {"State1", "State2", "State3", "State4", "Sink"}
    accepting_states = {"State4"}
    start_state = "State1"

    transitions = {}

    transitions[("State1", "0")] = "State2"
    transitions[("State1", "1")] = "State3"

    transitions[("State2", "0")] = "State1"
    transitions[("State2", "1")] = "State4"

    transitions[("State3", "0")] = "Sink"
    transitions[("State3", "1")] = "State2"

    transitions[("State4", "0")] = "State4"
    transitions[("State4", "1")] = "State4"

    transitions[("Sink", "0")] = "Sink"
    transitions[("Sink", "1")] = "Sink"

    dfa = Dfa(alphabet, states, accepting_states, start_state, transitions)

    cost_map = {}
    cost_map["State1"] = 1
    cost_map["State2"] = 2
    cost_map["State3"] = 3
    cost_map["State4"] = 4
    cost_map["Sink"] = 999999

    accumulated_cost_dfa = AccumulatedCostDfa(dfa, cost_map, max_word_length=6)

    assert set(accumulated_cost_dfa.costs) == {7, 10, 11, 13, 14, 15, 16, 17, 18, 19, 22, 23}

def test_accumulated_cost_dfa_decompose():
    """ Creates a simple AccumulatedCostDfa and ensures that the decompose function
    works correctly.
    """
    # Create a simple DFA and AccumulatedCostDfa
    alphabet = {"0", "1"}
    states = {"State1", "State2", "State3", "State4", "Sink"}
    accepting_states = {"State4"}
    start_state = "State1"

    transitions = {}

    transitions[("State1", "0")] = "State2"
    transitions[("State1", "1")] = "State3"

    transitions[("State2", "0")] = "State1"
    transitions[("State2", "1")] = "State4"

    transitions[("State3", "0")] = "Sink"
    transitions[("State3", "1")] = "State2"

    transitions[("State4", "0")] = "State4"
    transitions[("State4", "1")] = "State4"

    transitions[("Sink", "0")] = "Sink"
    transitions[("Sink", "1")] = "Sink"

    dfa = Dfa(alphabet, states, accepting_states, start_state, transitions)

    cost_map = {}
    cost_map["State1"] = 1
    cost_map["State2"] = 2
    cost_map["State3"] = 3
    cost_map["State4"] = 4
    cost_map["Sink"] = 999999

    max_word_length = 6

    accumulated_cost_dfa = AccumulatedCostDfa(dfa, cost_map, max_word_length)

    # Decompose the AccumulatedCostDfa and ensure that an indicator function
    # for each cost is present.
    decomposed_cost_func = accumulated_cost_dfa.decompose()

    assert set(accumulated_cost_dfa.costs) == set(decomposed_cost_func.keys())

    # Iterate through every possible word that has length <= the number
    # of states in the new Dfa to ensure they are equivalent.
    for word_length in range(max_word_length+1):
        for word in itertools.product(alphabet, repeat=word_length):
            cost = accumulated_cost_dfa.cost(word)

            if cost is None:
                for spec in decomposed_cost_func.values():
                    assert not spec.accepts(word)
            else:
                assert decomposed_cost_func[cost].accepts(word)

                for t_cost in accumulated_cost_dfa.costs:
                    if t_cost != cost:
                        assert not decomposed_cost_func[t_cost].accepts(word)

###################################################################################################
# Randomized Tests
###################################################################################################

# Randomized tests default parameters
_RANDOM_ACC_COST_DFA_TEST_NUM_ITERS = 1000    # Default to 1000, but can set lower when writing new tests.

_RANDOM_ACC_COST_DFA_MIN_COST = 0
_RANDOM_ACC_COST_DFA_MAX_COST = 10

_RANDOM_ACC_COST_DFA_MIN_WORD_LENGTH = 0
_RANDOM_ACC_COST_DFA_MAX_WORLD_LENGTH = 10

@pytest.mark.slow
def test_accumulated_cost_dfa_decompose_random():
    """ For _RANDOM_ACC_COST_DFA_TEST_NUM_ITERS iters, create a
    AccumulatedCostDfa and ensure that the decompose function works correctly.
    """
    for _ in range(_RANDOM_ACC_COST_DFA_TEST_NUM_ITERS):
        dfa = generate_random_dfa()

        cost_map = {}

        for state in dfa.states:
            cost_map[state] = random.randint(_RANDOM_ACC_COST_DFA_MIN_COST, _RANDOM_ACC_COST_DFA_MAX_COST)

        max_word_length = random.randint(_RANDOM_ACC_COST_DFA_MIN_WORD_LENGTH, _RANDOM_ACC_COST_DFA_MAX_WORLD_LENGTH)

        accumulated_cost_dfa = AccumulatedCostDfa(dfa, cost_map, max_word_length)

        decomposed_cost_func = accumulated_cost_dfa.decompose()

        assert set(accumulated_cost_dfa.costs) == set(decomposed_cost_func.keys())

        # Iterate through every possible word that has length <= the number
        # of states in the new Dfa to ensure they are equivalent.
        for word_length in range(max_word_length+1):
            for word in itertools.product(dfa.alphabet, repeat=word_length):
                cost = accumulated_cost_dfa.cost(word)

                if cost is None:
                    for spec in decomposed_cost_func.values():
                        assert not spec.accepts(word)
                else:
                    assert decomposed_cost_func[cost].accepts(word)

                    for t_cost in accumulated_cost_dfa.costs:
                        if t_cost != cost:
                            assert not decomposed_cost_func[t_cost].accepts(word)
