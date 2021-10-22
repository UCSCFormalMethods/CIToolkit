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

def test_accumulated_cost_dfa_cost():
    """ Creates a simple AccumulatedCostDfa and ensures that costs
    are calculated correctly.
    """
    # Create a simple DFA
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

    # Create cost_map and accumulated cost dfa mapping each state to a
    # cost equal to its state number.
    cost_map = {}
    cost_map["State1"] = 1
    cost_map["State2"] = 2
    cost_map["State3"] = 3
    cost_map["State4"] = 4
    cost_map["Sink"] = 999999

    accumulated_cost_dfa = AccumulatedCostDfa(dfa, cost_map, max_word_length=6)

    # Check that select words have the correct cost.
    assert accumulated_cost_dfa.cost(tuple("0001")) == 10
    assert accumulated_cost_dfa.cost(tuple("0000")) is None
    assert accumulated_cost_dfa.cost(tuple("110111")) == 16

def test_accumulated_cost_dfa_cost_set():
    """ Creates a simple AccumulatedCostDfa and ensures that the feasible cost
    set is calculated correctly.
    """
    # Create a simple DFA
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
    # Create a simple DFA
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

@pytest.mark.slow
def test_accumulated_cost_dfa_decompose_random():
    for _ in range(1000):
        dfa = generate_random_dfa()

        cost_map = {}

        for state in dfa.states:
            cost_map[state] = random.randint(0, 10)

        max_word_length = random.randint(0,10)

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
