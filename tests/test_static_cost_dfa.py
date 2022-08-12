""" Tests for the StaticCostDfa class"""

from fractions import Fraction

import pytest

from citoolkit.specifications.dfa import Dfa
from citoolkit.costfunctions.static_cost_dfa import StaticCostDfa

###################################################################################################
# Basic Tests
###################################################################################################

def test_static_cost_dfa_complete():
    """ Creates a simple complete StaticCostDfa and ensures
    this does not raise an error.
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

    # Construct a Static Cost DFA that assigns cost to strings based off
    # of the number of "1" symbols they have seen.
    cost_map = {}
    cost_map["1_Seen"] =  1
    cost_map["2_Seen"] =  2
    cost_map["3_Seen"] =  3
    cost_map["4+_Seen"] = Fraction(4.5)

    # Creates a Static Cost DFA, which should not raise an exception.
    static_cost_dfa = StaticCostDfa(dfa, cost_map)

    # Checks that the parsed costs are correct.
    assert len(static_cost_dfa.costs) == 4
    assert 1 in static_cost_dfa.costs
    assert 2 in static_cost_dfa.costs
    assert 3 in static_cost_dfa.costs
    assert 4.5 in static_cost_dfa.costs
    assert Fraction(4.5) in static_cost_dfa.costs
    assert Fraction(9, 2) in static_cost_dfa.costs

def test_static_cost_dfa_incomplete():
    """ Creates a simple incomplete StaticCostDfa and ensures
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

    # Construct a Static Cost DFA that assigns cost to strings based off
    # of the number of "1" symbols they have seen. However, neglect to add
    # a cost for "2_Seen", rendering this an incomplete StaticCostDfa.
    cost_map = {}
    cost_map["1_Seen"] =  1
    cost_map["3_Seen"] =  3
    cost_map["4+_Seen"] = Fraction(4.5)

    # Creates a Static Cost DFA, which should raise an exception.
    with pytest.raises(ValueError):
        StaticCostDfa(dfa, cost_map)

def test_static_cost_dfa_irrational():
    """ Creates a simple complete StaticCostDfa with a cost
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

    # Construct a Static Cost DFA that assigns cost to strings based off
    # of the number of "1" symbols they have seen. However, 4.5 is
    # actually of type Real, not Rational.
    cost_map = {}
    cost_map["1_Seen"] =  1
    cost_map["2_Seen"] =  2
    cost_map["3_Seen"] =  3
    cost_map["4+_Seen"] = 4.5

    # Creates a Static Cost DFA, which should raise an exception.
    with pytest.raises(ValueError):
        StaticCostDfa(dfa, cost_map)

def test_static_cost_dfa_no_costs():
    """ Creates a simple complete StaticCostDfa with no costs
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

    # Creates a StaticCostDFA, which should not raise an exception.
    static_cost_dfa = StaticCostDfa(dfa, cost_map)

    # Checks select strings against the Static Cost DFA
    assert static_cost_dfa.decompose() == {}

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
