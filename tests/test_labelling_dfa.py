""" Tests for the LabellingDfa class"""

import pytest

from citoolkit.specifications.dfa import Dfa
from citoolkit.labellingfunctions.labelling_dfa import LabellingDfa

###################################################################################################
# Basic Tests
###################################################################################################

def test_labelled_dfa_complete():
    """ Creates a simple complete LabellingDfa and ensures
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

    # Construct a Labelling DFA that labels strings based off of whether
    # the words contain 1-2 "1" symbols or 3+ "1" symbols.

    label_map = {}
    label_map["1_Seen"] =  "1-2_Label"
    label_map["2_Seen"] =  "1-2_Label"
    label_map["3_Seen"] =  "3+_Label"
    label_map["4+_Seen"] = "3+_Label"

    # Creates a Labelling DFA, which should not raise an exception.
    labelling_dfa = LabellingDfa(dfa, label_map)

    # Checks that the parsed labels are correct.
    assert len(labelling_dfa.labels) == 2
    assert "1-2_Label" in labelling_dfa.labels
    assert "3+_Label" in labelling_dfa.labels

def test_labelled_dfa_incomplete():
    """ Creates a simple incomplete LabellingDfa and ensures
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

    # Construct a Labelling DFA that labels strings based off of whether
    # the words contain 1-2 "1" symbols or 3+ "1" symbols. However, neglect
    # to label the "2_Seen" state rendering this an incomplete Labelling DFA

    label_map = {}
    label_map["1_Seen"] =  "1-2_Label"
    label_map["3_Seen"] =  "3+_Label"
    label_map["4+_Seen"] = "3+_Label"

    # Creates a Labelling DFA, which should raise an exception.
    with pytest.raises(ValueError):
        LabellingDfa(dfa, label_map)

def test_labelled_dfa_no_labels():
    """ Creates a simple complete LabellingDfa with no labels
    and ensures this does not rais an error.
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

    # Construct a Labelling DFA with no labels.
    label_map = {}

    # Creates a Labelling DFA, which should not raise an exception.
    labelling_dfa = LabellingDfa(dfa, label_map)

    # Checks select strings against the Labelling DFA
    assert labelling_dfa.decompose() == {}

    assert labelling_dfa.label(list("")) is None
    assert labelling_dfa.label(list("0")) is None
    assert labelling_dfa.label(list("1")) is None
    assert labelling_dfa.label(list("2")) is None

    assert labelling_dfa.label(list("00")) is None
    assert labelling_dfa.label(list("01")) is None
    assert labelling_dfa.label(list("11")) is None
    assert labelling_dfa.label(list("12")) is None
    assert labelling_dfa.label(list("22")) is None
    assert labelling_dfa.label(list("02")) is None

    assert labelling_dfa.label(list("011")) is None
    assert labelling_dfa.label(list("112")) is None
    assert labelling_dfa.label(list("012")) is None
    assert labelling_dfa.label(list("111")) is None

    assert labelling_dfa.label(list("01112")) is None
    assert labelling_dfa.label(list("11111")) is None

    assert labelling_dfa.label(list("1111111111111111111111111111111")) is None
    assert labelling_dfa.label(list("0010021200001100011002020011022")) is None
    assert labelling_dfa.label(list("0000000022020020202000222220000")) is None
    assert labelling_dfa.label(list("0000000002220202020202020022001")) is None
