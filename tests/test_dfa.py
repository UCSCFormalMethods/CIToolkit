""" Tests for the Dfa class"""

import pytest

from citoolkit.specifications.spec import Spec
from citoolkit.specifications.dfa import Dfa, State

###################################################################################################
# Tests
###################################################################################################

def test_dfa_complete():
    """ Creates a simple complete Dfa and ensures
    this does not raise an error.
    """

    # Create a DFA that only accepts strings that contain 3 "1"
    # symbols in a row with no "2" inputs after them.
    alphabet = {"0", "1", "2"}
    states = {"0_Seen", "1_Seen", "2_Seen", "3_Seen"}
    accepting_states = {"3_Seen"}
    start_state = "0_Seen"

    # Initialize transitions map so that all transitions go
    # to "0_Seen"
    transitions = {}
    for state in states:
        for symbol in alphabet:
            transitions[(state, symbol)] = "0_Seen"

    # Complete transitions map.
    transitions[("0_Seen", "1")] = "1_Seen"
    transitions[("1_Seen", "1")] = "2_Seen"
    transitions[("2_Seen", "1")] = "3_Seen"

    transitions[("3_Seen", "0")] = "3_Seen"
    transitions[("3_Seen", "1")] = "3_Seen"

    # Create the DFA, which should not raise an exception.
    Dfa(alphabet, states, accepting_states, start_state, transitions)

def test_dfa_not_complete():
    """ Attempts to create a simple incomplete Dfa and ensures
    that this raises a ValueError.
    """

    # Create a DFA that only accepts strings that contain 3 "1"
    # symbols in a row with no "2" inputs after them.
    alphabet = {"0", "1", "2"}
    states = {"0_Seen", "1_Seen", "2_Seen", "3_Seen"}
    accepting_states = {"3_Seen"}
    start_state = "0_Seen"

    transitions = {}

    # Partially completes transitions map.
    transitions[("0_Seen", "1")] = "1_Seen"
    transitions[("1_Seen", "1")] = "2_Seen"
    transitions[("2_Seen", "1")] = "3_Seen"

    transitions[("3_Seen", "0")] = "3_Seen"
    transitions[("3_Seen", "1")] = "3_Seen"

    # Create the DFA and check select strings against Dfa
    with pytest.raises(ValueError):
        Dfa(alphabet, states, accepting_states, start_state, transitions)

def test_dfa_string_states():
    """ Creates a simple Dfa and ensures that select
    words are correctly accepted or rejected. Dfa is
    constructed with string states.
    """

    # Create a DFA that only accepts strings that contain 3 "1"
    # symbols in a row with no "2" inputs after them.
    alphabet = {"0", "1", "2"}
    states = {"0_Seen", "1_Seen", "2_Seen", "3_Seen"}
    accepting_states = {"3_Seen"}
    start_state = "0_Seen"

    # Initialize transitions map so that all transitions go
    # to "0_Seen"
    transitions = {}
    for state in states:
        for symbol in alphabet:
            transitions[(state, symbol)] = "0_Seen"

    # Complete transitions map.
    transitions[("0_Seen", "1")] = "1_Seen"
    transitions[("1_Seen", "1")] = "2_Seen"
    transitions[("2_Seen", "1")] = "3_Seen"

    transitions[("3_Seen", "0")] = "3_Seen"
    transitions[("3_Seen", "1")] = "3_Seen"

    # Create the DFA and check select strings against Dfa
    dfa = Dfa(alphabet, states, accepting_states, start_state, transitions)

    assert not dfa.accepts("")
    assert not dfa.accepts("0")
    assert not dfa.accepts("1")
    assert not dfa.accepts("2")

    assert dfa.accepts("111")
    assert not dfa.accepts("1112")

    assert not dfa.accepts("000")
    assert not dfa.accepts("222")

    assert dfa.accepts("01110")

    assert not dfa.accepts("00000011000020011100020001")
    assert dfa.accepts("0000001100002001110002000111")

def test_dfa_class_states():
    """ Creates a simple Dfa and ensures that select
    words are correctly accepted or rejected. Dfa is
    constructed with State class states.
    """

    # Create a DFA that only accepts strings that contain 3 "1"
    # symbols in a row with no "2" inputs after them.
    alphabet = {"0", "1", "2"}
    states = {State("0_Seen"), State("1_Seen"), State("2_Seen"), State("3_Seen")}
    accepting_states = {State("3_Seen")}
    start_state = State("0_Seen")

    # Initialize transitions map so that all transitions go
    # to "0_Seen"
    transitions = {}
    for state in states:
        for symbol in alphabet:
            transitions[(state, symbol)] = State("0_Seen")

    # Complete transitions map.
    transitions[(State("0_Seen"), "1")] = State("1_Seen")
    transitions[(State("1_Seen"), "1")] = State("2_Seen")
    transitions[(State("2_Seen"), "1")] = State("3_Seen")

    transitions[(State("3_Seen"), "0")] = State("3_Seen")
    transitions[(State("3_Seen"), "1")] = State("3_Seen")

    # Create the DFA and check select strings against Dfa
    dfa = Dfa(alphabet, states, accepting_states, start_state, transitions)

    assert not dfa.accepts("")
    assert not dfa.accepts("0")
    assert not dfa.accepts("1")
    assert not dfa.accepts("2")

    assert dfa.accepts("111")
    assert not dfa.accepts("1112")

    assert not dfa.accepts("000")
    assert not dfa.accepts("222")

    assert dfa.accepts("01110")

    assert not dfa.accepts("00000011000020011100020001")
    assert dfa.accepts("0000001100002001110002000111")

def test_dfa_mixed_states():
    """ Creates a simple Dfa and ensures that select
    words are correctly accepted or rejected. Dfa is
    constructed with a mix of string and State class
    states.
    """

    # Create a DFA that only accepts strings that contain 3 "1"
    # symbols in a row with no "2" inputs after them.
    alphabet = {"0", "1", "2"}
    states = {State("0_Seen"), "1_Seen", "2_Seen", State("3_Seen")}
    accepting_states = {"3_Seen"}
    start_state = State("0_Seen")

    # Initialize transitions map so that all transitions go
    # to "0_Seen"
    transitions = {}
    for state in states:
        for symbol in alphabet:
            transitions[(state, symbol)] = "0_Seen"

    # Complete transitions map.
    transitions[(State("0_Seen"), "1")] = State("1_Seen")
    transitions[("1_Seen", "1")] = State("2_Seen")
    transitions[(State("2_Seen"), "1")] = "3_Seen"

    transitions[(State("3_Seen"), "0")] = State("3_Seen")
    transitions[("3_Seen", "1")] = "3_Seen"

    # Create the DFA and check select strings against Dfa
    dfa = Dfa(alphabet, states, accepting_states, start_state, transitions)

    assert not dfa.accepts("")
    assert not dfa.accepts("0")
    assert not dfa.accepts("1")
    assert not dfa.accepts("2")

    assert dfa.accepts("111")
    assert not dfa.accepts("1112")

    assert not dfa.accepts("000")
    assert not dfa.accepts("222")

    assert dfa.accepts("01110")

    assert not dfa.accepts("00000011000020011100020001")
    assert dfa.accepts("0000001100002001110002000111")

def test_dfa_minimize_no_reduction():
    """ Creates a simple Dfa that is already minimal,
    minimizes it, and ensures that select words are
    correctly accepted or rejected.
    """

    # Create a DFA that only accepts strings that contain 3 "1"
    # symbols in a row with no "2" inputs after them.
    alphabet = {"0", "1", "2"}
    states = {"0_Seen", "1_Seen", "2_Seen", "3_Seen"}
    accepting_states = {"3_Seen"}
    start_state = "0_Seen"

    # Initialize transitions map so that all transitions go
    # to "0_Seen"
    transitions = {}
    for state in states:
        for symbol in alphabet:
            transitions[(state, symbol)] = "0_Seen"

    # Complete transitions map.
    transitions[("0_Seen", "1")] = "1_Seen"
    transitions[("1_Seen", "1")] = "2_Seen"
    transitions[("2_Seen", "1")] = "3_Seen"

    transitions[("3_Seen", "0")] = "3_Seen"
    transitions[("3_Seen", "1")] = "3_Seen"

    # Create the DFA and minimizes it.
    dfa = Dfa(alphabet, states, accepting_states, start_state, transitions)

    minimized_dfa = dfa.minimize()

    # Assert the minimized Dfa's size is the same as the
    # original and check select strings against the minimized Dfa

    assert len(dfa.states) == len(minimized_dfa.states)

    assert not minimized_dfa.accepts("")
    assert not minimized_dfa.accepts("0")
    assert not minimized_dfa.accepts("1")
    assert not minimized_dfa.accepts("2")

    assert minimized_dfa.accepts("111")
    assert not minimized_dfa.accepts("1112")

    assert not minimized_dfa.accepts("000")
    assert not minimized_dfa.accepts("222")

    assert minimized_dfa.accepts("01110")

    assert not minimized_dfa.accepts("00000011000020011100020001")
    assert minimized_dfa.accepts("0000001100002001110002000111")

def test_dfa_minimize_reduction():
    """ Creates a Dfa that has many redundancies,
    minimizes it, and ensures that select words are
    correctly accepted or rejected.
    """
    # Create a very redundant DFA that accepts if and only if the
    # string contains a "1" symbol before any 0 symbols
    alphabet = {"0", "1", "2"}

    s_states = {"Start_A", "Start_B", "Start_C"}
    a_states = {"Accept_A", "Accept_B", "Accept_C"}
    r_states = {"Reject_A", "Reject_B", "Reject_C"}
    dr_states = {"DeadReject_A", "DeadReject_B"}
    da_states = {"DeadAccept_A", "DeadAccept_B"}
    states = s_states | a_states | r_states | da_states

    accepting_states = a_states
    start_state = "Start_A"

    # Create transitions map
    transitions = {}

    # S state transitions
    for s_state in s_states:
        transitions[(s_state, "0")] = "Reject_A"
        transitions[(s_state, "1")] = "Accept_A"

    transitions[("Start_A", "2")] = "Start_B"
    transitions[("Start_B", "2")] = "Start_C"
    transitions[("Start_C", "2")] = "Start_C"

    # A state transitions
    for symbol in alphabet:
        transitions[("Accept_A", symbol)] = "Accept_B"
    for symbol in alphabet:
        transitions[("Accept_B", symbol)] = "Accept_C"
    for symbol in alphabet:
        transitions[("Accept_C", symbol)] = "Accept_C"

    # R state transitions
    for symbol in alphabet:
        transitions[("Reject_A", symbol)] = "Reject_B"
    for symbol in alphabet:
        transitions[("Reject_B", symbol)] = "Reject_C"
    for symbol in alphabet:
        transitions[("Reject_C", symbol)] = "Reject_C"

    # Dead state transitions
    transitions[("DeadReject_A", "0")] = "Accept_A"
    transitions[("DeadReject_A", "1")] = "Reject_A"
    transitions[("DeadReject_A", "2")] = "Start_A"

    transitions[("DeadReject_B", "0")] = "DeadReject_B"
    transitions[("DeadReject_B", "1")] = "DeadReject_B"
    transitions[("DeadReject_B", "2")] = "DeadReject_B"

    transitions[("DeadAccept_A", "0")] = "Accept_A"
    transitions[("DeadAccept_A", "1")] = "Reject_A"
    transitions[("DeadAccept_A", "2")] = "Start_A"

    transitions[("DeadAccept_B", "0")] = "DeadAccept_B"
    transitions[("DeadAccept_B", "1")] = "DeadAccept_B"
    transitions[("DeadAccept_B", "2")] = "DeadAccept_B"

    # Create the DFA and minimizes it.
    dfa = Dfa(alphabet, states, accepting_states, start_state, transitions)

    minimized_dfa = dfa.minimize()

    # Assert the minimized Dfa's size appropriately minimized
    # and check select strings against the two DFAs.

    assert len(minimized_dfa.states) == 3

    assert not minimized_dfa.accepts("") and not dfa.accepts("")
    assert not minimized_dfa.accepts("0") and not dfa.accepts("0")
    assert minimized_dfa.accepts("1") and dfa.accepts("1")
    assert not minimized_dfa.accepts("2") and not dfa.accepts("2")

    assert not minimized_dfa.accepts("000") and not dfa.accepts("000")
    assert minimized_dfa.accepts("111") and dfa.accepts("111")
    assert not minimized_dfa.accepts("222") and not dfa.accepts("222")

    assert not minimized_dfa.accepts("2220000110011000020100002000") and not dfa.accepts("2220000110011000020100002000")
    assert minimized_dfa.accepts("222210000001100002001110002000111") and dfa.accepts("222210000001100002001110002000111")

###################################################################################################
# Helper Functions
###################################################################################################
