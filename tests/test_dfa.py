""" Tests for the Dfa class"""

import pytest

from citoolkit.specifications.spec import AbstractSpec
from citoolkit.specifications.dfa import Dfa, State

###################################################################################################
# Basic Tests
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

    assert not dfa.accepts([])
    assert not dfa.accepts(list("0"))
    assert not dfa.accepts(list("1"))
    assert not dfa.accepts(list("2"))

    assert dfa.accepts(list("111"))
    assert not dfa.accepts(list("1112"))

    assert not dfa.accepts(list("000"))
    assert not dfa.accepts(list("222"))

    assert dfa.accepts(list("01110"))

    assert not dfa.accepts(list("00000011000020011100020001"))
    assert dfa.accepts(list("0000001100002001110002000111"))

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

    assert not dfa.accepts([])
    assert not dfa.accepts(list("0"))
    assert not dfa.accepts(list("1"))
    assert not dfa.accepts(list("2"))

    assert dfa.accepts(list("111"))
    assert not dfa.accepts(list("1112"))

    assert not dfa.accepts(list("000"))
    assert not dfa.accepts(list("222"))

    assert dfa.accepts(list("01110"))

    assert not dfa.accepts(list("00000011000020011100020001"))
    assert dfa.accepts(list("0000001100002001110002000111"))

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

    assert not dfa.accepts([])
    assert not dfa.accepts(list("0"))
    assert not dfa.accepts(list("1"))
    assert not dfa.accepts(list("2"))

    assert dfa.accepts(list("111"))
    assert not dfa.accepts(list("1112"))

    assert not dfa.accepts(list("000"))
    assert not dfa.accepts(list("222"))

    assert dfa.accepts(list("01110"))

    assert not dfa.accepts(list("00000011000020011100020001"))
    assert dfa.accepts(list("0000001100002001110002000111"))

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

    assert not dfa.accepts([])
    assert not dfa.accepts(list("0"))
    assert not dfa.accepts(list("1"))
    assert not dfa.accepts(list("2"))

    assert dfa.accepts(list("111"))
    assert not dfa.accepts(list("1112"))

    assert not dfa.accepts(list("000"))
    assert not dfa.accepts(list("222"))

    assert dfa.accepts(list("01110"))

    assert not dfa.accepts(list("00000011000020011100020001"))
    assert dfa.accepts(list("0000001100002001110002000111"))

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
    states = s_states | a_states | r_states | dr_states |da_states

    accepting_states = a_states | da_states
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

    assert not minimized_dfa.accepts([]) and not dfa.accepts([])
    assert not minimized_dfa.accepts(list("0")) and not dfa.accepts(list("0"))
    assert minimized_dfa.accepts(list("1")) and dfa.accepts(list("1"))
    assert not minimized_dfa.accepts(list("2")) and not dfa.accepts(list("2"))

    assert not minimized_dfa.accepts(list("000")) and not dfa.accepts(list("000"))
    assert minimized_dfa.accepts(list("111")) and dfa.accepts(list("111"))
    assert not minimized_dfa.accepts(list("222")) and not dfa.accepts(list("222"))

    assert not minimized_dfa.accepts(list("2220000110011000020100002000")) and not dfa.accepts(list("2220000110011000020100002000"))
    assert minimized_dfa.accepts(list("222210000001100002001110002000111")) and dfa.accepts(list("222210000001100002001110002000111"))

def test_dfa_union():
    """ Creates two DFAs, one which accepts iff
    a string contains a "1" symbol and another which
    accepts iff a string contains a "2" symbol. Then ensure
    that their symbolic and explicit union have an equivalent
    and correct language
    """

    alphabet = {"0","1","2"}

    # Create DFA that accepts once it encounters a "1"
    states_1 = {"Reject", "Accept"}
    accepting_states_1 = {"Accept"}
    start_state_1 = "Reject"

    transitions_1 = {}

    transitions_1[("Reject", "0")] = "Reject"
    transitions_1[("Reject", "1")] = "Accept"
    transitions_1[("Reject", "2")] = "Reject"

    transitions_1[("Accept", "0")] = "Accept"
    transitions_1[("Accept", "1")] = "Accept"
    transitions_1[("Accept", "2")] = "Accept"

    dfa_1 = Dfa(alphabet, states_1, accepting_states_1, start_state_1, transitions_1)

    # Create DFA that accepts once it encounters a "2"
    states_2 = {"Reject", "Accept"}
    accepting_states_2 = {"Accept"}
    start_state_2 = "Reject"

    transitions_2 = {}

    transitions_2[("Reject", "0")] = "Reject"
    transitions_2[("Reject", "1")] = "Reject"
    transitions_2[("Reject", "2")] = "Accept"

    transitions_2[("Accept", "0")] = "Accept"
    transitions_2[("Accept", "1")] = "Accept"
    transitions_2[("Accept", "2")] = "Accept"

    dfa_2 = Dfa(alphabet, states_2, accepting_states_2, start_state_2, transitions_2)

    # Create abstract spec for the union of dfa_1 and dfa_2. Then compute its explicit form.
    abstract_union = dfa_1 | dfa_2

    explicit_union = abstract_union.explicit()

    assert isinstance(abstract_union, AbstractSpec)
    assert isinstance(explicit_union, Dfa)

    assert not abstract_union.accepts([]) and not explicit_union.accepts([])

    assert not abstract_union.accepts(list("0")) and not explicit_union.accepts(list("0"))
    assert abstract_union.accepts(list("1")) and explicit_union.accepts(list("1"))
    assert abstract_union.accepts(list("2")) and explicit_union.accepts(list("2"))

    assert not abstract_union.accepts(list("000")) and not explicit_union.accepts(list("000"))
    assert abstract_union.accepts(list("111")) and explicit_union.accepts(list("111"))
    assert abstract_union.accepts(list("222")) and explicit_union.accepts(list("222"))

    assert abstract_union.accepts(list("010")) and explicit_union.accepts(list("010"))
    assert abstract_union.accepts(list("020")) and explicit_union.accepts(list("020"))
    assert abstract_union.accepts(list("12")) and explicit_union.accepts(list("12"))

def test_dfa_intersection():
    """ Creates two DFAs, one which accepts iff
    a string contains a "1" symbol and another which
    accepts iff a string contains a "2" symbol. Then ensure
    that their symbolic and explicit intersection
    have an equivalent and correct language
    """

    alphabet = {"0","1","2"}

    # Create DFA that accepts once it encounters a "1"
    states_1 = {"Reject", "Accept"}
    accepting_states_1 = {"Accept"}
    start_state_1 = "Reject"

    transitions_1 = {}

    transitions_1[("Reject", "0")] = "Reject"
    transitions_1[("Reject", "1")] = "Accept"
    transitions_1[("Reject", "2")] = "Reject"

    transitions_1[("Accept", "0")] = "Accept"
    transitions_1[("Accept", "1")] = "Accept"
    transitions_1[("Accept", "2")] = "Accept"

    dfa_1 = Dfa(alphabet, states_1, accepting_states_1, start_state_1, transitions_1)

    # Create DFA that accepts once it encounters a "2"
    states_2 = {"Reject", "Accept"}
    accepting_states_2 = {"Accept"}
    start_state_2 = "Reject"

    transitions_2 = {}

    transitions_2[("Reject", "0")] = "Reject"
    transitions_2[("Reject", "1")] = "Reject"
    transitions_2[("Reject", "2")] = "Accept"

    transitions_2[("Accept", "0")] = "Accept"
    transitions_2[("Accept", "1")] = "Accept"
    transitions_2[("Accept", "2")] = "Accept"

    dfa_2 = Dfa(alphabet, states_2, accepting_states_2, start_state_2, transitions_2)

    # Create abstract spec for the intersection of dfa_1 and dfa_2. Then compute its explicit form.
    abstract_intersection = dfa_1 & dfa_2

    explicit_intersection = abstract_intersection.explicit()

    assert isinstance(abstract_intersection, AbstractSpec)
    assert isinstance(explicit_intersection, Dfa)

    assert not abstract_intersection.accepts([]) and not explicit_intersection.accepts([])

    assert not abstract_intersection.accepts(list("0")) and not explicit_intersection.accepts(list("0"))
    assert not abstract_intersection.accepts(list("1")) and not explicit_intersection.accepts(list("1"))
    assert not abstract_intersection.accepts(list("2")) and not explicit_intersection.accepts(list("2"))

    assert not abstract_intersection.accepts(list("000")) and not explicit_intersection.accepts(list("000"))
    assert not abstract_intersection.accepts(list("111")) and not explicit_intersection.accepts(list("111"))
    assert not abstract_intersection.accepts(list("222")) and not explicit_intersection.accepts(list("222"))

    assert not abstract_intersection.accepts(list("010")) and not explicit_intersection.accepts(list("010"))
    assert not abstract_intersection.accepts(list("020")) and not explicit_intersection.accepts(list("020"))
    assert abstract_intersection.accepts(list("12")) and explicit_intersection.accepts(list("12"))
    assert abstract_intersection.accepts(list("012210")) and explicit_intersection.accepts(list("012210"))

def test_dfa_negation():
    """ Creates a DFA which accepts iff a string contains a "1"
    symbol. Then ensure that its symbolic and explicit negation
    have an equivalent and correct language
    """

    alphabet = {"0","1","2"}

    # Create DFA that accepts once it encounters a "1"
    states = {"Reject", "Accept"}
    accepting_states = {"Accept"}
    start_state = "Reject"

    transitions = {}

    transitions[("Reject", "0")] = "Reject"
    transitions[("Reject", "1")] = "Accept"
    transitions[("Reject", "2")] = "Reject"

    transitions[("Accept", "0")] = "Accept"
    transitions[("Accept", "1")] = "Accept"
    transitions[("Accept", "2")] = "Accept"

    dfa = Dfa(alphabet, states, accepting_states, start_state, transitions)

    # Create abstract spec for the negation of dfa and compute its explicit form.
    abstract_negation = ~dfa

    explicit_negation = abstract_negation.explicit()

    assert isinstance(abstract_negation, AbstractSpec)
    assert isinstance(explicit_negation, Dfa)

    assert abstract_negation.accepts([]) and explicit_negation.accepts([])

    assert abstract_negation.accepts(list("0")) and explicit_negation.accepts(list("0"))
    assert not abstract_negation.accepts(list("1")) and not explicit_negation.accepts(list("1"))
    assert abstract_negation.accepts(list("2")) and explicit_negation.accepts(list("2"))

    assert abstract_negation.accepts(list("000")) and explicit_negation.accepts(list("000"))
    assert not abstract_negation.accepts(list("111")) and not explicit_negation.accepts(list("111"))
    assert abstract_negation.accepts(list("222")) and explicit_negation.accepts(list("222"))

    assert not abstract_negation.accepts(list("010")) and not explicit_negation.accepts(list("010"))
    assert abstract_negation.accepts(list("020")) and explicit_negation.accepts(list("020"))
    assert not abstract_negation.accepts(list("12")) and not explicit_negation.accepts(list("12"))
    assert not abstract_negation.accepts(list("012210")) and not explicit_negation.accepts(list("012210"))

def test_dfa_difference():
    """ Creates two DFAs, one which accepts iff
    a string contains a "1" symbol and another which
    accepts iff a string contains a "2" symbol. Then ensure
    that their symbolic and explicit difference
    have an equivalent and correct language
    """

    alphabet = {"0","1","2"}

    # Create DFA that accepts once it encounters a "1"
    states_1 = {"Reject", "Accept"}
    accepting_states_1 = {"Accept"}
    start_state_1 = "Reject"

    transitions_1 = {}

    transitions_1[("Reject", "0")] = "Reject"
    transitions_1[("Reject", "1")] = "Accept"
    transitions_1[("Reject", "2")] = "Reject"

    transitions_1[("Accept", "0")] = "Accept"
    transitions_1[("Accept", "1")] = "Accept"
    transitions_1[("Accept", "2")] = "Accept"

    dfa_1 = Dfa(alphabet, states_1, accepting_states_1, start_state_1, transitions_1)

    # Create DFA that accepts once it encounters a "2"
    states_2 = {"Reject", "Accept"}
    accepting_states_2 = {"Accept"}
    start_state_2 = "Reject"

    transitions_2 = {}

    transitions_2[("Reject", "0")] = "Reject"
    transitions_2[("Reject", "1")] = "Reject"
    transitions_2[("Reject", "2")] = "Accept"

    transitions_2[("Accept", "0")] = "Accept"
    transitions_2[("Accept", "1")] = "Accept"
    transitions_2[("Accept", "2")] = "Accept"

    dfa_2 = Dfa(alphabet, states_2, accepting_states_2, start_state_2, transitions_2)

    # Create abstract spec for the difference of dfa_1 and dfa_2. Then compute its explicit form.
    abstract_difference = dfa_1 - dfa_2

    explicit_difference = abstract_difference.explicit()

    assert isinstance(abstract_difference, AbstractSpec)
    assert isinstance(explicit_difference, Dfa)

    assert not abstract_difference.accepts([]) and not explicit_difference.accepts([])

    assert not abstract_difference.accepts(list("0")) and not explicit_difference.accepts(list("0"))
    assert abstract_difference.accepts(list("1")) and explicit_difference.accepts(list("1"))
    assert not abstract_difference.accepts(list("2")) and not explicit_difference.accepts(list("2"))

    assert not abstract_difference.accepts(list("000")) and not explicit_difference.accepts(list("000"))
    assert abstract_difference.accepts(list("111")) and explicit_difference.accepts(list("111"))
    assert not abstract_difference.accepts(list("222")) and not explicit_difference.accepts(list("222"))

    assert abstract_difference.accepts(list("010")) and explicit_difference.accepts(list("010"))
    assert not abstract_difference.accepts(list("020")) and not explicit_difference.accepts(list("020"))
    assert not abstract_difference.accepts(list("12")) and not explicit_difference.accepts(list("12"))
    assert not abstract_difference.accepts(list("012210")) and not explicit_difference.accepts(list("012210"))

###################################################################################################
# Advanced Tests
###################################################################################################

###################################################################################################
# Helper Functions
###################################################################################################
