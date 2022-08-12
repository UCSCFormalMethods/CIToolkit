""" Tests for the Dfa class"""

import math
import random
import itertools

import pytest
import pytest
from hypothesis import given, settings
from hypothesis.strategies import booleans, integers, shared

from citoolkit.specifications.spec import AbstractSpec, UniverseSpec, NullSpec
from citoolkit.specifications.dfa import Dfa, State, DfaCycleError, DfaEmptyLanguageError

from .test_utils import *

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

    # Create the DFA, which should raise an exception
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

def test_dfa_no_accepting():
    """ Creates a simple Dfa and ensures that select
    words are correctly accepted or rejected. Dfa is
    constructed with string states.
    """

    # Create a DFA that only accepts strings that contain 3 "1"
    # symbols in a row with no "2" inputs after them.
    alphabet = {"0", "1", "2"}
    states = {"0_Seen", "1_Seen", "2_Seen", "3_Seen"}
    accepting_states = set()
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

    assert not dfa.accepts(list("111"))
    assert not dfa.accepts(list("1112"))

    assert not dfa.accepts(list("000"))
    assert not dfa.accepts(list("222"))

    assert not dfa.accepts(list("01110"))

    assert not dfa.accepts(list("00000011000020011100020001"))
    assert not dfa.accepts(list("0000001100002001110002000111"))

def test_dfa_topological_ordering():
    """ Create an acyclic DFA and ensure that a correct
    topologically sorted list of states is computed.
    """
    # Create an acyclic DFA
    alphabet = {"0", "1"}
    states = {"A", "B", "C", "D", "E", "F", "Sink"}
    accepting_states = {"F"}
    start_state = "A"

    transitions = {}
    transitions[("A","0")] = "B"
    transitions[("A","1")] = "C"
    transitions[("B","0")] = "C"
    transitions[("B","1")] = "C"
    transitions[("C","0")] = "D"
    transitions[("C","1")] = "E"
    transitions[("D","0")] = "E"
    transitions[("D","1")] = "F"
    transitions[("E","0")] = "F"
    transitions[("E","1")] = "F"
    transitions[("F","0")] = "Sink"
    transitions[("F","1")] = "Sink"
    transitions[("Sink", "0")] = "Sink"
    transitions[("Sink", "1")] = "Sink"

    dfa = Dfa(alphabet, states, accepting_states, start_state, transitions)

    # Ensures that the one correct topological sort is generated.
    assert dfa.states_topological() == ["A", "B", "C", "D", "E", "F"]

def test_dfa_topological_ordering_cycle():
    """ Create a simple DFA with a reachable and accepting cycle
    and ensure that a ValueError is raised.
    """
    # Create a cyclic DFA
    alphabet = {"0", "1"}
    states = {"A", "B", "C", "D", "Sink"}
    accepting_states = {"D"}
    start_state = "A"

    transitions = {}
    transitions[("A","0")] = "B"
    transitions[("A","1")] = "C"
    transitions[("B","0")] = "D"
    transitions[("B","1")] = "Sink"
    transitions[("C","0")] = "B"
    transitions[("C","1")] = "Sink"
    transitions[("D","0")] = "C"
    transitions[("D","1")] = "Sink"
    transitions[("Sink", "0")] = "Sink"
    transitions[("Sink", "1")] = "Sink"

    dfa = Dfa(alphabet, states, accepting_states, start_state, transitions)

    # Ensures that a ValueError is rasied as a cyclical DFA does not
    # have a well defined topological odering.
    with pytest.raises(DfaCycleError):
        dfa.states_topological()

def test_dfa_language_size():
    """ Creates a Dfa that accepts only words of length
    7 and ensures that language_size returns the
    correct result.
    """
    dfa = Dfa.exact_length_dfa({"0","1"}, 7)

    assert dfa.language_size() == 2**7

def test_dfa_language_size_abstract():
    """ Creates an abstract specification that is
    the union of two exact length Dfas and ensures
    that language_size returns the correct result.
    """
    dfa_1 = Dfa.exact_length_dfa({"0","1"}, 5)
    dfa_2 = Dfa.exact_length_dfa({"0","1"}, 7)

    abstract_dfa = dfa_1 | dfa_2

    assert abstract_dfa.language_size() == (2**5 + 2**7)

def test_dfa_language_size_param():
    """ Creates a Dfa that accepts only words of length
    7 and ensures that language_size returns the
    correct result.
    """
    dfa = Dfa.max_length_dfa({"0","1"}, 7)

    assert dfa.language_size(min_length=5, max_length=7) == 2**5 + 2**6 + 2**7

def test_dfa_sample():
    """ Create a simple Dfa that when uniformly sampled
    should generate the following words with relatively
    uniform probabilities: [[], ["A"], ["A", "A"], ["B"]].
    Then verify that the sampling is over the correct
    words and reasonably accurate.
    """
    # Create test Dfa
    alphabet = {"A", "B"}
    states = {"Start", "Top", "Bottom1", "Bottom2", "Sink"}
    accepting_states = {"Start", "Top", "Bottom1", "Bottom2"}
    start_state = "Start"

    transitions = {}

    transitions[("Start", "A")] = "Bottom1"
    transitions[("Start", "B")] = "Top"
    transitions[("Top", "A")] = "Sink"
    transitions[("Top", "B")] = "Sink"
    transitions[("Bottom1", "A")] = "Bottom2"
    transitions[("Bottom1", "B")] = "Sink"
    transitions[("Bottom2", "A")] = "Sink"
    transitions[("Bottom2", "B")] = "Sink"
    transitions[("Sink", "A")] = "Sink"
    transitions[("Sink", "B")] = "Sink"

    dfa = Dfa(alphabet, states, accepting_states, start_state, transitions)

    # Sample 100,000 words and keep track of
    # how many of each are sampled.
    dfa_language = [tuple(), tuple("A"), ("A", "A"), tuple("B")]

    sample_counts = {}

    for word in dfa_language:
        sample_counts[word] = 0

    for _ in range(100000):
        # Sample a word from our Dfa's language
        sampled_word = dfa.sample()

        # Ensure we didn't sample a word not in our language
        assert sampled_word in dfa_language

        # Increment the count for the word we sampled
        sample_counts[tuple(sampled_word)] += 1

    # Assert the sampled ratios are relatively correct
    for word in dfa_language:
        word_prob = sample_counts[word]/100000

        assert word_prob > .24
        assert word_prob < .26

def test_dfa_sample_abstract():
    """ Create a simple Dfa that when uniformly sampled
    should generate the following words with relatively
    uniform probabilities: [[], ["A"], ["A", "A"], ["B"]].
    Then verify that sampling is performed over the
    correct words and reasonably accurate.
    """
    # Create test Dfa
    alphabet = {"A", "B"}
    states = {"Start", "Top", "Bottom1", "Bottom2", "Sink"}
    accepting_states = {"Start", "Top", "Bottom1", "Bottom2"}
    start_state = "Start"

    transitions = {}

    transitions[("Start", "A")] = "Bottom1"
    transitions[("Start", "B")] = "Top"
    transitions[("Top", "A")] = "Sink"
    transitions[("Top", "B")] = "Sink"
    transitions[("Bottom1", "A")] = "Bottom2"
    transitions[("Bottom1", "B")] = "Sink"
    transitions[("Bottom2", "A")] = "Sink"
    transitions[("Bottom2", "B")] = "Sink"
    transitions[("Sink", "A")] = "Sink"
    transitions[("Sink", "B")] = "Sink"

    main_dfa = Dfa(alphabet, states, accepting_states, start_state, transitions)
    length_dfa = Dfa.exact_length_dfa(alphabet, 1)

    dfa = main_dfa & length_dfa

    # Sample 100,000 words and keep track of
    # how many of each are sampled.
    dfa_language = [tuple("A"), tuple("B")]

    sample_counts = {}

    for word in dfa_language:
        sample_counts[word] = 0

    for _ in range(100000):
        # Sample a word from our Dfa's language
        sampled_word = dfa.sample()

        # Ensure we didn't sample a word not in our language
        assert sampled_word in dfa_language

        # Increment the count for the word we sampled
        sample_counts[tuple(sampled_word)] += 1

    # Assert the sampled ratios are relatively correct
    for word in dfa_language:
        word_prob = sample_counts[word]/100000

        assert word_prob > .49
        assert word_prob < .51

def test_dfa_sample_param():
    """ Create a simple Dfa that when uniformly sampled
    with length parameters should generate the following
    words with relatively uniform probabilities:
    [[], ["A"], ["A", "A"], ["B"]]. Then verify that
    sampling is performed over the correct words and is
    reasonably accurate.
    """
    # Create test Dfa
    alphabet = {"A", "B"}
    states = {"Start", "Top", "Bottom1", "Bottom2", "Bottom3", "Bottom4", "Sink"}
    accepting_states = {"Start", "Top", "Bottom1", "Bottom2", "Bottom3", "Bottom4"}
    start_state = "Start"

    transitions = {}

    transitions[("Start", "A")] = "Bottom1"
    transitions[("Start", "B")] = "Top"
    transitions[("Top", "A")] = "Sink"
    transitions[("Top", "B")] = "Sink"
    transitions[("Bottom1", "A")] = "Bottom2"
    transitions[("Bottom1", "B")] = "Sink"
    transitions[("Bottom2", "A")] = "Bottom3"
    transitions[("Bottom2", "B")] = "Sink"
    transitions[("Bottom3", "A")] = "Bottom4"
    transitions[("Bottom3", "B")] = "Sink"
    transitions[("Bottom4", "A")] = "Sink"
    transitions[("Bottom4", "B")] = "Sink"
    transitions[("Sink", "A")] = "Sink"
    transitions[("Sink", "B")] = "Sink"

    dfa = Dfa(alphabet, states, accepting_states, start_state, transitions)

    # Sample 100,000 words and keep track of
    # how many of each are sampled.
    dfa_language = [tuple("A"), tuple("B"), ("A", "A"), ("A", "A", "A")]

    sample_counts = {}

    for word in dfa_language:
        sample_counts[word] = 0

    for _ in range(100000):
        # Sample a word from our Dfa's language
        sampled_word = dfa.sample(min_length=1, max_length=3)

        # Ensure we didn't sample a word not in our language
        assert sampled_word in dfa_language

        # Increment the count for the word we sampled
        sample_counts[tuple(sampled_word)] += 1

    # Assert the sampled ratios are relatively correct
    for word in dfa_language:
        word_prob = sample_counts[word]/100000

        assert word_prob > .24
        assert word_prob < .26

def test_dfa_sample_empty():
    """ Tests that attempting to sample from a DFA who's
    language is empty raises a value exception.
    """
    # Creates a Dfa with an empty language.
    dfa = Dfa.exact_length_dfa({"0","1"}, 1) & Dfa.exact_length_dfa({"0", "1"}, 2)

    with pytest.raises(DfaEmptyLanguageError):
        dfa.sample()

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

def test_dfa_abstract_universe_null():
    """ Tests that combining a DFA with UniverseSpecs/NullSpecs
    works as intended.
    """

    # Create a simple DFA
    alphabet = {"0", "1", "2"}
    states = {"0_Seen", "1_Seen", "2_Seen", "3_Seen"}
    accepting_states = {"3_Seen"}
    start_state = "0_Seen"

    transitions = {}
    for state in states:
        for symbol in alphabet:
            transitions[(state, symbol)] = "0_Seen"

    transitions[("0_Seen", "1")] = "1_Seen"
    transitions[("1_Seen", "1")] = "2_Seen"
    transitions[("2_Seen", "1")] = "3_Seen"

    transitions[("3_Seen", "0")] = "3_Seen"
    transitions[("3_Seen", "1")] = "3_Seen"

    dfa = Dfa(alphabet, states, accepting_states, start_state, transitions)

    # Check operations with UniverseSpec/NullSpec
    assert (UniverseSpec() | dfa).explicit() == UniverseSpec()
    assert (dfa | UniverseSpec()).explicit() == UniverseSpec()
    assert (UniverseSpec() & dfa).explicit() == dfa
    assert (dfa & UniverseSpec()).explicit() == dfa

    assert (NullSpec() | dfa).explicit() == dfa
    assert (dfa | NullSpec()).explicit() == dfa
    assert (NullSpec() & dfa).explicit() == NullSpec()
    assert (dfa & NullSpec()).explicit() == NullSpec()

def test_dfa_exact_length_constructor():
    """ Tests that the Dfa returned by the exact_length_dfa
    constructor works as expected.
    """
    dfa = Dfa.exact_length_dfa({"0","1"}, 7)

    assert not dfa.accepts("")
    assert not dfa.accepts("0")
    assert not dfa.accepts("1")
    assert not dfa.accepts("01")
    assert not dfa.accepts("011")
    assert not dfa.accepts("0110")
    assert not dfa.accepts("01101")
    assert not dfa.accepts("011010")
    assert dfa.accepts("0110100")
    assert not dfa.accepts("01101000")
    assert not dfa.accepts("000000001111000000001100001000111100110110110")

def test_dfa_min_length_constructor():
    """ Tests that the Dfa returned by the min_length_dfa
    constructor works as expected.
    """
    dfa = Dfa.min_length_dfa({"0", "1"}, 7)

    assert not dfa.accepts("")
    assert not dfa.accepts("0")
    assert not dfa.accepts("1")
    assert not dfa.accepts("01")
    assert not dfa.accepts("011")
    assert not dfa.accepts("0110")
    assert not dfa.accepts("01101")
    assert not dfa.accepts("011010")
    assert dfa.accepts("0110100")
    assert dfa.accepts("01101000")
    assert dfa.accepts("000000001111000000001100001000111100110110110")

def test_dfa_max_length_constructor():
    """ Tests that the Dfa returned by the max_length_dfa
    constructor works as expected.
    """
    dfa = Dfa.max_length_dfa({"0", "1"}, 7)

    assert dfa.accepts("")
    assert dfa.accepts("0")
    assert dfa.accepts("1")
    assert dfa.accepts("01")
    assert dfa.accepts("011")
    assert dfa.accepts("0110")
    assert dfa.accepts("01101")
    assert dfa.accepts("011010")
    assert dfa.accepts("0110100")
    assert not dfa.accepts("01101000")
    assert not dfa.accepts("000000001111000000001100001000111100110110110")

@pytest.mark.advanced
def test_dfa_massive_1():
    """ Tests that a massive union of relatively
    equally sized Dfas can be simplified correctly.
    """
    alphabet = {"0","1"}

    union_dfa = Dfa.min_length_dfa(alphabet, 1000) | Dfa.max_length_dfa(alphabet, 1000)
    explicit_dfa = union_dfa.explicit()

    assert len(explicit_dfa.states) == 1

@pytest.mark.advanced
def test_dfa_massive_2():
    """ Tests that a massive union of a very small and
    very large Dfa can be simplified correctly.
    """
    alphabet = {"0","1"}

    union_dfa = Dfa.min_length_dfa(alphabet, 1) | Dfa.max_length_dfa(alphabet, 1000000)
    explicit_dfa = union_dfa.explicit()

    assert len(explicit_dfa.states) == 1

###################################################################################################
# Randomized Tests
###################################################################################################

@given(orig_dfa=random_dfa(num_states=integers(1,8)))
@settings(deadline=None, max_examples=1000)
@pytest.mark.advanced
def test_dfa_minimize_random(orig_dfa):
    """ Generates a random DFA and minimizes the dfa. Then ensures
    that the minimizes version and the complete version either
    accept or reject all strings of length less than or equal to
    the number of states.
    """
    # Calculate DFA's minimized form.
    min_dfa = orig_dfa.minimize()

    # Check that construction is valid
    assert isinstance(orig_dfa, Dfa)
    assert isinstance(min_dfa, Dfa)
    assert len(min_dfa.states) <= len(orig_dfa.states)

    # Iterate through every possible word that has length <= the number
    # of states in the original DFAs to ensure that the specs are equivalent.
    for word_length in range(len(orig_dfa.states)+1):
        for word in itertools.product(orig_dfa.alphabet, repeat=word_length):
            assert orig_dfa.accepts(word) == min_dfa.accepts(word)

@given(dfa_1=random_dfa(num_states=integers(1,3), num_symbols=shared(integers(1,3), key="foo")),
       dfa_2=random_dfa(num_states=integers(1,3), num_symbols=shared(integers(1,3), key="foo")))
@settings(deadline=None, max_examples=1000)
@pytest.mark.advanced
def test_dfa_union_random(dfa_1, dfa_2):
    """ Generates two random DFAs and takes the
    logical and explicit union of the 2 DFAs. Then ensures that they are consistent
    on all strings of length less than or equal to the number of states.
    """
    abstract_dfa = dfa_1 | dfa_2
    explicit_dfa = abstract_dfa.explicit()

    # Check that construction is valid
    assert isinstance(abstract_dfa, AbstractSpec)
    assert isinstance(explicit_dfa, Dfa)

    # Iterate through every possible word that has length <= the number
    # of states in the new Dfa to ensure they are equivalent.
    for word_length in range(len(explicit_dfa.states)+1):
        for word in itertools.product(explicit_dfa.alphabet, repeat=word_length):
            assert abstract_dfa.accepts(word) == explicit_dfa.accepts(word)

@given(dfa_1=random_dfa(num_states=integers(1,3), num_symbols=shared(integers(1,3), key="foo")),
       dfa_2=random_dfa(num_states=integers(1,3), num_symbols=shared(integers(1,3), key="foo")))
@settings(deadline=None, max_examples=1000)
@pytest.mark.advanced
def test_dfa_intersection_random(dfa_1, dfa_2):
    """ Generates two random DFAs and takes the
    logical and explicit intersection of the 2 DFAs. Then ensures that they are consistent
    on all strings of length less than or equal to the number of states.
    """
    abstract_dfa = dfa_1 & dfa_2
    explicit_dfa = abstract_dfa.explicit()

    # Check that construction is valid
    assert isinstance(abstract_dfa, AbstractSpec)
    assert isinstance(explicit_dfa, Dfa)

    # Iterate through every possible word that has length <= the number
    # of states in the new Dfa to ensure they are equivalent.
    for word_length in range(len(explicit_dfa.states)+1):
        for word in itertools.product(explicit_dfa.alphabet, repeat=word_length):
            assert abstract_dfa.accepts(word) == explicit_dfa.accepts(word)

@given(dfa=random_dfa(num_states=integers(1,8)))
@settings(deadline=None, max_examples=1000)
@pytest.mark.advanced
def test_dfa_negation_random(dfa):
    """ Generates a random DFA and takes the
    logical and explicit negation of it. Then ensures that they are consistent
    on all strings of length less than or equal to the number of states.
    """
    abstract_dfa = ~dfa
    explicit_dfa = abstract_dfa.explicit()

    # Check that construction is valid
    assert isinstance(abstract_dfa, AbstractSpec)
    assert isinstance(explicit_dfa, Dfa)

    # Iterate through every possible word that has length <= the number
    # of states in the new DFA to ensure that the specs are equivalent.
    for word_length in range(len(explicit_dfa.states)+1):
        for word in itertools.product(explicit_dfa.alphabet, repeat=word_length):
            assert abstract_dfa.accepts(word) == explicit_dfa.accepts(word)

@given(base_dfa=random_dfa(num_states=integers(1,8)),
       max_length=integers(1,8))
@settings(deadline=None, max_examples=1000)
@pytest.mark.advanced
def test_dfa_language_size_random(base_dfa, max_length):
    """ Generates a random DFA and counts its language size. Then ensures that this is consistent 
    with the count of all strings of length less than or equal to the number of states accepted by the DFA.
    """
    # Pick a length and create a random dfa that acceps words up to that length
    length_limit_dfa = Dfa.max_length_dfa(base_dfa.alphabet, max_length)

    dfa = base_dfa & length_limit_dfa
    explicit_dfa = dfa.explicit()

    # Enumerate all words to ensure that the calculated language size is correct.
    enumerated_count = 0

    for word_length in range(max_length+1):
        for word in itertools.product(base_dfa.alphabet, repeat=word_length):
            if explicit_dfa.accepts(word):
                enumerated_count += 1

    assert explicit_dfa.language_size() == enumerated_count
