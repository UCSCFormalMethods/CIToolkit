""" Tests for the CI class"""

import pytest

from citoolkit.improvisers.improviser import InfeasibleSoftConstraintError, InfeasibleRandomnessError
from citoolkit.improvisers.ci import CI
from citoolkit.specifications.dfa import Dfa

###################################################################################################
# Basic Tests
###################################################################################################

def test_ci_improvise():
    """ Test a simple CI instance. """
    # Create a hard constraint Dfa that accepts all words of the form "{01}*"
    alphabet = {"0", "1"}
    states = {"Start", "State0", "State1", "Sink"}
    accepting_states = {"State1"}
    start_state = "Start"

    transitions = {}
    transitions[("Start", "0")] = "State0"
    transitions[("Start", "1")] = "Sink"
    transitions[("State0", "0")] = "Sink"
    transitions[("State0", "1")] = "State1"
    transitions[("State1", "0")] = "State0"
    transitions[("State1", "1")] = "Sink"
    transitions[("Sink", "0")] = "Sink"
    transitions[("Sink", "1")] = "Sink"

    hard_constraint = Dfa(alphabet, states, accepting_states, start_state, transitions)

    # Create a soft constraint that accepts only words of length 2
    soft_constraint = Dfa.exact_length_dfa(alphabet, 2)

    # Fix length and probability bounds
    length_bounds = (0,6)
    prob_bounds = (0,.75)
    epsilon = 0.5

    # Create improviser and map to count all words
    improviser = CI(hard_constraint, soft_constraint, length_bounds)
    improviser.parameterize(epsilon, prob_bounds)

    improvisations = {tuple("01"), tuple("0101"), tuple("010101")}
    improvisation_count = {improvisation:0 for improvisation in improvisations}

    # Sample a collection of words from the improviser.
    for _ in range(100000):
        word = improviser.improvise()

        assert word in improvisations

        improvisation_count[word] += 1

    # Check counts for improvisations
    assert 0.74  < improvisation_count[tuple("01")]/100000 < 0.76
    assert 0.115 < improvisation_count[tuple("0101")]/100000 < 0.135
    assert 0.115 < improvisation_count[tuple("010101")]/100000 < 0.135

def test_ci_generator():
    """ Test a simple CI instance."""
    # Create a hard constraint Dfa that accepts all words of the form "{01}*"
    alphabet = {"0", "1"}
    states = {"Start", "State0", "State1", "Sink"}
    accepting_states = {"State1"}
    start_state = "Start"

    transitions = {}
    transitions[("Start", "0")] = "State0"
    transitions[("Start", "1")] = "Sink"
    transitions[("State0", "0")] = "Sink"
    transitions[("State0", "1")] = "State1"
    transitions[("State1", "0")] = "State0"
    transitions[("State1", "1")] = "Sink"
    transitions[("Sink", "0")] = "Sink"
    transitions[("Sink", "1")] = "Sink"

    hard_constraint = Dfa(alphabet, states, accepting_states, start_state, transitions)

    # Create a soft constraint that accepts only words of length 2
    soft_constraint = Dfa.exact_length_dfa(alphabet, 2)

    # Fix length and probability bounds
    length_bounds = (0,6)
    prob_bounds = (0,.75)
    epsilon = 0.5

    # Create map to count all words
    improvisations = [tuple("01"), tuple("0101"), tuple("010101")]
    improvisation_count = {}

    for improvisation in improvisations:
        improvisation_count[improvisation] = 0

    words_improvised = 0

    # Create an improviser and samples a collection of words from it as a generator.
    improviser = CI(hard_constraint, soft_constraint, length_bounds)
    improviser.parameterize(epsilon, prob_bounds)

    for word in improviser.generator():
        assert word in improvisations

        improvisation_count[word] += 1

        # Check if we have improvised enough words
        words_improvised += 1
        if words_improvised >= 100000:
            break

    # Check counts for improvisations
    assert 0.74  < improvisation_count[tuple("01")]/100000 < 0.76
    assert 0.115 < improvisation_count[tuple("0101")]/100000 < 0.135
    assert 0.115 < improvisation_count[tuple("010101")]/100000 < 0.135

def test_ci_infeasible():
    """ Tests that an infeasible CI instance raises an exception"""
    # Create a hard constraint Dfa that accepts all words of the form "{01}*"
    alphabet = {"0", "1"}
    states = {"Start", "State0", "State1", "Sink"}
    accepting_states = {"State1"}
    start_state = "Start"

    transitions = {}
    transitions[("Start", "0")] = "State0"
    transitions[("Start", "1")] = "Sink"
    transitions[("State0", "0")] = "Sink"
    transitions[("State0", "1")] = "State1"
    transitions[("State1", "0")] = "State0"
    transitions[("State1", "1")] = "Sink"
    transitions[("Sink", "0")] = "Sink"
    transitions[("Sink", "1")] = "Sink"

    hard_constraint = Dfa(alphabet, states, accepting_states, start_state, transitions)

    # Create a soft constraint that accepts only words of length 2
    soft_constraint = Dfa.exact_length_dfa(alphabet, 2)

    # Fix length and probability bounds
    length_bounds = (0,6)
    prob_bounds = (0,.75)
    epsilon = 0.5

    # Ensure that the base LCI problem is feasible
    improviser = CI(hard_constraint, soft_constraint, length_bounds)
    improviser.parameterize(epsilon, prob_bounds)

    # Check that various parameter tweaks that render the
    # problem infeasible are identified by the improviser.
    with pytest.raises(InfeasibleSoftConstraintError):
        improviser = CI(hard_constraint, soft_constraint, length_bounds)
        improviser.parameterize(0, prob_bounds)

    with pytest.raises(InfeasibleSoftConstraintError):
        improviser = CI(hard_constraint, soft_constraint, (3,6))
        improviser.parameterize(epsilon, prob_bounds)

    with pytest.raises(InfeasibleRandomnessError):
        improviser = CI(hard_constraint, soft_constraint, length_bounds)
        improviser.parameterize(epsilon, (0.25, 0.25))
