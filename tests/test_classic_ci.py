""" Tests for the ClassicCI class"""

import random
from fractions import Fraction

import pytest

from citoolkit.improvisers.improviser import InfeasibleImproviserError,\
    InfeasibleSoftConstraintError, InfeasibleRandomnessError
from citoolkit.improvisers.classic_ci import ClassicCI
from citoolkit.specifications.dfa import Dfa

from .test_dfa import generate_random_dfa

###################################################################################################
# Basic Tests
###################################################################################################

def test_classic_ci_improvise():
    """ Test a simple classic CI instance. """
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
    improviser = ClassicCI(hard_constraint, soft_constraint, length_bounds, epsilon, prob_bounds)

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

def test_classic_ci_generator():
    """ Test a simple classic CI instance."""
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
    for word in ClassicCI(hard_constraint, soft_constraint, length_bounds, epsilon, prob_bounds).generator():
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

def test_classic_ci_infeasible():
    """ Tests that an infeasible classic CI instance raises an exception"""
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
    ClassicCI(hard_constraint, soft_constraint, length_bounds, epsilon, prob_bounds)

    # Check that various parameter tweaks that render the
    # problem infeasible are identified by the improviser.
    with pytest.raises(InfeasibleSoftConstraintError):
        ClassicCI(hard_constraint, soft_constraint, length_bounds, 0, prob_bounds)

    with pytest.raises(InfeasibleSoftConstraintError):
        ClassicCI(hard_constraint, soft_constraint, (3,6), epsilon, prob_bounds)

    with pytest.raises(InfeasibleRandomnessError):
        ClassicCI(hard_constraint, soft_constraint, length_bounds, epsilon, (.25,.25))

###################################################################################################
# Randomized Tests
###################################################################################################

# Randomized tests default parameters
_RANDOM_CI_TEST_NUM_ITERS = 1000
_RANDOM_CI_TEST_NUM_SAMPLES = 25000
_MIN_WORD_LENGTH_BOUND = 0
_MAX_WORD_LENGTH_BOUND = 10

@pytest.mark.slow
def test_classic_ci_improvise_random():
    """ Tests generating a random Classic CI improviser
    instances and ensuring that they either are infeasible
    or are feasible and improvise correctly. Primarily to
    detect any crashes. Generates feasible hard constraint
    all the time, feasible epsilon at least 50% of the time,
    and feasible prob_bounds at least 50% of the time. This
    results in roughly 25%-50% of the generated improvisers being
    feasible.
    """
    for _ in range(_RANDOM_CI_TEST_NUM_ITERS):
        # Generate random set of ClassicCI parameters.
        min_length = random.randint(_MIN_WORD_LENGTH_BOUND, _MAX_WORD_LENGTH_BOUND)
        max_length = random.randint(min_length, _MAX_WORD_LENGTH_BOUND)
        length_bounds = (min_length, max_length)

        hard_constraint = generate_random_dfa(max_states=8)

        while hard_constraint.language_size(*length_bounds) == 0:
            hard_constraint = generate_random_dfa()

        soft_constraint = generate_random_dfa(max_states=8, alphabet=hard_constraint.alphabet)

        if random.random() < 0.5:
            # Pick a random epsilon.
            epsilon_feasible = False

            epsilon = random.uniform(0,1)
        else:
            # Pick a guaranteed feasible epsilon.
            epsilon_feasible = True

            a_prob = Fraction((hard_constraint & soft_constraint).language_size(*length_bounds), hard_constraint.language_size(*length_bounds))
            epsilon =  min(1, Fraction(1.1) * (1 - a_prob))

        if random.random() < 0.5:
            # Pick random min and max probability bounds.
            prob_feasible = False

            min_prob = random.uniform(0,1)
            max_prob = random.uniform(min_prob, 1)
        else:
            # Pick guaranteed feasible min and max probability bounds.
            prob_feasible = True

            min_prob = Fraction(1, hard_constraint.language_size(*length_bounds) * random.randint(1, 10))
            max_prob = min(1, Fraction(random.randint(1, 10), hard_constraint.language_size(*length_bounds)))

        prob_bounds = (min_prob, max_prob)

        # Attempt to create the improviser. If it is a feasible problem,
        # attempt to sample it and ensure that the output distribution
        # is relatively correct. Also ensure that the improviser fails only
        # if the problem is not guaranteed feasible.
        try:
            improviser = ClassicCI(hard_constraint, soft_constraint, length_bounds, epsilon, prob_bounds)
        except InfeasibleImproviserError:
            assert not (epsilon_feasible and prob_feasible)
            continue

        # Sample the improviser and ensure that the sampled distribution
        # is correct.
        improvisation_count = {}

        for _ in range(_RANDOM_CI_TEST_NUM_SAMPLES):
            word = improviser.improvise()

            if word not in improvisation_count.keys():
                improvisation_count[word] = 1
            else:
                improvisation_count[word] += 1

        # Ensures the sampled distribution is relatively correct within a tolerance.
        for word in improvisation_count:
            assert hard_constraint.accepts(word)

        a_count = 0

        for word, count in improvisation_count.items():
            if soft_constraint.accepts(word):
                a_count += count

        assert a_count/_RANDOM_CI_TEST_NUM_SAMPLES >= .99 - epsilon

        for word, count in improvisation_count.items():
            assert min_prob - 0.02 <= count/_RANDOM_CI_TEST_NUM_SAMPLES <= max_prob + 0.02
