from hypothesis import given, settings
from hypothesis.strategies import booleans, integers, fractions, permutations, sampled_from, composite

from citoolkit.specifications.dfa import Dfa
from citoolkit.costfunctions.static_cost_dfa import StaticCostDfa
from citoolkit.costfunctions.accumulated_cost_dfa import AccumulatedCostDfa
from citoolkit.labellingfunctions.labelling_dfa import LabellingDfa

###################################################################################################
# Helper Functions
###################################################################################################

@composite
def random_dfa(draw, num_states=integers(1,6), num_symbols=integers(1,3)):
    """ Generates a random Dfa object."""
    # Create states and alphabet
    states = ["State_" + str(state_iter) for state_iter in range(draw(num_states))]
    alphabet = [str(symbol) for symbol in range(draw(num_symbols))]

    # Select random accepting states
    num_accepting_states = draw(integers(1, len(states)))
    accepting_states = draw(permutations(states).map(lambda x: x[:num_accepting_states]))

    # Picks a random start state
    start_state = draw(sampled_from(states))

    # Randomly generates transitions
    transitions = {}
    for symbol in alphabet:
        for state in states:
            transitions[(state, symbol)] = draw(sampled_from(states))

    # Create and return Dfa
    return Dfa(alphabet, states, accepting_states, start_state, transitions)

@composite
def random_label_dfa(draw, num_states=integers(1,6), num_symbols=integers(1,3)):
    """ Generates a random label DFA object."""
    dfa = draw(random_dfa(num_states=num_states, num_symbols=num_symbols))

    # Generate random labels
    num_labels = draw(integers(1, len(dfa.accepting_states)))
    labels = ["Label_" + str(label_iter) for label_iter in range(num_labels)]

    # Randomly assign labels to accepting states
    label_map = {}
    for state in sorted(dfa.accepting_states, key=lambda x: str(x)):
        label_map[state] = draw(sampled_from(labels))

    return LabellingDfa(dfa, label_map)

@composite
def random_label_dfa_bounds(draw, num_states=integers(1,6), num_symbols=integers(1,3)):
    """ Generates a random label DFA and word probability bounds for each label."""
    label_dfa = draw(random_label_dfa(num_states, num_symbols))

    # Generate random probability bounds for each label
    word_prob_bounds = {}
    for label in label_dfa.labels:
        lower_bound = draw(fractions(0,1))
        upper_bound = draw(fractions(lower_bound,1))
        word_prob_bounds[label] = (lower_bound, upper_bound)

    return (label_dfa, word_prob_bounds)

@composite
def random_static_cost_dfa(draw, num_states=integers(1,6), num_symbols=integers(1,3), max_cost=100):
    """ Generates a random cost DFA object,"""
    dfa = draw(random_dfa(num_states=num_states, num_symbols=num_symbols))

    # Randomly assign costs to accepting states
    cost_map = {}
    for state in sorted(dfa.accepting_states, key=lambda x: str(x)):
        cost_map[state] = draw(fractions(0, max_cost))

    return StaticCostDfa(dfa, cost_map)

@composite
def random_accumulated_cost_dfa(draw, num_states=integers(1,6), num_symbols=integers(1,3),
                                max_word_length=integers(1,9), max_cost=5):
    """ Generates a random cost DFA object,"""
    dfa = draw(random_dfa(num_states=num_states, num_symbols=num_symbols))

    # Randomly assign costs to accepting states
    cost_map = {}
    for state in sorted(dfa.states, key=lambda x: str(x)):
        cost_map[state] = draw(fractions(0, max_cost))

    return AccumulatedCostDfa(dfa, cost_map, draw(max_word_length))
