""" Tests for the LabelledCI class"""

import random

import pytest

from citoolkit.improvisers.improviser import InfeasibleImproviserError
from citoolkit.improvisers.labelled_ci import LabelledCI, MaxEntropyLabelledCI
from citoolkit.specifications.dfa import Dfa
from citoolkit.labellingfunctions.labelling_dfa import LabellingDfa

from .test_dfa import generate_random_dfa

###################################################################################################
# Basic Tests
###################################################################################################

def test_labelled_ci_improvise():
    """ Test a simple Labelled CI instance. """
    # Create a hard constraint Dfa that accepts all words that end with "01"
    alphabet = {"0", "1"}
    h_states = {"Start", "Seen0", "Accept"}
    h_accepting_states = {"Accept"}
    h_start_state = "Start"

    h_transitions = {}
    h_transitions[("Start", "0")] = "Seen0"
    h_transitions[("Start", "1")] = "Start"
    h_transitions[("Seen0", "0")] = "Seen0"
    h_transitions[("Seen0", "1")] = "Accept"
    h_transitions[("Accept", "0")] = "Seen0"
    h_transitions[("Accept", "1")] = "Start"

    hard_constraint = Dfa(alphabet, h_states, h_accepting_states, h_start_state, h_transitions)

    # Create a soft constraint that accepts all words of length 4
    soft_constraint = Dfa.exact_length_dfa(alphabet, 4)

    # Create a labelling function that labels a word depending on how many "1"
    # symbols it has
    lf_states = {"Seen0", "Seen1", "Seen2", "Seen3", "Sink"}
    lf_accepting_states = {"Seen1", "Seen2", "Seen3"}
    lf_start_state = "Seen0"

    lf_transitions = {}
    lf_transitions[("Seen0", "0")] = "Seen0"
    lf_transitions[("Seen0", "1")] = "Seen1"
    lf_transitions[("Seen1", "0")] = "Seen1"
    lf_transitions[("Seen1", "1")] = "Seen2"
    lf_transitions[("Seen2", "0")] = "Seen2"
    lf_transitions[("Seen2", "1")] = "Seen3"
    lf_transitions[("Seen3", "0")] = "Seen3"
    lf_transitions[("Seen3", "1")] = "Sink"
    lf_transitions[("Sink", "0")] = "Sink"
    lf_transitions[("Sink", "1")] = "Sink"

    lf_dfa = Dfa(alphabet, lf_states, lf_accepting_states, lf_start_state, lf_transitions)

    label_map = {}
    label_map["Seen1"] = "Label1"
    label_map["Seen2"] = "Label2"
    label_map["Seen3"] = "Label3"

    label_func = LabellingDfa(lf_dfa, label_map)

    # Layout remaining Labelled CI parameters.
    length_bounds = (1,4)
    epsilon = 0.2
    label_prob_bounds = (.25, .4)

    word_prob_bounds = {}
    word_prob_bounds["Label1"] = (.2,.8)
    word_prob_bounds["Label2"] = (.2,.8)
    word_prob_bounds["Label3"] = (0,1)

    # Create Labelled CI Improviser
    improviser = LabelledCI(hard_constraint, soft_constraint, label_func, length_bounds, epsilon, label_prob_bounds, word_prob_bounds)

    # Check that the calculated probabilities are valid
    assert sum(improviser.sorted_labels_weights) == pytest.approx(1)

    for label_weight in improviser.sorted_labels_weights:
        assert label_prob_bounds[0] <= label_weight <= label_prob_bounds[1]

    for label in label_func.labels:
        if improviser.i_specs[label].language_size(*length_bounds) == 0:
            assert improviser.i_probs[label] == 0
        else:
            assert word_prob_bounds[label][0] <= improviser.i_probs[label] / improviser.i_specs[label].language_size(*length_bounds) <= word_prob_bounds[label][1]

        if improviser.a_specs[label].language_size(*length_bounds) == 0:
            assert improviser.a_probs[label] == 0
        else:
            assert word_prob_bounds[label][0] <= improviser.a_probs[label] / improviser.a_specs[label].language_size(*length_bounds) <= word_prob_bounds[label][1]

        assert improviser.i_probs[label] + improviser.a_probs[label] == pytest.approx(1)

    # Create sampling testing variables
    improvisations = {tuple("01"), tuple("001"), tuple("101"), tuple("0001"), tuple("0101"), tuple("1001"), tuple("1101")}
    improvisation_count = {improvisation:0 for improvisation in improvisations}

    soft_constraint_improvisations = {tuple("0001"), tuple("0101"), tuple("1001"), tuple("1101")}
    soft_constraint_count = 0

    label_count = {label:0 for label in ["Label1", "Label2", "Label3"]}

    label_improvisation_map = {}
    label_improvisation_map["Label1"] = {tuple("01"), tuple("001"), tuple("0001")}
    label_improvisation_map["Label2"] = {tuple("101"), tuple("0101"), tuple("1001")}
    label_improvisation_map["Label3"] = {tuple("1101")}

    improvisation_label_map = {}
    for label, items in label_improvisation_map.items():
        for item in items:
            improvisation_label_map[item] = label

    # Sample a collection of words from the improviser.
    for _ in range(100000):
        word = improviser.improvise()

        assert word in improvisations

        improvisation_count[word] += 1
        label_count[improvisation_label_map[word]] += 1

        if word in soft_constraint_improvisations:
            soft_constraint_count+=1

    # Check that sampled word probabilities are valid
    for label in label_func.labels:
        label_sampled_prob = label_count[label]/100000
        assert label_prob_bounds[0]-.01 <= label_sampled_prob <= label_prob_bounds[1]+.01

        for word in label_improvisation_map[label]:
            cond_word_sampled_prob = (improvisation_count[word]/100000)/label_sampled_prob
            assert word_prob_bounds[label][0]-0.1 <= cond_word_sampled_prob <= word_prob_bounds[label][1]+0.1

    assert soft_constraint_count/100000 >= .99 - epsilon

def test_labelled_ci_infeasible():
    """ Test that different infeasible Labelled CI
    problems correctly raise an exception.
    """
    # Create a hard constraint Dfa that accepts all words that end with "01"
    alphabet = {"0", "1"}
    h_states = {"Start", "Seen0", "Accept"}
    h_accepting_states = {"Accept"}
    h_start_state = "Start"

    h_transitions = {}
    h_transitions[("Start", "0")] = "Seen0"
    h_transitions[("Start", "1")] = "Start"
    h_transitions[("Seen0", "0")] = "Seen0"
    h_transitions[("Seen0", "1")] = "Accept"
    h_transitions[("Accept", "0")] = "Seen0"
    h_transitions[("Accept", "1")] = "Start"

    hard_constraint = Dfa(alphabet, h_states, h_accepting_states, h_start_state, h_transitions)

    # Create a soft constraint that accepts all words of length 4
    soft_constraint = Dfa.exact_length_dfa(alphabet, 4)

    # Create a labelling function that labels a word depending on how many "1"
    # symbols it has
    lf_states = {"Seen0", "Seen1", "Seen2", "Seen3", "Sink"}
    lf_accepting_states = {"Seen1", "Seen2", "Seen3"}
    lf_start_state = "Seen0"

    lf_transitions = {}
    lf_transitions[("Seen0", "0")] = "Seen0"
    lf_transitions[("Seen0", "1")] = "Seen1"
    lf_transitions[("Seen1", "0")] = "Seen1"
    lf_transitions[("Seen1", "1")] = "Seen2"
    lf_transitions[("Seen2", "0")] = "Seen2"
    lf_transitions[("Seen2", "1")] = "Seen3"
    lf_transitions[("Seen3", "0")] = "Seen3"
    lf_transitions[("Seen3", "1")] = "Sink"
    lf_transitions[("Sink", "0")] = "Sink"
    lf_transitions[("Sink", "1")] = "Sink"

    lf_dfa = Dfa(alphabet, lf_states, lf_accepting_states, lf_start_state, lf_transitions)

    label_map = {}
    label_map["Seen1"] = "Label1"
    label_map["Seen2"] = "Label2"
    label_map["Seen3"] = "Label3"

    label_func = LabellingDfa(lf_dfa, label_map)

    # Layout remaining Labelled CI parameters.
    length_bounds = (1,4)
    epsilon = 0.2
    label_prob_bounds = (.25, .4)

    word_prob_bounds = {}
    word_prob_bounds["Label1"] = (.2,.8)
    word_prob_bounds["Label2"] = (.2,.8)
    word_prob_bounds["Label3"] = (0,1)

    # Ensure that the base LCI problem is feasible
    LabelledCI(hard_constraint, soft_constraint, label_func, length_bounds, epsilon, label_prob_bounds, word_prob_bounds)

    # Check that various parameter tweaks that render the
    # problem infeasible are identified by the improviser.
    with pytest.raises(InfeasibleImproviserError):
        LabelledCI(hard_constraint, soft_constraint, label_func, length_bounds, 0.16, label_prob_bounds, word_prob_bounds)

    with pytest.raises(InfeasibleImproviserError):
        LabelledCI(hard_constraint, soft_constraint, label_func, length_bounds, epsilon, (0.33,0.33), word_prob_bounds)

    with pytest.raises(InfeasibleImproviserError):
        LabelledCI(hard_constraint, soft_constraint, label_func, length_bounds, epsilon, label_prob_bounds, {label:(.2,.8) for label in label_func.labels})

def test_max_entropy_labelled_ci_improvise():
    """ Test a simple Labelled CI instance. """
    # Create a hard constraint Dfa that accepts all words that end with "01"
    alphabet = {"0", "1"}
    h_states = {"Start", "Seen0", "Accept"}
    h_accepting_states = {"Accept"}
    h_start_state = "Start"

    h_transitions = {}
    h_transitions[("Start", "0")] = "Seen0"
    h_transitions[("Start", "1")] = "Start"
    h_transitions[("Seen0", "0")] = "Seen0"
    h_transitions[("Seen0", "1")] = "Accept"
    h_transitions[("Accept", "0")] = "Seen0"
    h_transitions[("Accept", "1")] = "Start"

    hard_constraint = Dfa(alphabet, h_states, h_accepting_states, h_start_state, h_transitions)

    # Create a soft constraint that accepts all words of length 4
    soft_constraint = Dfa.exact_length_dfa(alphabet, 4)

    # Create a labelling function that labels a word depending on how many "1"
    # symbols it has
    lf_states = {"Seen0", "Seen1", "Seen2", "Seen3", "Sink"}
    lf_accepting_states = {"Seen1", "Seen2", "Seen3"}
    lf_start_state = "Seen0"

    lf_transitions = {}
    lf_transitions[("Seen0", "0")] = "Seen0"
    lf_transitions[("Seen0", "1")] = "Seen1"
    lf_transitions[("Seen1", "0")] = "Seen1"
    lf_transitions[("Seen1", "1")] = "Seen2"
    lf_transitions[("Seen2", "0")] = "Seen2"
    lf_transitions[("Seen2", "1")] = "Seen3"
    lf_transitions[("Seen3", "0")] = "Seen3"
    lf_transitions[("Seen3", "1")] = "Sink"
    lf_transitions[("Sink", "0")] = "Sink"
    lf_transitions[("Sink", "1")] = "Sink"

    lf_dfa = Dfa(alphabet, lf_states, lf_accepting_states, lf_start_state, lf_transitions)

    label_map = {}
    label_map["Seen1"] = "Label1"
    label_map["Seen2"] = "Label2"
    label_map["Seen3"] = "Label3"

    label_func = LabellingDfa(lf_dfa, label_map)

    # Layout remaining Labelled CI parameters.
    length_bounds = (1,4)
    epsilon = 0.2
    label_prob_bounds = (.25, .4)

    # Create Labelled CI Improviser
    improviser = MaxEntropyLabelledCI(hard_constraint, soft_constraint, label_func, length_bounds, epsilon, label_prob_bounds)

    # Check that the calculated probabilities are valid
    assert sum(improviser.sorted_label_class_weights) == pytest.approx(1)

    for label_iter in range(len(improviser.label_func.labels)):
        sum_label_class_prob = improviser.sorted_label_class_weights[2*label_iter] + improviser.sorted_label_class_weights[2*label_iter + 1]
        assert label_prob_bounds[0] <= sum_label_class_prob <= label_prob_bounds[1]

    sum_soft_constraint_prob = 0

    for label_iter in range(len(improviser.label_func.labels)):
        sum_soft_constraint_prob += improviser.sorted_label_class_weights[2*label_iter+1]

    assert sum_soft_constraint_prob >= 1 - epsilon

    # Create sampling testing variables
    improvisations = {tuple("01"), tuple("001"), tuple("101"), tuple("0001"), tuple("0101"), tuple("1001"), tuple("1101")}
    improvisation_count = {improvisation:0 for improvisation in improvisations}

    soft_constraint_improvisations = {tuple("0001"), tuple("0101"), tuple("1001"), tuple("1101")}
    soft_constraint_count = 0

    label_count = {label:0 for label in ["Label1", "Label2", "Label3"]}

    label_improvisation_map = {}
    label_improvisation_map["Label1"] = {tuple("01"), tuple("001"), tuple("0001")}
    label_improvisation_map["Label2"] = {tuple("101"), tuple("0101"), tuple("1001")}
    label_improvisation_map["Label3"] = {tuple("1101")}

    improvisation_label_map = {}
    for label, items in label_improvisation_map.items():
        for item in items:
            improvisation_label_map[item] = label

    # Sample a collection of words from the improviser.
    for _ in range(100000):
        word = improviser.improvise()

        assert word in improvisations

        improvisation_count[word] += 1
        label_count[improvisation_label_map[word]] += 1

        if word in soft_constraint_improvisations:
            soft_constraint_count+=1

    # Check that sampled word probabilities are valid
    for label in label_func.labels:
        label_sampled_prob = label_count[label]/100000
        assert label_prob_bounds[0]-.01 <= label_sampled_prob <= label_prob_bounds[1]+.01

    assert soft_constraint_count/100000 >= .99 - epsilon

def test_max_entropy_labelled_ci_infeasible():
    """ Test that different infeasible Labelled CI
    problems correctly raise an exception.
    """
    # Create a hard constraint Dfa that accepts all words that end with "01"
    alphabet = {"0", "1"}
    h_states = {"Start", "Seen0", "Accept"}
    h_accepting_states = {"Accept"}
    h_start_state = "Start"

    h_transitions = {}
    h_transitions[("Start", "0")] = "Seen0"
    h_transitions[("Start", "1")] = "Start"
    h_transitions[("Seen0", "0")] = "Seen0"
    h_transitions[("Seen0", "1")] = "Accept"
    h_transitions[("Accept", "0")] = "Seen0"
    h_transitions[("Accept", "1")] = "Start"

    hard_constraint = Dfa(alphabet, h_states, h_accepting_states, h_start_state, h_transitions)

    # Create a soft constraint that accepts all words of length 3
    soft_constraint = Dfa.exact_length_dfa(alphabet, 3)

    # Create a labelling function that labels a word depending on how many "1"
    # symbols it has
    lf_states = {"Seen0", "Seen1", "Seen2", "Seen3", "Sink"}
    lf_accepting_states = {"Seen1", "Seen2", "Seen3"}
    lf_start_state = "Seen0"

    lf_transitions = {}
    lf_transitions[("Seen0", "0")] = "Seen0"
    lf_transitions[("Seen0", "1")] = "Seen1"
    lf_transitions[("Seen1", "0")] = "Seen1"
    lf_transitions[("Seen1", "1")] = "Seen2"
    lf_transitions[("Seen2", "0")] = "Seen2"
    lf_transitions[("Seen2", "1")] = "Seen3"
    lf_transitions[("Seen3", "0")] = "Seen3"
    lf_transitions[("Seen3", "1")] = "Sink"
    lf_transitions[("Sink", "0")] = "Sink"
    lf_transitions[("Sink", "1")] = "Sink"

    lf_dfa = Dfa(alphabet, lf_states, lf_accepting_states, lf_start_state, lf_transitions)

    label_map = {}
    label_map["Seen1"] = "Label1"
    label_map["Seen2"] = "Label2"
    label_map["Seen3"] = "Label3"

    label_func = LabellingDfa(lf_dfa, label_map)

    # Layout remaining Labelled CI parameters.
    length_bounds = (1,4)
    epsilon = 0.2
    label_prob_bounds = (.2, .4)

    # Ensure that the base LCI problem is feasible
    MaxEntropyLabelledCI(hard_constraint, soft_constraint, label_func, length_bounds, epsilon, label_prob_bounds)

    # Check that various parameter tweaks that render the
    # problem infeasible are identified by the improviser.
    with pytest.raises(InfeasibleImproviserError):
        MaxEntropyLabelledCI(hard_constraint, soft_constraint, label_func, length_bounds, 0.1, label_prob_bounds)

    with pytest.raises(InfeasibleImproviserError):
        MaxEntropyLabelledCI(hard_constraint, soft_constraint, label_func, length_bounds, epsilon, (0.25,0.25))

###################################################################################################
# Randomized Tests
###################################################################################################

# Randomized tests default parameters
_RANDOM_LCI_TEST_NUM_ITERS = 1000
_RANDOM_LCI_TEST_NUM_SAMPLES = 50000
_MIN_WORD_LENGTH_BOUND = 0
_MAX_WORD_LENGTH_BOUND = 10

@pytest.mark.slow
def test_labelled_ci_improvise_random():
    """ Tests generating a mis of fully random and random
    but feasible Labelled CI improviser instances and
    ensuring that they either are infeasible or are
    feasible and improvise correctly.
    """
    for _ in range(_RANDOM_LCI_TEST_NUM_ITERS):
        # Generate random set of LabelledCI parameters. 50% chance to
        # generate an instance that is guaranteed satisfiable.
        if random.random() < 0.5:
            # Generate completely random LCI instance.
            guaranteed_feasible = False

            min_length = random.randint(_MIN_WORD_LENGTH_BOUND, _MAX_WORD_LENGTH_BOUND)
            max_length = random.randint(min_length, _MAX_WORD_LENGTH_BOUND)
            length_bounds = (min_length, max_length)

            hard_constraint = generate_random_dfa(max_states=8)

            soft_constraint = generate_random_dfa(max_states=8, alphabet=hard_constraint.alphabet)

            lf_dfa = generate_random_dfa(max_states=8, alphabet=hard_constraint.alphabet)

            label_set = ["Label" + str(label_iter) for label_iter in range(len(lf_dfa.accepting_states))]
            label_map = {}

            for accepting_state in lf_dfa.accepting_states:
                label_map[accepting_state] = random.choice(label_set)

            label_func = LabellingDfa(lf_dfa, label_map)

            epsilon = random.uniform(0,1)

            label_min_prob = random.uniform(0,1)
            label_max_prob = random.uniform(label_min_prob, 1)
            label_prob_bounds = (label_min_prob, label_max_prob)

            word_min_prob = {label:random.uniform(0,1) for label in label_func.labels}
            word_max_prob = {label:random.uniform(word_min_prob[label], 1) for label in label_func.labels}
            word_prob_bounds = {label:(word_min_prob[label], word_max_prob[label]) for label in label_func.labels}
        else:
            # Generate a random LCI instance while making each of the parameters individually feasible.
            guaranteed_feasible = True

            min_length = random.randint(_MIN_WORD_LENGTH_BOUND, _MAX_WORD_LENGTH_BOUND)
            max_length = random.randint(min_length, _MAX_WORD_LENGTH_BOUND)
            length_bounds = (min_length, max_length)

            hard_constraint = generate_random_dfa(max_states=8)
            lf_dfa = generate_random_dfa(max_states=8, alphabet=hard_constraint.alphabet)

            label_set = ["Label" + str(label_iter) for label_iter in range(len(lf_dfa.accepting_states))]
            label_map = {}

            for accepting_state in lf_dfa.accepting_states:
                label_map[accepting_state] = random.choice(label_set)

            label_func = LabellingDfa(lf_dfa, label_map)

            while True:
                empty_label_class = False

                if (hard_constraint & lf_dfa).language_size(*length_bounds) == 0:
                    empty_label_class = True
                else:
                    for label, label_spec in label_func.decompose().items():
                        if (hard_constraint & label_spec).language_size(*length_bounds) == 0:
                            empty_label_class = True
                            break

                if not empty_label_class:
                    break

                hard_constraint = generate_random_dfa(max_states=8)
                lf_dfa = generate_random_dfa(max_states=8, alphabet=hard_constraint.alphabet)

                label_set = ["Label" + str(label_iter) for label_iter in range(len(lf_dfa.accepting_states))]
                label_map = {}

                for accepting_state in lf_dfa.accepting_states:
                    label_map[accepting_state] = random.choice(label_set)

                label_func = LabellingDfa(lf_dfa, label_map)

            soft_constraint = generate_random_dfa(max_states=8, alphabet=hard_constraint.alphabet)

            label_class_sizes = {label:(hard_constraint & label_spec).language_size(*length_bounds) for (label, label_spec) in label_func.decompose().items()}
            words_total = sum([label_class_sizes[label] for label in label_func.labels])

            a_base_spec = hard_constraint & soft_constraint
            a_words_total = sum([(a_base_spec & label_spec).language_size(*length_bounds) for (label, label_spec) in label_func.decompose().items()])

            epsilon = random.uniform(1-(a_words_total/words_total), 1)

            label_min_prob = random.uniform(0,1/len(label_func.labels))
            label_max_prob = random.uniform(1/len(label_func.labels), 1)
            label_prob_bounds = (label_min_prob, label_max_prob)

            word_min_prob = {label:random.uniform(0,1/label_class_sizes[label]) for label in label_func.labels}
            word_max_prob = {label:random.uniform(1/label_class_sizes[label], 1) for label in label_func.labels}
            word_prob_bounds = {label:(word_min_prob[label], word_max_prob[label]) for label in label_func.labels}

        # Attempt to create the improviser. If it is a feasible problem,
        # attempt to sample it and ensure that the output distribution
        # is relatively correct.
        try:
            improviser = LabelledCI(hard_constraint, soft_constraint, label_func, length_bounds, epsilon, label_prob_bounds, word_prob_bounds)
        except InfeasibleImproviserError:
            assert not guaranteed_feasible
            continue

        # Sample the improviser and ensure that the sampled distribution
        # is correct.
        improvisation_count = {}

        for _ in range(_RANDOM_LCI_TEST_NUM_SAMPLES):
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

        assert a_count/_RANDOM_LCI_TEST_NUM_SAMPLES >= .99 - epsilon

        for label in label_func.labels:
            label_count = 0
            label_spec = label_func.decompose()[label]
            label_words = set()

            for word, count in improvisation_count.items():
                if label_spec.accepts(word):
                    label_count += count
                    label_words.add(word)

            assert label_prob_bounds[0] - 0.01 <= label_count/_RANDOM_LCI_TEST_NUM_SAMPLES <= label_prob_bounds[1] + 0.01

            for word in label_words:
                count = improvisation_count[word]
                assert word_prob_bounds[label][0] - 0.05 <= count/label_count <= word_prob_bounds[label][1] + 0.05

@pytest.mark.slow
def test_max_entropy_labelled_ci_improvise_random():
    """ Tests generating a mix of fully random and random
    but feasible Max Entropy Labelled CI improviser instances
    and ensuring that they either are infeasible or are
    feasible and improvise correctly. Also ensures that a
    MELCI instance is feasible iff the associated LCI instance
    is feasible.
    """
    for _ in range(_RANDOM_LCI_TEST_NUM_ITERS):
        # Generate random set of MaxEntropyLabelledCI parameters. 50% chance to
        # generate an instance that is guaranteed satisfiable.
        if random.random() < 0.5:
            # Generate completely random MELCI instance.
            guaranteed_feasible = False

            min_length = random.randint(_MIN_WORD_LENGTH_BOUND, _MAX_WORD_LENGTH_BOUND)
            max_length = random.randint(min_length, _MAX_WORD_LENGTH_BOUND)
            length_bounds = (min_length, max_length)

            hard_constraint = generate_random_dfa(max_states=8)

            soft_constraint = generate_random_dfa(max_states=8, alphabet=hard_constraint.alphabet)

            lf_dfa = generate_random_dfa(max_states=8, alphabet=hard_constraint.alphabet)

            label_set = ["Label" + str(label_iter) for label_iter in range(len(lf_dfa.accepting_states))]
            label_map = {}

            for accepting_state in lf_dfa.accepting_states:
                label_map[accepting_state] = random.choice(label_set)

            label_func = LabellingDfa(lf_dfa, label_map)

            epsilon = random.uniform(0,1)

            label_min_prob = random.uniform(0,1)
            label_max_prob = random.uniform(label_min_prob, 1)
            label_prob_bounds = (label_min_prob, label_max_prob)

        else:
            # Generate a random LCI instance while making each of the parameters individually feasible.
            guaranteed_feasible = True

            min_length = random.randint(_MIN_WORD_LENGTH_BOUND, _MAX_WORD_LENGTH_BOUND)
            max_length = random.randint(min_length, _MAX_WORD_LENGTH_BOUND)
            length_bounds = (min_length, max_length)

            hard_constraint = None
            label_func = None

            while True:
                hard_constraint = generate_random_dfa(max_states=8)
                lf_dfa = generate_random_dfa(max_states=8, alphabet=hard_constraint.alphabet)

                label_set = ["Label" + str(label_iter) for label_iter in range(len(lf_dfa.accepting_states))]
                label_map = {}

                for accepting_state in lf_dfa.accepting_states:
                    label_map[accepting_state] = random.choice(label_set)

                label_func = LabellingDfa(lf_dfa, label_map)

                empty_label_class = False

                if (hard_constraint & lf_dfa).language_size(*length_bounds) == 0:
                    empty_label_class = True
                else:
                    for label, label_spec in label_func.decompose().items():
                        if (hard_constraint & label_spec).language_size(*length_bounds) == 0:
                            empty_label_class = True
                            break

                if not empty_label_class:
                    break

            soft_constraint = generate_random_dfa(max_states=8, alphabet=hard_constraint.alphabet)

            label_class_sizes = {label:(hard_constraint & label_spec).language_size(*length_bounds) for (label, label_spec) in label_func.decompose().items()}
            words_total = sum([label_class_sizes[label] for label in label_func.labels])

            a_base_spec = hard_constraint & soft_constraint
            a_words_total = sum([(a_base_spec & label_spec).language_size(*length_bounds) for (label, label_spec) in label_func.decompose().items()])

            epsilon = random.uniform(1-(a_words_total/words_total), 1)

            label_min_prob = random.uniform(0,1/len(label_func.labels))
            label_max_prob = random.uniform(1/len(label_func.labels), 1)
            label_prob_bounds = (label_min_prob, label_max_prob)

        # Attempt to create the MELCI improviser. Then check that
        # the associated LCI instance is also feasible/infeasible.
        # If it is a feasible problem, attempt to sample it and
        # ensure that the output distribution is relatively correct.
        try:
            improviser = MaxEntropyLabelledCI(hard_constraint, soft_constraint, label_func, length_bounds, epsilon, label_prob_bounds)
        except InfeasibleImproviserError:
            assert not guaranteed_feasible
            melci_feasible = False
        else:
            melci_feasible = True

        try:
            trivial_word_bounds = {label:(0,1) for label in label_func.labels}
            LabelledCI(hard_constraint, soft_constraint, label_func, length_bounds, epsilon, label_prob_bounds, trivial_word_bounds)
        except InfeasibleImproviserError:
            assert not guaranteed_feasible
            lci_feasible = False
        else:
            lci_feasible = True

        assert melci_feasible == lci_feasible

        if not melci_feasible:
            continue

        # Sample the improviser and ensure that the sampled distribution
        # is correct.
        improvisation_count = {}

        for _ in range(_RANDOM_LCI_TEST_NUM_SAMPLES):
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

        assert a_count/_RANDOM_LCI_TEST_NUM_SAMPLES >= .99 - epsilon

        for label in label_func.labels:
            label_count = 0
            label_spec = label_func.decompose()[label]
            label_words = set()

            for word, count in improvisation_count.items():
                if label_spec.accepts(word):
                    label_count += count
                    label_words.add(word)

            assert label_prob_bounds[0] - 0.01 <= label_count/_RANDOM_LCI_TEST_NUM_SAMPLES <= label_prob_bounds[1] + 0.01
