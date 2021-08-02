""" Contains the LabelledCI class, which acts as an improviser
for the Labelled CI problem.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", module="np")

import random
import itertools
import cvxpy as cp
import numpy as np

from citoolkit.improvisers.improviser import Improviser, InfeasibleImproviserError
from citoolkit.specifications.spec import Spec
from citoolkit.labellingfunctions.labelling_func import LabellingFunc

class LabelledCI(Improviser):
    """ An improviser for the Labelled Control Improvisation problem.

    :param hard_constraint: A specification that must accept all improvisations
    :param soft_constraint: A specification that must accept improvisations with
        probability 1 - epsilon.
    :param label_func: A labelling function that must associate a label with all
        improvisations.
    :param length_bounds: A tuple containing lower and upper bounds on the length
        of a generated word.
    :param epsilon: The allowed tolerance with which we can not satisfy the soft constraint.
    :param label_prob_bounds: A tuple containing lower and upper bounds on the
        marginal probability with which we can generate a word with a particular label.
    :param word_prob_bounds: A dictionary mapping each label in label_func to a tuple. Each
        tuple contains a lower and upper bound on the conditional probability of selecting
        a word with the associated label conditioned on the fact that we do select a word
        with label.
    :raises InfeasibleImproviserError: If the resulting improvisation problem is not feasible.
    """
    def __init__(self, hard_constraint: Spec, soft_constraint: Spec, label_func: LabellingFunc, \
                 length_bounds: tuple[int, int], epsilon: float, \
                 label_prob_bounds: tuple[float, float], word_prob_bounds: dict[str, tuple[float, float]]) -> None:
        # Checks that parameters are well formed
        if not isinstance(hard_constraint, Spec):
            raise ValueError("The hard_constraint parameter must be a member of the Spec class.")

        if not isinstance(soft_constraint, Spec):
            raise ValueError("The soft_constraint parameter must be a member of the Spec class.")

        if not isinstance(label_func, LabellingFunc):
            raise ValueError("The label_func parameter must be a member of the LabellingFunc class.")

        if (len(length_bounds) != 2) or (length_bounds[0] < 0) or (length_bounds[0] > length_bounds[1]):
            raise ValueError("The length_bounds parameter should contain two integers, with 0 <= length_bounds[0] <= length_bounds[1].")

        if epsilon < 0 or epsilon > 1:
            raise ValueError("The epsilon parameter should be between 0 and 1 inclusive.")

        if (len(label_prob_bounds) != 2) or (label_prob_bounds[0] < 0) or (label_prob_bounds[0] > label_prob_bounds[1]) or (label_prob_bounds[1] > 1):
            raise ValueError("The prob_bounds parameter should contain two floats, with 0 <= prob_bounds[0] <= prob_bounds[1] <= 1.")

        for label in label_func.labels:
            target_prob_bounds = word_prob_bounds[label]

            if label not in word_prob_bounds.keys():
                raise ValueError("The word_prob_bounds parameter is missing conditional probability bounds for the label '" + label + "'.")

            if (len(target_prob_bounds) != 2) or (target_prob_bounds[0] < 0) or (target_prob_bounds[0] > target_prob_bounds[1]) or (target_prob_bounds[1] > 1):
                raise ValueError("The prob_bounds parameter should contain two floats, with 0 <= prob_bounds[0] <= prob_bounds[1] <= 1.")

        # Store all constructor parameters
        self.hard_constraint = hard_constraint
        self.soft_constraint = soft_constraint
        self.label_func = label_func
        self.length_bounds = length_bounds
        self.epsilon = epsilon
        self.label_prob_bounds = label_prob_bounds
        self.word_prob_bounds = word_prob_bounds

        # Initialize label class specs. Note that i_specs refers to I\A, not I.
        label_specs = label_func.decompose()

        self.i_specs = {}
        self.a_specs = {}

        for label in label_func.labels:
            base_label_class = hard_constraint & label_specs[label]
            self.i_specs[label] = base_label_class - soft_constraint
            self.a_specs[label] = base_label_class & soft_constraint

        # Pick the conditional probabilities i_prob and a_prob for each label class.
        self.i_probs = {}
        self.a_probs = {}

        for label in label_func.labels:
            i_size = self.i_specs[label].language_size(*length_bounds)
            a_size = self.a_specs[label].language_size(*length_bounds)

            if label_prob_bounds[0] > 0 and i_size + a_size == 0:
                raise InfeasibleImproviserError("No strings are labelled with label '" + label + "', but label_prob_bounds[0] > 0.")

            min_word_prob, max_word_prob = word_prob_bounds[label]

            self.i_probs[label] = max(1 - max_word_prob * a_size, min_word_prob * i_size)
            self.a_probs[label] = 1 - self.i_probs[label]

        # Greedily assign marginal probability to label classes with high
        # probability of selecting words that satisfy the soft constraint.
        self.label_prob = {}
        min_label_prob, max_label_prob = label_prob_bounds

        # Compute number of label classes that are assigned max_label_prob.
        # Note: The enumerate below is 0 indexed unlike in paper.
        if min_label_prob != max_label_prob:
            num_max_prob_classes = int((1 - len(label_func.labels)*min_label_prob)/(max_label_prob - min_label_prob))
        else:
            num_max_prob_classes = len(label_func.labels)

        # Probability assigned to the class at index (num_max_prob_classes + 1) at sorted_label_classes.
        mid_class_prob = 1 - max_label_prob*num_max_prob_classes - min_label_prob*(len(label_func.labels) - num_max_prob_classes - 1)

        for label_num, label in enumerate(sorted(label_func.labels, key=lambda x: self.a_probs[x], reverse=True)):
            if label_num < num_max_prob_classes:
                self.label_prob[label] = max_label_prob
            elif label_num == num_max_prob_classes:
                self.label_prob[label] = mid_class_prob
            else:
                self.label_prob[label] = min_label_prob

        # Place improviser values in form used by improvise function
        self.sorted_labels = sorted(filter(lambda x: self.i_specs[x].language_size(*length_bounds) + self.a_specs[x].language_size(*length_bounds) > 0, label_func.labels))
        self.sorted_labels_weights = [self.label_prob[label] for label in self.sorted_labels]

        # Checks that improviser is feasible. If not raise an InfeasibleImproviserError.
        if len(label_func.labels) < (1/max_label_prob) or (min_label_prob != 0 and len(label_func.labels) > (1/min_label_prob)):
            if min_label_prob == 0:
                inv_min_label_prob = float("inf")
            else:
                inv_min_label_prob = 1/min_label_prob

            raise InfeasibleImproviserError("Violation of condition 1/label_prob_bounds[1] <= len(label_func.labels) <= 1/label_prob_bounds[0]. Instead, " \
                                            + str(1/max_label_prob) + " <= " + str(len(label_func.labels)) + " <= " + str(inv_min_label_prob))

        for label in label_func.labels:
            label_class_size = self.i_specs[label].language_size(*length_bounds) + self.a_specs[label].language_size(*length_bounds)
            min_word_prob, max_word_prob = word_prob_bounds[label]

            if label_class_size < (1/max_word_prob) or (min_word_prob != 0 and label_class_size > (1/min_word_prob)):
                if min_word_prob == 0:
                    inv_min_word_prob = float("inf")
                else:
                    inv_min_word_prob = 1/min_word_prob

                raise InfeasibleImproviserError("Violation for label '" + label + "' " +
                                                "of condition 1/word_prob_bounds[" + label + "][1] <= label_class_size <= 1/word_prob_bounds[" + label + "][0]." +
                                                " Instead, " + str(1/max_word_prob) + " <= " + str(label_class_size) + " <= " + str(inv_min_word_prob))

        soft_constraint_prob = sum([self.label_prob[label]*self.a_probs[label] for label in label_func.labels])

        if 1 - epsilon > soft_constraint_prob:
            raise InfeasibleImproviserError("Greedy construction does not satisfy soft constraint, meaning no improviser can.")

    def improvise(self) -> tuple[str,...]:
        """ Improvise a single word.

        :returns: A single improvised word.
        """
        selected_label = random.choices(population=self.sorted_labels, weights=self.sorted_labels_weights, k=1)[0]

        rand = random.random()

        if rand < self.i_probs[selected_label]:
            return self.i_specs[selected_label].sample(*self.length_bounds)
        else:
            return self.a_specs[selected_label].sample(*self.length_bounds)

class MaxEntropyLabelledCI(Improviser):
    """ An improviser for the Labelled Control Improvisation problem.

    :param hard_constraint: A specification that must accept all improvisations
    :param soft_constraint: A specification that must accept improvisations with
        probability 1 - epsilon.
    :param label_func: A labelling function that must associate a label with all
        improvisations.
    :param length_bounds: A tuple containing lower and upper bounds on the length
        of a generated word.
    :param epsilon: The allowed tolerance with which we can not satisfy the soft constraint.
    :param label_prob_bounds: A tuple containing lower and upper bounds on the
        marginal probability with which we can generate a word with a particular label.
    :raises InfeasibleImproviserError: If the resulting improvisation problem is not feasible.
    """
    def __init__(self, hard_constraint: Spec, soft_constraint: Spec, label_func: LabellingFunc, \
                 length_bounds: tuple[int, int], epsilon: float, \
                 label_prob_bounds: tuple[float, float]) -> None:
        # Checks that parameters are well formed
        if not isinstance(hard_constraint, Spec):
            raise ValueError("The hard_constraint parameter must be a member of the Spec class.")

        if not isinstance(soft_constraint, Spec):
            raise ValueError("The soft_constraint parameter must be a member of the Spec class.")

        if not isinstance(label_func, LabellingFunc):
            raise ValueError("The label_func parameter must be a member of the LabellingFunc class.")

        if (len(length_bounds) != 2) or (length_bounds[0] < 0) or (length_bounds[0] > length_bounds[1]):
            raise ValueError("The length_bounds parameter should contain two integers, with 0 <= length_bounds[0] <= length_bounds[1].")

        if epsilon < 0 or epsilon > 1:
            raise ValueError("The epsilon parameter should be between 0 and 1 inclusive.")

        if (len(label_prob_bounds) != 2) or (label_prob_bounds[0] < 0) or (label_prob_bounds[0] > label_prob_bounds[1]) or (label_prob_bounds[1] > 1):
            raise ValueError("The prob_bounds parameter should contain two floats, with 0 <= prob_bounds[0] <= prob_bounds[1] <= 1.")

        # Checks that there are any words that satisfy the hard constraint and have a label
        if (hard_constraint & label_func.dfa).language_size(*length_bounds) == 0:
            raise InfeasibleImproviserError("There are no words that both satisfy the hard constraint and have an assigned label.")

        # Store all constructor parameters
        self.hard_constraint = hard_constraint
        self.soft_constraint = soft_constraint
        self.label_func = label_func
        self.length_bounds = length_bounds
        self.epsilon = epsilon
        self.label_prob_bounds = label_prob_bounds

        # Initialize label class specs. Note that i_specs refers to I\A, not I.
        label_specs = label_func.decompose()

        self.i_specs = {}
        self.a_specs = {}

        for label in label_func.labels:
            base_label_class = hard_constraint & label_specs[label]
            self.i_specs[label] = base_label_class - soft_constraint
            self.a_specs[label] = base_label_class & soft_constraint

        # Create the optimization problem parameters.
        sorted_labels = sorted(label_func.labels)
        omega = len(sorted_labels)

        # x is a vector containing all variables. I\A_i is at
        # position (2i) and A_i is at position (2i + 1) where
        # position is determined by location in sorted_labels.
        x = cp.Variable(2 * omega, nonneg=True)

        # Create objective
        label_class_sizes = np.concatenate([[max(1,self.i_specs[label].language_size(*length_bounds)), max(1,self.a_specs[label].language_size(*length_bounds))] for label in sorted_labels])

        opt_equation = - cp.sum(cp.entr(x)) - (x @ np.log(label_class_sizes).T)

        objective = cp.Minimize(opt_equation)

        # Create constraints list
        constraints = []

        # (C1) Satisfaction of the soft constraint
        constraints.append(x @ np.array([[0, 1]*omega]).T >= 1 - epsilon)

        # (C2) Randomness over Labels lower bound
        for label_iter, label in enumerate(sorted_labels):
            constraints.append(x @ np.concatenate([([1, 1] if i == label_iter else [0, 0]) for i in range(omega)]).T >= self.label_prob_bounds[0])

        # (C3) Randomness over Labels upper bound
        for label_iter, label in enumerate(sorted_labels):
            constraints.append(x @ np.concatenate([([1, 1] if i == label_iter else [0, 0]) for i in range(omega)]).T <= self.label_prob_bounds[1])

        # (C4) - (C5) Non negative I\A and A set probability
        constraints.append(x >= 0)

        # (C6) Probability distribution sums to 1
        constraints.append(x @ np.array([1, 1]*omega).T == 1)

        # (C7) Empty I\A sets have 0 probability
        for label_iter, label in enumerate(sorted_labels):
            if self.i_specs[label].language_size(*length_bounds) == 0:
                constraints.append(x @ np.concatenate([([1, 0] if i == label_iter else [0, 0]) for i in range(omega)]).T == 0)

        # (C8) Empty A sets have 0 probability
        for label_iter, label in enumerate(sorted_labels):
            if self.a_specs[label].language_size(*length_bounds) == 0:
                constraints.append(x @ np.concatenate([([0, 1] if i == label_iter else [0, 0]) for i in range(omega)]).T == 0)

        # Create and solve problem
        prob = cp.Problem(objective, constraints)
        result = prob.solve()

        # Check if problem is feasible. If not, raise an InfeasibleImproviserError.
        if prob.status == "infeasible":
            raise InfeasibleImproviserError()

        if prob.status != "optimal":
            warnings.warn("Got unexpected value '" + prob.status + "' as optimizer output.")

        # Store improvisation variables
        self.sorted_label_class_specs = list(itertools.chain(*[[self.i_specs[label], self.a_specs[label]] for label in sorted_labels]))
        self.sorted_label_class_weights = list(x.value)
        self.entropy = -result

    def improvise(self) -> tuple[str, ...]:
        """ Improvise a single word.

        :returns: A single improvised word.
        """
        selected_label_class_spec = random.choices(population=self.sorted_label_class_specs, weights=self.sorted_label_class_weights, k=1)[0]

        return selected_label_class_spec.sample(*self.length_bounds)
