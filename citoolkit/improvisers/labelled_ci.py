""" Contains the LabelledCI class, which acts as an improviser
for the Labelled CI problem.
"""

from __future__ import annotations

import warnings
import random
import itertools
import cvxpy as cp
import numpy as np

from citoolkit.improvisers.labelled_quantitative_ci import LabelledQuantitativeCI
from citoolkit.improvisers.improviser import Improviser, InfeasibleImproviserError
from citoolkit.specifications.spec import Spec
from citoolkit.labellingfunctions.labelling_func import LabellingFunc
from citoolkit.costfunctions.cost_func import SoftConstraintCostFunc

class LabelledCI(LabelledQuantitativeCI):
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

        # Convert to equivalent LQCI parameters
        cost_func = SoftConstraintCostFunc(soft_constraint)
        cost_bound = epsilon

        # Solve associated LQCI problem.
        super().__init__(hard_constraint, cost_func, label_func, length_bounds, cost_bound, label_prob_bounds, word_prob_bounds)


class MaxEntropyLabelledCI(Improviser):
    """ An improviser for the Maximum Entropy Labelled Control Improvisation problem.

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
        if "infeasible" in prob.status:
            raise InfeasibleImproviserError()

        if prob.status != "optimal":
            warnings.warn("Got unexpected value '" + prob.status + "' as optimizer output.")

        # Store improvisation variables
        self.sorted_label_class_specs = list(itertools.chain(*[[self.i_specs[label], self.a_specs[label]] for label in sorted_labels]))
        self.sorted_label_class_weights = list(x.value)
        self.entropy = -result
        self.status = prob.status

    def improvise(self) -> tuple[str, ...]:
        """ Improvise a single word.

        :returns: A single improvised word.
        """
        selected_label_class_spec = random.choices(population=self.sorted_label_class_specs, weights=self.sorted_label_class_weights, k=1)[0]

        return selected_label_class_spec.sample(*self.length_bounds)
