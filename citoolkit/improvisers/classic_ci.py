""" Contains the ClassicCI class, which acts as an improviser
for the Classic CI problem.
"""

from __future__ import annotations

from citoolkit.improvisers.labelled_quantitative_ci import LabelledQuantitativeCI
from citoolkit.improvisers.improviser import InfeasibleImproviserError
from citoolkit.specifications.spec import Spec
from citoolkit.costfunctions.cost_func import SoftConstraintCostFunc
from citoolkit.labellingfunctions.labelling_func import TrivialLabelFunc

class ClassicCI(LabelledQuantitativeCI):
    """ An improviser for the original Control Improvisation problem.

    :param hard_constraint: A specification that must accept all improvisations
    :param soft_constraint: A specification that must accept improvisations with
        probability 1 - epsilon.
    :param length_bounds: A tuple containing lower and upper bounds on the length
        of a generated word.
    :param epsilon: The allowed tolerance with which we can not satisfy the soft constraint.
    :param prob_bounds: A tuple containing lower and upper bounds on the probability with
        which we can generate a word.
    :raises ValueError: If passed parameters are not of the correct type.
    :raises InfeasibleImproviserError: If the resulting improvisation problem is not feasible.
    """
    def __init__(self, hard_constraint: Spec, soft_constraint: Spec, length_bounds: tuple[int, int], \
                 epsilon: float, prob_bounds: tuple[float, float]) -> None:
        # Checks that parameters are well formed
        if not isinstance(hard_constraint, Spec):
            raise ValueError("The hard_constraint parameter must be a member of the Spec class.")

        if not isinstance(soft_constraint, Spec):
            raise ValueError("The soft_constraint parameter must be a member of the Spec class.")

        if (len(length_bounds) != 2) or (length_bounds[0] < 0) or (length_bounds[0] > length_bounds[1]):
            raise ValueError("The length_bounds parameter should contain two integers, with 0 <= length_bounds[0] <= length_bounds[1].")

        if epsilon < 0 or epsilon > 1:
            raise ValueError("The epsilon parameter should be between 0 and 1 inclusive.")

        if (len(prob_bounds) != 2) or (prob_bounds[0] < 0) or (prob_bounds[0] > prob_bounds[1]) or (prob_bounds[1] > 1):
            raise ValueError("The prob_bounds parameter should contain two floats, with 0 <= prob_bounds[0] <= prob_bounds[1] <= 1.")

        # Convert to equivalent LQCI parameters
        cost_func = SoftConstraintCostFunc(soft_constraint)
        label_func = TrivialLabelFunc()
        cost_bound = epsilon
        label_prob_bounds = (1,1)
        word_prob_bounds = {"TrivialLabel": prob_bounds}

        # Solve associated LQCI problem.
        super().__init__(hard_constraint, cost_func, label_func, length_bounds, cost_bound, label_prob_bounds, word_prob_bounds)
