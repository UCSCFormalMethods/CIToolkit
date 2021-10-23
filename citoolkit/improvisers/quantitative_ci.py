""" Contains the QuantitativeCI class, which acts as an improviser
for the Quantitative CI problem.
"""

from __future__ import annotations

import random

from citoolkit.improvisers.labelled_quantitative_ci import LabelledQuantitativeCI
from citoolkit.improvisers.improviser import InfeasibleImproviserError
from citoolkit.specifications.spec import Spec
from citoolkit.costfunctions.cost_func import CostFunc
from citoolkit.labellingfunctions.labelling_func import TrivialLabellingFunc

class QuantitativeCI(LabelledQuantitativeCI):
    """ An improviser for the Quantitative Control Improvisation problem.

    :param hard_constraint: A specification that must accept all improvisations
    :param cost_func: A cost function that must associate a rational cost
        with all improvisations.
    :param length_bounds: A tuple containing lower and upper bounds on the length
        of a generated word.
    :param cost_bound: The maximum allowed expected cost for our improviser.
    :param prob_bounds: A tuple containing lower and upper bounds on the
        probability with which we can generate a word.
    :raises InfeasibleImproviserError: If the resulting improvisation problem is not feasible.
    """
    def __init__(self, hard_constraint: Spec, cost_func: CostFunc, length_bounds: tuple[int, int], \
                 cost_bound: float, prob_bounds: tuple[float, float]) -> None:
        # Checks that parameters are well formed
        if not isinstance(hard_constraint, Spec):
            raise ValueError("The hard_constraint parameter must be a member of the Spec class.")

        if not isinstance(cost_func, CostFunc):
            raise ValueError("The cost_func parameter must be a member of the CostFunc class.")

        if (len(length_bounds) != 2) or (length_bounds[0] < 0) or (length_bounds[0] > length_bounds[1]):
            raise ValueError("The length_bounds parameter should contain two integers, with 0 <= length_bounds[0] <= length_bounds[1].")

        if cost_bound < 0:
            raise ValueError("The cost_bound parameter must be a number >= 0.")

        if (len(prob_bounds) != 2) or (prob_bounds[0] < 0) or (prob_bounds[0] > prob_bounds[1]) or (prob_bounds[1] > 1):
            raise ValueError("The prob_bounds parameter should contain two floats, with 0 <= prob_bounds[0] <= prob_bounds[1] <= 1.")

        # Convert to equivalent LQCI parameters
        label_func = TrivialLabellingFunc()
        label_prob_bounds = (1,1)
        word_prob_bounds = {"TrivialLabel": prob_bounds}

        # Solve associated LQCI problem.
        super().__init__(hard_constraint, cost_func, label_func, length_bounds, cost_bound, label_prob_bounds, word_prob_bounds)
