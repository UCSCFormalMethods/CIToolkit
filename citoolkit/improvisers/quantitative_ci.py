""" Contains the QuantitativeCI class, which acts as an improviser
for the Quantitative CI problem.
"""

from __future__ import annotations

from citoolkit.improvisers.labelled_quantitative_ci import LabelledQuantitativeCI
from citoolkit.improvisers.improviser import InfeasibleImproviserError, InfeasibleRandomnessError, \
                                             InfeasibleLabelRandomnessError, InfeasibleWordRandomnessError
from citoolkit.specifications.spec import ExactSpec
from citoolkit.costfunctions.cost_func import ExactCostFunc
from citoolkit.labellingfunctions.labelling_func import TrivialLabellingFunc
from citoolkit.util.logging import cit_log

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
    def __init__(self, hard_constraint: ExactSpec, cost_func: ExactCostFunc, length_bounds: tuple[int, int], \
                 cost_bound: float, prob_bounds: tuple[float, float],\
                 num_threads:int =1, verbose:bool =False) -> None:
        # Checks that parameters are well formed
        if not isinstance(hard_constraint, ExactSpec):
            raise ValueError("The hard_constraint parameter must be a member of the ExactSpec class.")

        if not isinstance(cost_func, ExactCostFunc):
            raise ValueError("The cost_func parameter must be a member of the ExactCostFunc class.")

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

        if verbose:
            cit_log("Generalizing QCI problem to equivalent LQCI problem.")

        # Solve associated LQCI problem, catching and transforming InfeasibleImproviserExceptions to fit this problem.
        try:
            super().__init__(hard_constraint, cost_func, label_func, length_bounds, cost_bound, label_prob_bounds, word_prob_bounds, \
                num_threads=num_threads, verbose=verbose)
        except InfeasibleLabelRandomnessError as exc:
            raise InfeasibleImproviserError("There are no feasible improvisations.") from exc
        except InfeasibleWordRandomnessError as exc:
            if prob_bounds[0] == 0:
                inv_min_prob = float("inf")
            else:
                inv_min_prob = 1/prob_bounds[0]

            raise InfeasibleRandomnessError("Violation of condition 1/prob_bounds[1] <= i_size <= 1/prob_bounds[0]. Instead, " \
                + str(1/prob_bounds[1]) + " <= " + str(exc.set_size)  + " <= " + str(inv_min_prob), exc.set_size) from exc
