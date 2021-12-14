""" Contains the LabelledCI class, which acts as an improviser
for the Labelled CI problem.
"""

from __future__ import annotations

from citoolkit.improvisers.labelled_quantitative_ci import LabelledQuantitativeCI, MaxEntropyLabelledQuantitativeCI
from citoolkit.improvisers.improviser import InfeasibleCostError, InfeasibleSoftConstraintError
from citoolkit.specifications.spec import ExactSpec
from citoolkit.labellingfunctions.labelling_func import ExactLabellingFunc
from citoolkit.costfunctions.cost_func import SoftConstraintCostFunc
from citoolkit.util.logging import cit_log

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
    def __init__(self, hard_constraint: ExactSpec, soft_constraint: ExactSpec, label_func: ExactLabellingFunc, \
                 length_bounds: tuple[int, int], epsilon: float, \
                 label_prob_bounds: tuple[float, float], word_prob_bounds: dict[str, tuple[float, float]],
                 num_threads: int =1,verbose: bool =False) -> None:
        # Checks that parameters are well formed
        if not isinstance(hard_constraint, ExactSpec):
            raise ValueError("The hard_constraint parameter must be a member of the ExactSpec class.")

        if not isinstance(soft_constraint, ExactSpec):
            raise ValueError("The soft_constraint parameter must be a member of the ExactSpec class.")

        if not isinstance(label_func, ExactLabellingFunc):
            raise ValueError("The label_func parameter must be a member of the ExactLabellingFunc class.")

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

        if verbose:
            cit_log("Generalizing LCI problem to equivalent LQCI problem.")

        # Solve associated LQCI problem, catching and transforming InfeasibleImproviserExceptions to fit this problem.
        try:
            super().__init__(hard_constraint, cost_func, label_func, length_bounds, cost_bound, label_prob_bounds, word_prob_bounds,\
                num_threads=num_threads, verbose=verbose)
        except InfeasibleCostError as exc:
            raise InfeasibleSoftConstraintError("Greedy construction does not satisfy soft constraint, meaning no improviser can."\
                + " Maximum soft constraint probability was " + str(1 - exc.best_cost) + ".", (1-exc.best_cost)) from exc

class MaxEntropyLabelledCI(MaxEntropyLabelledQuantitativeCI):
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
    def __init__(self, hard_constraint: ExactSpec, soft_constraint: ExactSpec, label_func: ExactLabellingFunc, \
                 length_bounds: tuple[int, int], epsilon: float, \
                 label_prob_bounds: tuple[float, float], num_threads: int =1, verbose:bool =False) -> None:
        # Checks that parameters are well formed
        if not isinstance(hard_constraint, ExactSpec):
            raise ValueError("The hard_constraint parameter must be a member of the ExactSpec class.")

        if not isinstance(soft_constraint, ExactSpec):
            raise ValueError("The soft_constraint parameter must be a member of the ExactSpec class.")

        if not isinstance(label_func, ExactLabellingFunc):
            raise ValueError("The label_func parameter must be a member of the ExactLabellingFunc class.")

        if (len(length_bounds) != 2) or (length_bounds[0] < 0) or (length_bounds[0] > length_bounds[1]):
            raise ValueError("The length_bounds parameter should contain two integers, with 0 <= length_bounds[0] <= length_bounds[1].")

        if epsilon < 0 or epsilon > 1:
            raise ValueError("The epsilon parameter should be between 0 and 1 inclusive.")

        if (len(label_prob_bounds) != 2) or (label_prob_bounds[0] < 0) or (label_prob_bounds[0] > label_prob_bounds[1]) or (label_prob_bounds[1] > 1):
            raise ValueError("The prob_bounds parameter should contain two floats, with 0 <= prob_bounds[0] <= prob_bounds[1] <= 1.")

        # Convert to equivalent LQCI parameters
        cost_func = SoftConstraintCostFunc(soft_constraint)
        cost_bound = epsilon

        if verbose:
            cit_log("Generalizing MELCI problem to equivalent MELQCI problem.")

        # Solve associated MELQCI problem.
        super().__init__(hard_constraint, cost_func, label_func, length_bounds, cost_bound, label_prob_bounds,\
            num_threads=num_threads, verbose=verbose)
