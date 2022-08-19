""" Contains the CI class, which acts as an improviser
for the original Control Improvisation problem.
"""

from __future__ import annotations

# Initialize Logger
import logging

logger = logging.getLogger(__name__)

from citoolkit.improvisers.lqci import LQCI
from citoolkit.improvisers.improviser import (
    InfeasibleImproviserError,
    InfeasibleRandomnessError,
    InfeasibleCostError,
    InfeasibleSoftConstraintError,
    InfeasibleLabelRandomnessError,
    InfeasibleWordRandomnessError,
)
from citoolkit.specifications.spec import ExactSpec
from citoolkit.costfunctions.cost_func import SoftConstraintCostFunc
from citoolkit.labellingfunctions.labelling_func import TrivialLabellingFunc


class CI(LQCI):
    """An improviser for the original Control Improvisation (CI) problem.

    :param hard_constraint: A specification that must accept all improvisations
    :param soft_constraint: A specification that must accept improvisations with
        probability 1 - epsilon.
    :param length_bounds: A tuple containing lower and upper bounds on the length
        of a generated word.
    :param num_threads: The number of threads to use in initialization computation.
    :param lazy: Whether or not to lazily initialize the improvizer.
    :raises ValueError: If passed parameters are not well formed.
    """

    def __init__(
        self,
        hard_constraint: ExactSpec,
        soft_constraint: ExactSpec,
        length_bounds: tuple[int, int],
        num_threads: int = 1,
        lazy:bool =False,
    ) -> None:
        # Checks that parameters are well formed
        if not isinstance(hard_constraint, ExactSpec):
            raise ValueError(
                "The hard_constraint parameter must be a member of the ExactSpec class."
            )

        if not isinstance(soft_constraint, ExactSpec):
            raise ValueError(
                "The soft_constraint parameter must be a member of the ExactSpec class."
            )

        if (
            (len(length_bounds) != 2)
            or (length_bounds[0] < 0)
            or (length_bounds[0] > length_bounds[1])
        ):
            raise ValueError(
                "The length_bounds parameter should contain two integers, with 0 <="
                " length_bounds[0] <= length_bounds[1]."
            )

        # Convert to equivalent LQCI parameters
        cost_func = SoftConstraintCostFunc(soft_constraint)
        label_func = TrivialLabellingFunc()

        # Initialize associated LQCI problem
        logger.info("Generalizing CI problem to equivalent LQCI problem.")
        super().__init__(
            hard_constraint,
            cost_func,
            label_func,
            length_bounds,
            num_threads=num_threads,
            lazy=lazy,
        )

    def parameterize(
        self, epsilon: float, prob_bounds: tuple[float, float], num_threads: int = 1
    ) -> None:
        """Parameterize the CI improviser.

        :param epsilon: The maximum allowed percentage of words that are allowed to not satisfy
            the soft constraint.
        :param prob_bounds: A tuple containing lower and upper bounds on the probability of
            generating a word.
        :param num_threads: The number of threads to use in parameterization computation.
        :raises ValueError: If passed parameters are not well formed.
        :raises InfeasibleImproviserError: If the resulting improvisation problem is not feasible.
        """
        # Checks that parameters are well formed
        if epsilon < 0 or epsilon > 1:
            raise ValueError(
                "The epsilon parameter should be between 0 and 1 inclusive."
            )

        if (
            (len(prob_bounds) != 2)
            or (prob_bounds[0] < 0)
            or (prob_bounds[0] > prob_bounds[1])
            or (prob_bounds[1] > 1)
        ):
            raise ValueError(
                "The prob_bounds parameter should contain two floats, with 0 <="
                " prob_bounds[0] <= prob_bounds[1] <= 1."
            )

        # Convert to equivalent LQCI parameters
        cost_bound = epsilon
        label_prob_bounds = (1, 1)
        word_prob_bounds = {"TrivialLabel": prob_bounds}

        # Solve associated LQCI problem, catching and transforming InfeasibleImproviserExceptions to fit this problem.
        try:
            super().parameterize(
                cost_bound, label_prob_bounds, word_prob_bounds, num_threads=num_threads
            )
        except InfeasibleLabelRandomnessError as exc:
            raise InfeasibleImproviserError(
                "There are no feasible improvisations."
            ) from exc
        except InfeasibleWordRandomnessError as exc:
            if prob_bounds[0] == 0:
                inv_min_prob = float("inf")
            else:
                inv_min_prob = 1 / prob_bounds[0]

            raise InfeasibleRandomnessError(
                "Violation of condition 1/prob_bounds[1] <= i_size <="
                f" 1/prob_bounds[0]. Instead, {1 / prob_bounds[1]} <="
                f" {exc.set_size} <= {inv_min_prob}",
                exc.set_size,
            ) from exc
        except InfeasibleCostError as exc:
            raise InfeasibleSoftConstraintError(
                "Greedy construction does not satisfy soft constraint, meaning no"
                f" improviser can. Maximum percentage of words satisfying soft constraint was {1 - exc.best_cost}.",
                (1 - exc.best_cost),
            ) from exc
