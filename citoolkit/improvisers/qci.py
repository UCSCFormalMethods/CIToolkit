""" Contains the QCI class, which acts as an improviser
for the Quantitative Control Improvisation problem.
"""

from __future__ import annotations

# Initialize Logger
import logging

logger = logging.getLogger(__name__)

from citoolkit.improvisers.lqci import LQCI
from citoolkit.improvisers.improviser import (
    InfeasibleImproviserError,
    InfeasibleRandomnessError,
    InfeasibleLabelRandomnessError,
    InfeasibleWordRandomnessError,
)
from citoolkit.specifications.spec import ExactSpec
from citoolkit.costfunctions.cost_func import ExactCostFunc
from citoolkit.labellingfunctions.labelling_func import TrivialLabellingFunc


class QCI(LQCI):
    """An improviser for the Quantitative Control Improvisation (QCI) problem.

    :param hard_constraint: A specification that must accept all improvisations
    :param cost_func: A cost function that must associate a rational cost
        with all improvisations.
    :param length_bounds: A tuple containing lower and upper bounds on the length
        of a generated word.
    :param num_threads: The number of threads to use in parameterization computation.
    :param lazy: Whether or not to lazily initialize the improvizer.
    :raises ValueError: If passed parameters are not well formed.
    """

    def __init__(
        self,
        hard_constraint: ExactSpec,
        cost_func: ExactCostFunc,
        length_bounds: tuple[int, int],
        num_threads: int = 1,
        lazy:bool =False,
    ):
        # Checks that parameters are well formed
        if not isinstance(hard_constraint, ExactSpec):
            raise ValueError(
                "The hard_constraint parameter must be a member of the ExactSpec class."
            )

        if not isinstance(cost_func, ExactCostFunc):
            raise ValueError(
                "The cost_func parameter must be a member of the ExactCostFunc class."
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
        label_func = TrivialLabellingFunc()

        # Initialize associated LQCI problem
        logger.info("Generalizing QCI problem to equivalent LQCI problem.")
        super().__init__(
            hard_constraint,
            cost_func,
            label_func,
            length_bounds,
            num_threads=num_threads,
            lazy=lazy,
        )

    def parameterize(
        self, cost_bound: float, prob_bounds: tuple[float, float], num_threads: int = 1
    ) -> None:
        """Parameterize the QCI improviser.

        :param cost_bound: The maximum allowed expected cost for our improviser.
        :param prob_bounds: A tuple containing lower and upper bounds on the
            probability with which we can generate a word.
        :param num_threads: The number of threads to use in parameterization computation.
        :raises ValueError: If passed parameters are not well formed.
        :raises InfeasibleImproviserError: If the resulting improvisation problem is not feasible.
        """
        # Checks that parameters are well formed
        if cost_bound < 0:
            raise ValueError("The cost_bound parameter must be a number >= 0.")

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
