""" Contains the LQCI and MELQCI classes, which act as improvisers
for the Labelled Quantitative Control Improvisation problem and Maximum Entropy
Labelled Quantitiative Control Improvisation problem.
"""

from __future__ import annotations

# Initialize Logger
import logging

logger = logging.getLogger(__name__)

import time
import math
import warnings
import random
from numbers import Rational

import cvxpy as cp
import numpy as np
from multiprocess import Pool

from citoolkit.improvisers.improviser import (
    Improviser,
    InfeasibleImproviserError,
    InfeasibleCostError,
    InfeasibleLabelRandomnessError,
    InfeasibleWordRandomnessError,
)
from citoolkit.specifications.spec import ExactSpec
from citoolkit.costfunctions.cost_func import ExactCostFunc
from citoolkit.labellingfunctions.labelling_func import ExactLabellingFunc


class _LQCIBase(Improviser):
    """The base class for the LQCI class and the MELQCI class.

        When this class is intialized, all the label/cost class specs are
        created and stored in the class_specs dictionary attribute which matches
        all (label,cost) tuples to the spec that recognizes words with that label
        and cost.

        All child classes must then initialize two list attributes: class_keys
        which contains a (label,cost) tuple for every label/cost class to be sampled
        from, and class_probabilities which contains the probability for the label/cost
        class at the same index in class_keys.

        :param hard_constraint: A specification that must accept all improvisations.
        :param cost_func: A cost function that must associate a rational cost
            with all improvisations.
        :param label_func: A labelling function that must associate a label with all
            improvisations.
        :param length_bounds: A tuple containing lower and upper bounds on the length
            of a generated word.
    `   :param num_threads: The number of threads to use in parameterization computation.
        :param lazy: Whether or not to lazily initialize the improvizer.
    """

    def __init__(
        self,
        hard_constraint: ExactSpec,
        cost_func: ExactCostFunc,
        label_func: ExactLabellingFunc,
        length_bounds: tuple[int, int],
        num_threads: int,
        lazy: bool,
    ) -> None:
        ## Initialization ##
        self.length_bounds = length_bounds
        self.lazy = lazy
        self.labels = label_func.labels
        self.costs = cost_func.costs

        self.class_specs = {}
        self.class_keys = None
        self.class_probabilities = None

        ## Function Decomposition ##
        logger.info("Beginning function decomposition and abstract spec construction.")

        label_specs = label_func.decompose()
        cost_specs = cost_func.decompose()

        class_ids = [
            (label, cost) for label in label_func.labels for cost in cost_func.costs
        ]

        for class_id in class_ids:
            label, cost = class_id

            # Create abstract spec
            self.class_specs[class_id] = (
                hard_constraint & label_specs[label] & cost_specs[cost]
            )

        logger.info("Function decomposition and abstract spec construction completed.")

        if not lazy:
            ## Spec Construction ##
            start_time = time.time()
            logger.info(
                "Beginning explicit compound Spec construction. Using %d thread(s).",
                num_threads,
            )

            # Compute spec for each label/cost class.
            if num_threads <= 1:
                cpu_time = -1

                for class_id in class_ids:
                    label, cost = class_id

                    # Create abstract spec
                    self.class_specs[class_id] = (
                        hard_constraint & label_specs[label] & cost_specs[cost]
                    )

                    # Try to make the spec explicit if possible
                    try:
                        self.class_specs[class_id] = self.class_specs[
                            class_id
                        ].explicit()
                    except NotImplementedError:
                        pass
            else:
                # Multiple threads, so create wrapper and thread pool and map abstract specs before
                # resaving explicit specs.

                with Pool(num_threads) as pool:
                    # Helper function for pool.map
                    def explicit_wrapper(class_id):
                        process_start_time = time.process_time()

                        label, cost = class_id

                        # Create abstract spec
                        spec = hard_constraint & label_specs[label] & cost_specs[cost]

                        # Try to make the spec explicit if possible
                        try:
                            spec = spec.explicit()
                        except NotImplementedError:
                            pass

                        return (
                            class_id,
                            spec,
                            time.process_time() - process_start_time,
                        )

                    pool_output = pool.map(explicit_wrapper, class_ids)

                    # Extract relevant info from pool_output
                    cpu_time = sum([runtime for _, _, runtime in pool_output])

                    self.class_specs = {
                        class_id: class_spec for class_id, class_spec, _ in pool_output
                    }

            wall_time = time.time() - start_time
            logger.info(
                "Explicit compound Spec construction completed. Wallclock Runtime: %.4f"
                " CPU Runtime: %.4f",
                wall_time,
                cpu_time,
            )

            ## Spec Counting ##
            # Count the language size for each spec if precount is enabled.
            start_time = time.time()
            logger.info(
                "Beginning language size counting. Using %d thread(s).", num_threads
            )

            if num_threads <= 1:
                cpu_time = -1
                # 1 thread, so compute all sizes iteratively.
                for spec in self.class_specs.values():
                    spec.language_size(*self.length_bounds)
            else:
                # Multiple threads, so create wrapper and thread pool and map specs before
                # resaving specs containing cached language sizes.

                with Pool(num_threads) as pool:
                    # Sort class spec keys for reproducability
                    class_ids = sorted(self.class_specs.keys())

                    # Helper function for pool.map
                    def count_wrapper(class_id):
                        process_start_time = time.process_time()
                        spec = self.class_specs[class_id]

                        spec.language_size(*self.length_bounds)

                        return (
                            class_id,
                            spec,
                            time.process_time() - process_start_time,
                        )

                    pool_output = pool.map(count_wrapper, class_ids)

                    # Extract relevant info from pool_output
                    cpu_time = sum([runtime for _, _, runtime in pool_output])

                    self.class_specs = {
                        class_id: class_spec for class_id, class_spec, _ in pool_output
                    }

            wall_time = time.time() - start_time
            logger.info(
                "Language size counting completed. Wallclock Runtime: %.4f CPU Runtime:"
                " %.4f",
                wall_time,
                cpu_time,
            )

    def improvise(self, seed:int =None) -> Any:
        """Improvise a single word. Base class must populate self.class_probabilities
        before this method is called.

        :returns: A single improvised word.
        :raises RuntimeError: If this function is called before the improviser is parameterized.
        """
        if (self.class_probabilities is None) or (self.class_keys is None):
            raise RuntimeError(
                "Improvise function was called without first computing"
                " self.class_probabilities or self.class_keys. Did you forget to call"
                " parameterize?"
            )

        # Set seed if specified
        if seed is not None:
            random.seed(seed)

        target_class = random.choices(
            population=self.class_keys, weights=self.class_probabilities, k=1
        )[0]

        sample = self.class_specs[target_class].sample(
            *self.length_bounds, seed=random.getrandbits(32)
        )

        return sample


class LQCI(_LQCIBase):
    """An improviser for the Labelled Quantitative Control Improvisation (LQCI) problem.

        :param hard_constraint: A specification that must accept all improvisations.
        :param cost_func: A cost function that must associate a rational cost
            with all improvisations.
        :param label_func: A labelling function that must associate a label with all
            improvisations.
        :param length_bounds: A tuple containing lower and upper bounds on the length
            of a generated word.
    `   :param num_threads: The number of threads to use in parameterization computation.
        :param lazy: Whether or not to lazily initialize the improvizer.
        :raises ValueError: If passed parameters are not well formed.
    """

    def __init__(
        self,
        hard_constraint: ExactSpec,
        cost_func: ExactCostFunc,
        label_func: ExactLabellingFunc,
        length_bounds: tuple[int, int],
        num_threads: int = 1,
        lazy: bool = False,
    ) -> None:
        # Checks that parameters are well formed.
        if not isinstance(hard_constraint, ExactSpec):
            raise ValueError(
                "The hard_constraint parameter must be a member of the ExactSpec class."
            )

        if not isinstance(cost_func, ExactCostFunc):
            raise ValueError(
                "The cost_func parameter must be a member of the ExactCostFunc class."
            )

        if not isinstance(label_func, ExactLabellingFunc):
            raise ValueError(
                "The label_func parameter must be a member of the ExactLabellingFunc"
                " class."
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

        if len(label_func.labels) == 0:
            raise InfeasibleImproviserError(
                "This problem has no labels and therefore no improvisations."
            )

        if len(cost_func.costs) == 0:
            raise InfeasibleImproviserError(
                "This problem has no costs and therefore no improvisations."
            )

        # Initialize LQCI base class.
        super().__init__(
            hard_constraint,
            cost_func,
            label_func,
            length_bounds,
            num_threads=num_threads,
            lazy=lazy,
        )

        logger.info("LQCI Improviser successfully initialized")

    def parameterize(
        self,
        cost_bound: float,
        label_prob_bounds: tuple[float, float],
        word_prob_bounds: dict[str, tuple[float, float]],
        num_threads: int = 1,
    ) -> None:
        """Parameterize this improviser, computing an improvising distribution if possible and throwing
            an exception otherwise.

            :param cost_bound: The maximum allowed expected cost for our improviser.
            :param label_prob_bounds: A tuple containing lower and upper bounds on the
                marginal probability with which we can generate a word with a particular label.
            :param word_prob_bounds: A dictionary mapping each label in label_func to a tuple. Each
                tuple contains a lower and upper bound on the conditional probability of selecting
                a word with the associated label conditioned on the fact that we do select a word
                with label.
        `   :param num_threads: The number of threads to use in parameterization computation.
            :raises ValueError: If passed parameters are not well formed.
            :raises InfeasibleImproviserError: If the resulting improvisation problem is not feasible.
        """
        # Checks that parameters are well formed.
        if cost_bound < 0:
            raise ValueError("The cost_bound parameter must be a number >= 0.")

        if (
            (len(label_prob_bounds) != 2)
            or (label_prob_bounds[0] < 0)
            or (label_prob_bounds[0] > label_prob_bounds[1])
            or (label_prob_bounds[1] > 1)
        ):
            raise ValueError(
                "The label_prob_bounds parameter should contain two floats, with 0 <="
                " label_prob_bounds[0] <= label_prob_bounds[1] <= 1."
            )

        for label in self.labels:
            target_prob_bounds = word_prob_bounds[label]

            if label not in word_prob_bounds.keys():
                raise ValueError(
                    "The word_prob_bounds parameter is missing conditional probability"
                    f" bounds for the label '{label}'."
                )

            if (
                (len(target_prob_bounds) != 2)
                or (target_prob_bounds[0] < 0)
                or (target_prob_bounds[0] > target_prob_bounds[1])
                or (target_prob_bounds[1] > 1)
            ):
                raise ValueError(
                    "The word_prob_bounds parameter should contain two floats, with 0"
                    f" <= word_prob_bounds[{label}][0] <= word_prob_bounds[{label}][1]"
                    " <= 1."
                )

        if label_prob_bounds[1] == 0:
            raise InfeasibleImproviserError(
                "No label can be assigned probability with label_prob_bounds[1] == 0."
            )

        start_time = time.time()
        logger.info("Beginning LQCI distribution calculation.")

        ## Distribution Computation ##
        feasible_labels = set()

        for label in self.labels:
            for cost in self.costs:
                if (
                    self.class_specs[(label, cost)].language_size(*self.length_bounds)
                    > 0
                ):
                    feasible_labels.add(label)
                    break
            if label_prob_bounds[0] > 0 and label not in feasible_labels:
                raise InfeasibleLabelRandomnessError(
                    f"No strings are labelled with label '{label}', but"
                    " label_prob_bounds[0] > 0.",
                    0,
                )

        # Initialize conditional weights to alpha for each word and sum up current marginal label probabilities.
        conditional_weights = {}

        for class_id in self.class_specs:
            label, cost = class_id
            # If alpha is 0, we can just set everything to 0. Otherwise determine probability based on class count.
            if word_prob_bounds[label][0] == 0:
                conditional_weights[class_id] = 0
            else:
                class_count = self.class_specs[class_id].language_size(
                    *self.length_bounds
                )
                conditional_weights[class_id] = class_count * word_prob_bounds[label][0]

        label_sum_probs = {
            label: sum(
                [prob for ((l, _), prob) in conditional_weights.items() if l == label]
            )
            for label, _ in self.class_specs
        }

        # Check if we've already broken probability bounds.
        for label, label_prob in label_sum_probs.items():
            if label_prob > 1:
                raise InfeasibleWordRandomnessError(
                    f"Assigning minimum probability ({word_prob_bounds[label][0]})"
                    f" to all words with label '{label}' results in a probability of"
                    f" '{label_prob}', which is greater than 1.",
                    None,
                )

        # Add probabilites to appropriate classes (up to beta per word) without assigning more than 1 over each label.
        if self.lazy and num_threads > 1 and len(self.labels) > 1:
            # Probabilities are not yet computed and we have several threads available and several labels to compute. Advantageous to multithread.
            with Pool(num_threads) as pool:
                # Helper function that counts buckets for a label and assigns probability.
                def assign_beta(func_input):
                    label, label_specs, cond_label_weights = func_input

                    for cost in sorted(self.costs):
                        # If we assigned all probability, terminate early
                        if sum(cond_label_weights.values()) == 1:
                            break

                        class_count = label_specs[cost].language_size(
                            *self.length_bounds
                        )

                        # Assign at least beta probability to all words in this bucket, or
                        # all remaining probability (plus what we have already assigned), whichever is smaller.
                        cond_label_weights[cost] = min(
                            class_count * word_prob_bounds[label][1],
                            1
                            - sum(cond_label_weights.values())
                            + cond_label_weights[cost],
                        )

                    return (label, label_specs, cond_label_weights)

                # Create helper function inputs and run them through assign_beta
                func_inputs = []
                for label in self.labels:
                    label_specs = {
                        c: (spec)
                        for (l, c), spec in self.class_specs.items()
                        if l == label
                    }
                    cond_label_weights = {
                        c: weight
                        for (l, c), weight in conditional_weights.items()
                        if l == label
                    }
                    func_inputs.append((label, label_specs, cond_label_weights))

                pool_output = pool.map(assign_beta, func_inputs)
                pool_dict = {
                    label: (specs, weights) for label, specs, weights in pool_output
                }

                # Parse pool outputs back into appropriate variables
                for class_id in self.class_specs:
                    label, cost = class_id
                    self.class_specs[class_id] = pool_dict[label][0][cost]

                conditional_weights = {
                    (l, b): pool_dict[l][1][b] for l, b in self.class_specs
                }

                # Update label_sum_probs
                label_sum_probs = {
                    label: sum(
                        [
                            prob
                            for (l, _), prob in conditional_weights.items()
                            if l == label
                        ]
                    )
                    for label in self.labels
                }

        else:
            # Probabilities already computed or only one thread, so no point in multithreading.
            for label in self.labels:
                for cost in sorted(self.costs):
                    # If we assigned all probability, terminate early
                    if label_sum_probs[label] == 1:
                        break

                    class_id = (label, cost)
                    class_count = self.class_specs[class_id].language_size(
                        *self.length_bounds
                    )

                    conditional_weights[class_id] = min(
                        class_count * word_prob_bounds[label][1],
                        1 - label_sum_probs[label] + conditional_weights[class_id],
                    )

                    # Update label sum probability
                    label_sum_probs[label] = sum(
                        [
                            prob
                            for ((l, _), prob) in conditional_weights.items()
                            if l == label
                        ]
                    )

        # Check if we've now broken probability bounds
        for label, label_prob in label_sum_probs.items():
            if label_prob != 1:
                raise InfeasibleWordRandomnessError(
                    "Assigning maximum probability"
                    f" ({(word_prob_bounds[label][1])}) to all words with label"
                    f" '{label}' results in a probability of '{label_prob}', which"
                    " is less than 1.",
                    None,
                )

        # Calculate conditional expected costs
        conditional_costs = {}
        for label in self.labels:
            conditional_costs[label] = sum(
                [conditional_weights[(label, cost)] * cost for cost in self.costs]
            )

        # Calculate marginal weights.
        marginal_weights = {}

        # Calculate u, the number of labels that can get maximum probability
        if label_prob_bounds[1] == label_prob_bounds[0]:
            # In this case all labels must get the same probability, which will be handled by case 3 of the if statement below.
            u = 0
        else:
            u = math.floor(
                (1 - len(self.labels) * label_prob_bounds[0])
                / (label_prob_bounds[1] - label_prob_bounds[0])
            )

        for label_iter, label in enumerate(
            sorted(self.labels, key=lambda x: conditional_costs[x])
        ):
            if label_iter < u:
                marginal_weights[label] = label_prob_bounds[1]
            elif label_iter == u:
                marginal_weights[label] = (
                    1
                    - label_prob_bounds[1] * u
                    - label_prob_bounds[0] * (len(self.labels) - u - 1)
                )
            else:
                marginal_weights[label] = label_prob_bounds[0]

        expected_cost = sum(
            [
                marginal_weights[label] * conditional_costs[label]
                for label in self.labels
            ]
        )

        logger.info("Calculated Distribution Expected Cost: %.4f", expected_cost)

        # Store improvisation values.
        self.class_keys = [
            (label, cost)
            for label in sorted(self.labels)
            for cost in sorted(self.costs)
        ]
        self.class_probabilities = [
            marginal_weights[label] * conditional_weights[(label, cost)]
            for label, cost in self.class_keys
        ]

        wall_time = time.time() - start_time
        logger.info(
            "LQCI distribution calculation completed. Wallclock Runtime: %.4f",
            wall_time,
        )

        ## Cleanup ##

        # Checks that this improviser is feasible. If not raise an InfeasibleImproviserError.
        if len(feasible_labels) < (1 / label_prob_bounds[1]) or (
            label_prob_bounds[0] != 0
            and len(feasible_labels) > (1 / label_prob_bounds[0])
        ):
            if label_prob_bounds[0] == 0:
                inv_min_label_prob = float("inf")
            else:
                inv_min_label_prob = 1 / label_prob_bounds[0]

            raise InfeasibleLabelRandomnessError(
                "Violation of condition 1/label_prob_bounds[1] <= len(feasible_labels)"
                " <= 1/label_prob_bounds[0]. Instead,"
                f" {1 / label_prob_bounds[1]} <= {len(feasible_labels)} <="
                f" {inv_min_label_prob}",
                len(feasible_labels),
            )

        if expected_cost > cost_bound:
            raise InfeasibleCostError(
                "Greedy construction does not satisfy cost_bound, meaning no"
                f" improviser can. Minimum expected cost was {expected_cost}.",
                expected_cost,
            )

        logger.info("LQCI Improviser successfully parameterized")


class MELQCI(_LQCIBase):
    """An improviser for the Maximum Entropy Labelled Quantitative Control Improvisation (MELQCI) problem.

    :param hard_constraint: A specification that must accept all improvisations.
    :param cost_func: A cost function that must associate a rational cost
        with all improvisations.
    :param label_func: A labelling function that must associate a label with all
        improvisations.
    :param length_bounds: A tuple containing lower and upper bounds on the length
        of a generated word.
    :param num_threads: The number of threads to use in parameterization computation.
    :raises ValueError: If passed parameters are not well formed.
    """

    def __init__(
        self,
        hard_constraint: ExactSpec,
        cost_func: ExactCostFunc,
        label_func: ExactLabellingFunc,
        length_bounds: tuple[int, int],
        num_threads: int = 1,
    ) -> None:
        # Checks that parameters are well formed.
        if not isinstance(hard_constraint, ExactSpec):
            raise ValueError(
                "The hard_constraint parameter must be a member of the ExactSpec class."
            )

        if not isinstance(cost_func, ExactCostFunc):
            raise ValueError(
                "The cost_func parameter must be a member of the ExactCostFunc class."
            )

        if not isinstance(label_func, ExactLabellingFunc):
            raise ValueError(
                "The label_func parameter must be a member of the ExactLabellingFunc"
                " class."
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

        if len(label_func.labels) == 0:
            raise InfeasibleImproviserError(
                "This problem has no labels and therefore no improvisations."
            )

        if len(cost_func.costs) == 0:
            raise InfeasibleImproviserError(
                "This problem has no costs and therefore no improvisations."
            )

        # Initialize LQCI base class.
        super().__init__(
            hard_constraint,
            cost_func,
            label_func,
            length_bounds,
            num_threads=num_threads,
            lazy=False,
        )

        # Initialize reporting variables to None
        self.status = None
        self.entropy = None

        logger.info("MELQCI Improviser successfully initialized")

    def parameterize(
        self,
        cost_bound: float,
        label_prob_bounds: tuple[float, float],
        num_threads: int = 1,
    ) -> None:
        """Parameterize the MELQCI improviser.

        :param cost_bound: The maximum allowed expected cost for our improviser.
        :param label_prob_bounds: A tuple containing lower and upper bounds on the
            marginal probability with which we can generate a word with a particular label.
        :param num_threads: The number of threads to use in parameterization computation.
        :raises ValueError: If passed parameters are not well formed.
        :raises InfeasibleImproviserError: If the resulting improvisation problem is not feasible.
        """
        # Checks that parameters are well formed.
        if cost_bound < 0:
            raise ValueError("The cost_bound parameter must be a number >= 0.")

        if (
            (len(label_prob_bounds) != 2)
            or (label_prob_bounds[0] < 0)
            or (label_prob_bounds[0] > label_prob_bounds[1])
            or (label_prob_bounds[1] > 1)
        ):
            raise ValueError(
                "The label_prob_bounds parameter should contain two floats, with 0 <="
                " label_prob_bounds[0] <= label_prob_bounds[1] <= 1."
            )

        if label_prob_bounds[1] == 0:
            raise InfeasibleImproviserError(
                "No label can be assigned probability with label_prob_bounds[1] == 0."
            )

        start_time = time.time()
        logger.info("Beginning MELQCI distribution calculation.")

        # Extract class sizes from class specs.
        cost_class_sizes = {}

        for label in self.labels:
            for cost in self.costs:
                cost_class_sizes[(label, cost)] = self.class_specs[
                    (label, cost)
                ].language_size(*self.length_bounds)

        # Create optimization variables and constants. Assuming n labels and m costs, the variable at position
        # x*m + y represents the probability allocated to words with label x and cost y.
        x = cp.Variable(len(self.labels) * len(self.costs), nonneg=True)

        cost_class_sizes_vector = [
            max(1, cost_class_sizes[(label, cost)])
            for label in sorted(self.labels)
            for cost in sorted(self.costs)
        ]

        entropy_equation = -cp.sum(
            cp.entr(x) + cp.multiply(x, np.log(cost_class_sizes_vector))
        )

        objective = cp.Minimize(entropy_equation)

        # Create constraints list
        constraints = []

        # (C1) Satisfaction of the cost bound
        expected_cost_equation = cp.sum(
            cp.multiply(x, sorted(self.costs) * len(self.labels))
        )
        constraints.append(expected_cost_equation <= cost_bound)

        # (C2) - (C3) Randomness over Labels lower and upper bound
        for label_iter, label in enumerate(sorted(self.labels)):
            label_prob_equation = cp.sum(
                cp.multiply(
                    x,
                    np.concatenate(
                        [
                            (
                                [1] * len(self.costs)
                                if i == label_iter
                                else [0] * len(self.costs)
                            )
                            for i in range(len(self.labels))
                        ]
                    ),
                )
            )

            constraints.append(label_prob_equation >= label_prob_bounds[0])
            constraints.append(label_prob_equation <= label_prob_bounds[1])

        # (C4) Non negative probability
        constraints.append(x >= 0)

        # (C5) Probability distribution sums to 1
        constraints.append(cp.sum(x) == 1)

        # (C6) Empty Cost Classes have 0 probability
        for label_iter, label in enumerate(sorted(self.labels)):
            for cost_iter, cost in enumerate(sorted(self.costs)):
                if cost_class_sizes[(label, cost)] == 0:
                    empty_cost_class_vector = [0] * (len(self.labels) * len(self.costs))
                    empty_cost_class_vector[
                        label_iter * len(self.costs) + cost_iter
                    ] = 1
                    constraints.append(cp.multiply(x, empty_cost_class_vector) == 0)

        # Create and solve problem
        prob = cp.Problem(objective, constraints)

        logger.info("Solving MELQCI distribution optimization problem.")
        result = prob.solve()

        wall_time = time.time() - start_time
        logger.info(
            "MELQCI distribution calculation completed. Wallclock Runtime: %.4f",
            wall_time,
        )

        # Check if problem is infeasible. If so, raise an InfeasibleImproviserError.
        if "infeasible" in prob.status:
            raise InfeasibleImproviserError(
                "Optimization problem infeasible. For more details run the associated"
                " non ME problem."
            )

        # Check if problem result was not infeasible, but also not optimal. If so, raise a warning.
        if prob.status != "optimal":
            warnings.warn(f"Got unexpected value '{prob.status}' as optimizer output.")

        # Store improvisation variables
        self.class_keys = [
            (label, cost)
            for label in sorted(self.labels)
            for cost in sorted(self.costs)
        ]
        self.class_probabilities = list(x.value)
        self.status = prob.status
        self.entropy = -1 * result

        logger.info("Optimizer Return States: %s", str(self.status))
        logger.info("Calculated Distribution Entropy: %.4f", self.entropy)

        # Set all sorted_cost_class_weights that have empty cost classes to absolutely zero instead of very near 0.
        for label_iter, label in enumerate(sorted(self.labels)):
            for cost_iter, cost in enumerate(sorted(self.costs)):
                if cost_class_sizes[(label, cost)] == 0:
                    self.class_probabilities[
                        label_iter * len(self.costs) + cost_iter
                    ] = 0
