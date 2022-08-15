from __future__ import annotations

# Initialize Logger
import logging

logger = logging.getLogger(__name__)

import time
import random
import math

from multiprocess import Pool

from citoolkit.improvisers.improviser import (
    Improviser,
    InfeasibleImproviserError,
    InfeasibleCostError,
    InfeasibleLabelRandomnessError,
    InfeasibleWordRandomnessError,
)
from citoolkit.specifications.spec import ApproxSpec
from citoolkit.costfunctions.cost_func import ApproxCostFunc
from citoolkit.labellingfunctions.labelling_func import ApproxLabelFunc


class _ALQCIBase(Improviser):
    """The base class for the ALQCI class.

    When this class is intialized, all the label/cost class specs are
    created and stored in the class_specs dictionary attribute which matches
    all (label,cost) tuples to the spec that recognizes words with that label
    and cost.

    If lazy is set to False, then all label/cost class specs are also
    counted.

    All child classes must then initialize two list attributes: class_keys
    which contains a (label,cost) tuple for every label/cost class to be sampled
    from, and class_probabilities which contains the probability for the label/cost
    class at the same index in class_keys.

    :param hard_constraint: A specification that must accept all improvisations.
    :param cost_func: A cost function that must associate a natural number cost
        with all improvisations.
    :param label_func: A labelling function that must associate a label with all
        improvisations.
    :param bucket_ratio: The ratio between successive buckets. Must be greater than 1.
    :param counting_tol: The desired counting tolerance.
    :param sampling_tol: The desired sampling tolerance.
    :param conf: The desired confidence in the counting/sampling. Used with the number
        of feasible buckets to determine counting_conf, which is the confidence we need
        from each bucket we actually count.
    :param seed: Random seed for reproducible results.
    :param num_threads: Number of threads to be used when computing improviser.
    :param verbose: If verbose is enabled, information will be printed to STDOUT.
    :param lazy: Whether or not to compute lazily. If true, all spec counts will
        be computed after spec construction.
    """

    def __init__(
        self,
        hard_constraint: ApproxSpec,
        cost_func: ApproxCostFunc,
        label_func: ApproxLabelFunc,
        bucket_ratio: float,
        counting_tol: float,
        sampling_tol: float,
        conf: float,
        seed: int,
        num_threads: int,
        lazy: bool,
    ) -> None:
        ## Initialization ##
        # Save random state and seed new state.
        if seed is None:
            seed = random.getrandbits(32)

        random.seed(seed)

        # Save relevant parameters
        self.lazy = lazy
        self.labels = label_func.labels
        self.sampling_tol = sampling_tol
        self.counting_tol = counting_tol

        self.class_specs = {}
        self.class_keys = None
        self.class_probabilities = None

        # Compute bucket dimensions and total number of buckets.
        last_bucket_max_cost = 0
        self.bucket_bounds_list = [(0, 0)]
        while last_bucket_max_cost < cost_func.max_cost:
            min_bucket_cost = last_bucket_max_cost + 1
            max_bucket_cost = min(
                math.floor(min_bucket_cost * bucket_ratio), cost_func.max_cost
            )
            last_bucket_max_cost = max_bucket_cost

            self.bucket_bounds_list.append((min_bucket_cost, max_bucket_cost))

        self.num_buckets = len(self.bucket_bounds_list)

        ## Spec Construction ##
        # Compute spec for each label/cost class.
        start_time = time.time()
        logger.info(
            "Beginning compound Spec construction. Using %d thread(s).", num_threads
        )

        class_ids = set()

        for label in label_func.labels:
            for bucket_iter, bucket_bounds in enumerate(self.bucket_bounds_list):
                class_ids.add((label, bucket_iter))

        for class_id in sorted(class_ids):
            label, bucket_iter = class_id
            bucket_bounds = self.bucket_bounds_list[bucket_iter]

            # Create abstract spec
            self.class_specs[class_id] = (
                hard_constraint
                & label_func.realize(label)
                & cost_func.realize(*bucket_bounds)
            )

            # Try to make the spec explicit if possible
            try:
                self.class_specs[class_id] = self.class_specs[class_id].explicit()
            except NotImplementedError:
                pass

        wall_time = time.time() - start_time
        logger.info(
            "Compound Spec construction completed. Wallclock Runtime: %.4f", wall_time
        )

        ## Counting Confidence Calculation ##
        # Determine counting confidence needed
        feasible_buckets = sum([spec.feasible for spec in self.class_specs.values()])

        self.counting_conf = 1 - math.pow((1 - conf), 1 / feasible_buckets)

        logger.info("Total Buckets: %d", len(self.class_specs))
        logger.info("Feasible Buckets: %d", feasible_buckets)
        logger.info(
            "Desired Confidence %.4f requires counting confidence of %.4f",
            conf,
            self.counting_conf,
        )

        ## Spec Counting ##
        if not lazy:
            # Count the language size for each spec if precount is enabled.
            start_time = time.time()
            logger.info(
                "Beginning language size counting. Using %d thread(s).", num_threads
            )

            if num_threads <= 1:
                cpu_time = -1
                # 1 thread, so compute all sizes iteratively.
                for class_id in sorted(class_ids):
                    spec = self.class_specs[class_id]
                    spec.language_size(
                        tolerance=self.counting_tol,
                        confidence=self.counting_conf,
                        seed=random.getrandbits(32),
                    )
            else:
                # Multiple threads, so create wrapper and thread pool and map specs before
                # resaving specs containing cached language sizes. Only use the internal
                # boolean formula specs to avoid Z3 context memory blowup.

                with Pool(num_threads) as pool:
                    # Sort class spec keys for reproducability
                    class_ids = sorted(self.class_specs.keys())

                    # Get a random seed for each language size calculation
                    random_seeds = {}
                    for class_id in class_ids:
                        random_seeds[class_id] = random.getrandbits(32)

                    # Assemble wrapper inputs
                    wrapper_inputs = [
                        (
                            class_id,
                            self.class_specs[class_id].bool_formula_spec,
                            self.counting_tol,
                            self.counting_conf,
                            random_seeds[class_id],
                        )
                        for class_id in class_ids
                    ]

                    # Helper function for pool.map
                    def count_wrapper(input):
                        class_id, spec, counting_tol, counting_conf, random_seed = input
                        process_start_time = time.process_time()

                        spec.language_size(
                            tolerance=counting_tol,
                            confidence=counting_conf,
                            seed=random_seed,
                        )

                        return (
                            class_id,
                            spec,
                            time.process_time() - process_start_time,
                        )

                    pool_output = pool.map(count_wrapper, wrapper_inputs)

                    # Extract relevant info from pool_output
                    cpu_time = sum([runtime for _, _, runtime in pool_output])

                    # Reinsert all bool specs
                    for output in pool_output:
                        class_id, bool_spec, _ = output
                        self.class_specs[class_id].set_bool_formula_spec(bool_spec)

                wall_time = time.time() - start_time
                logger.info(
                    "Language size counting completed. Wallclock Runtime: %.4f CPU"
                    " Runtime: %.4f",
                    wall_time,
                    cpu_time,
                )

    def improvise(self, seed: int = None) -> Any:
        """Improvise a single word. Base class must populate self.class_probabilities
        before this method is called.

        :returns: A single improvised word.
        """
        if (self.class_probabilities is None) or (self.class_keys is None):
            raise Exception(
                "Improvise function was called without first computing"
                " self.class_probabilities or self.class_keys."
            )

        # Cache the random state, set it to the seed, pick a target class, sample, and finally, restore the random state.
        if seed is None:
            seed = random.getrandbits(32)

        random.seed(seed)

        target_class = random.choices(
            population=self.class_keys, weights=self.class_probabilities, k=1
        )[0]

        sample = self.class_specs[target_class].sample(
            tolerance=self.sampling_tol, seed=random.getrandbits(32)
        )

        return sample


class ALQCI(_ALQCIBase):
    """An improviser for the Approximate Labelled Quantitative Control Improvisation problem.

    When this class is intialized, all the label/cost class specs are
    created and stored in the class_specs dictionary attribute which matches
    all (label,cost) tuples to the spec that recognizes words with that label
    and cost.

    If lazy is set to False, then all label/cost class specs are also
    counted.

    :param hard_constraint: A specification that must accept all improvisations.
    :param cost_func: A cost function that must associate a natural number cost
        with all improvisations.
    :param label_func: A labelling function that must associate a label with all
        improvisations.
    :param bucket_ratio: The ratio between successive buckets. Must be greater than 1.
    :param counting_tol: The desired counting tolerance. Must be greater than 0.
    :param sampling_tol: The desired sampling tolerance. Must be greater than 0.
    :param conf: The desired confidence in the counting/sampling. Used with the number
        of feasible buckets to determine counting_conf, which is the confidence we need
        from each bucket we actually count. The guarantees for the improviser
        will hold with confidence (1 - conf). Must be in the range (0,1)
    :param seed: Random seed for reproducible results.
    :param num_threads: Number of threads to be used when computing improviser.
    :param verbose: If verbose is enabled, information will be printed to STDOUT.
    :param lazy: Whether or not to count lazily. If true, all spec counts will
        be computed after spec construction. If false, all spec counts will be computed when needed.
        If None, the improviser will decide based on a heuristic.
    """

    def __init__(
        self,
        hard_constraint: ApproxSpec,
        cost_func: ApproxCostFunc,
        label_func: ApproxLabelFunc,
        bucket_ratio: float,
        counting_tol: float,
        sampling_tol: float,
        conf: float,
        seed=None,
        num_threads: int = 1,
        lazy: bool = False,
    ) -> None:
        ## Initialization ##
        # Checks that parameters are well formed.
        if not isinstance(hard_constraint, ApproxSpec):
            raise ValueError(
                "The hard_constraint parameter must be a member of the ApproxSpec"
                " class."
            )

        if not isinstance(cost_func, ApproxCostFunc):
            raise ValueError(
                "The cost_func parameter must be a member of the ApproxCostFunc class."
            )

        if not isinstance(label_func, ApproxLabelFunc):
            raise ValueError(
                "The label_func parameter must be a member of the ApproxLabelFunc"
                " class."
            )

        if len(label_func.labels) == 0:
            raise InfeasibleImproviserError(
                "This problem has no labels and therefore no improvisations."
            )

        if bucket_ratio <= 1:
            raise ValueError("The bucket_ratio parameter must be a number >= 1.")

        if counting_tol <= 0:
            raise ValueError("The counting_tol parameter must be a number > 0.")

        if sampling_tol <= 0:
            raise ValueError("The sampling_tol parameter must be a number > 0.")

        if conf <= 0 or conf >= 1:
            raise ValueError("The conf parameter must be a number in the range (0,1).")

        # Set seed if specified
        if seed is not None:
            random.seed(seed)

        # Initialize LQCI base class.
        super().__init__(
            hard_constraint,
            cost_func,
            label_func,
            bucket_ratio,
            counting_tol,
            sampling_tol,
            conf,
            seed,
            num_threads,
            lazy,
        )

        logger.info("LQCI Improviser successfully initialized")

    def parameterize(
        self,
        cost_bound: float,
        label_prob_bounds: tuple[float, float],
        word_prob_bounds: dict[str, tuple[float, float]],
        seed:int =None,
        num_threads: int = 1,
    ) -> None:
        """
        :param cost_bound: The desired maximum cost that should be returned. The expected
            cost of the resulting improvising distribution is guaranteed to be less than
            or equal to (cost_bound * bucket_ratio) with confidence conf.
        :param label_prob_bounds: A tuple containing lower and upper bounds on the
            marginal probability with which we can generate a word with a particular label.
            The marginal probability of each label in the resulting improvising distribution
            is guaranteed to be in this range with confidence conf.
        :param word_prob_bounds: A dictionary mapping each label in label_func to a tuple. Each
            tuple contains a desired lower and upper bound on the conditional probability of selecting
            a word with the associated label conditioned on the fact that we do select a word
            with that label. The conditional probability of each word is guaranteed to be between (inclusive)
            (word_prob_bounds[label][0] / ((1 + counting_tol)**2 (1 + sampling_tol))) and
            (word_prob_bounds[label][1] * ((1 + counting_tol)**2 (1 + sampling_tol)))
        """
        ## Initialization ##
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

        # Set seed if specified
        if seed is not None:
            random.seed(seed)

        ## Distribution Computation ##
        start_time = time.time()
        logger.info("Beginning ALQCI distribution calculation.")

        feasible_labels = set()

        for label in sorted(self.labels):
            for bucket_iter, _ in enumerate(self.bucket_bounds_list):
                if self.class_specs[(label, bucket_iter)].feasible:
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

        for class_id in sorted(self.class_specs):
            label, bucket_iter = class_id
            # If alpha is 0, we can just set everything to 0. Otherwise determine probability based on class count.
            if word_prob_bounds[label][0] == 0:
                conditional_weights[class_id] = 0
            else:
                class_count = self.class_specs[class_id].language_size(
                    tolerance=self.counting_tol,
                    confidence=self.counting_conf,
                    seed=random.getrandbits(32),
                )
                conditional_weights[class_id] = class_count * (
                    word_prob_bounds[label][0] / (1 + self.counting_tol)
                )

        label_sum_probs = {
            label: sum(
                [prob for ((l, _), prob) in conditional_weights.items() if l == label]
            )
            for label, bucket_iter in self.class_specs
        }

        # Check if we've already broken probability bounds.
        for label, label_prob in label_sum_probs.items():
            if label_prob > 1:
                raise InfeasibleWordRandomnessError(
                    f"Assigning minimum probability ({word_prob_bounds[label][0]:.2f})"
                    f" to all words with label '{label}' results in a probability of"
                    f" {label_prob:.2f}, which is greater than 1.",
                    None,
                )

        # Add probabilites to appropriate classes (up to beta per word) without assigning more than 1 over each label.
        if self.lazy and num_threads > 1 and len(self.labels) > 1:
            # Probabilities are not yet computed and we have several threads available and several labels to compute. Advantageous to multithread.
            with Pool(num_threads) as pool:
                # Helper function that counts buckets for a label and assigns probability.
                def assign_beta(func_input):
                    (
                        label,
                        label_specs,
                        cond_label_weights,
                        counting_tol,
                        counting_conf,
                        num_buckets,
                        seed,
                    ) = func_input

                    for bucket_iter in range(num_buckets):
                        # If we assigned all probability, terminate early
                        if sum(cond_label_weights.values()) == 1:
                            break

                        class_count = label_specs[bucket_iter].language_size(
                            tolerance=counting_tol, confidence=counting_conf, seed=seed
                        )

                        # Assign at least beta probability to all words in this bucket (accounting for counting errors), or
                        # all remaining probability (plus what we have already assigned), whichever is smaller.
                        new_cost = min(
                            (1 + counting_tol)
                            * word_prob_bounds[label][1]
                            * class_count,
                            1
                            - sum(cond_label_weights.values())
                            + cond_label_weights[bucket_iter],
                        )

                        cond_label_weights[bucket_iter] = new_cost

                    return (label, label_specs, cond_label_weights)

                # Create helper function inputs and run them through assign_beta
                func_inputs = []
                for label in sorted(self.labels):
                    label_specs = {
                        b: (spec.bool_formula_spec)
                        for (l, b), spec in self.class_specs.items()
                        if l == label
                    }
                    cond_label_weights = {
                        b: weight
                        for (l, b), weight in conditional_weights.items()
                        if l == label
                    }
                    seed = random.getrandbits(32)
                    func_inputs.append(
                        (
                            label,
                            label_specs,
                            cond_label_weights,
                            self.counting_tol,
                            self.counting_conf,
                            self.num_buckets,
                            seed,
                        )
                    )

                pool_output = pool.map(assign_beta, func_inputs)
                pool_dict = {
                    label: (specs, weights) for label, specs, weights in pool_output
                }

                # Parse pool outputs back into appropriate variables
                for class_id in self.class_specs:
                    label, bucket_iter = class_id
                    self.class_specs[class_id].set_bool_formula_spec(
                        pool_dict[label][0][bucket_iter]
                    )

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
            for label in sorted(self.labels):
                for bucket_iter in range(self.num_buckets):
                    # If we assigned all probability, terminate early
                    if label_sum_probs[label] == 1:
                        break

                    class_id = (label, bucket_iter)
                    class_count = self.class_specs[class_id].language_size(
                        tolerance=self.counting_tol,
                        confidence=self.counting_conf,
                        seed=random.getrandbits(32),
                    )

                    conditional_weights[class_id] = min(
                        (1 + self.counting_tol)
                        * word_prob_bounds[label][1]
                        * class_count,
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
                    f"Assigning maximum probability ({word_prob_bounds[label][1]:.2f})"
                    f" to all words with label '{label}' results in a probability of"
                    f" {label_prob:.2f} which is less than 1.",
                    None,
                )

        # Calculate conditional expected costs
        conditional_costs = {}
        for label in self.labels:
            conditional_costs[label] = sum(
                [
                    conditional_weights[(label, bucket_iter)]
                    * self.bucket_bounds_list[bucket_iter][0]
                    for bucket_iter in range(self.num_buckets)
                ]
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

        min_expected_cost = sum(
            [
                marginal_weights[label] * conditional_costs[label]
                for label in self.labels
            ]
        )

        logger.info(
            "Lower Bound on Calculated Distribution Cost: %.4f", min_expected_cost
        )

        wall_time = time.time() - start_time
        logger.info(
            "ALQCI distribution calculation completed. Wallclock Runtime: %.4f",
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
                f" {1 / label_prob_bounds[1]:.2f} <= {len(feasible_labels)} <="
                f" {inv_min_label_prob:.2f}",
                len(feasible_labels),
            )

        if min_expected_cost > cost_bound:
            raise InfeasibleCostError(
                "Greedy construction does not satisfy cost_bound, meaning no"
                f" improviser can. Minimum expected cost was {min_expected_cost:.2f}.",
                min_expected_cost,
            )

        # Store improvisation values.
        self.class_keys = [
            (label, bucket)
            for label in sorted(self.labels)
            for bucket in range(self.num_buckets)
        ]
        self.class_probabilities = [
            marginal_weights[label] * conditional_weights[(label, cost)]
            for label, cost in self.class_keys
        ]
