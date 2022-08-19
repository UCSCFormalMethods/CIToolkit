..  _usage:

Usage
============

The core classes of the ``citoolkit`` are broken into four packages: **Improvisers**, **Specifications**, **Labelling Functions**, and **Cost Functions**. 

Improvisers
**************

The improviser classes are implementations of the algorithms to solve the Control Improvisation problem and its extensions. The Improviser base class has three abstract methods: ``parameterize``, ``improvise``, and ``generator``. An improviser is created with only some of the parameters to the Control Improvisation extension it solves. These parameters are mostly integral to the structure of the improviser, i.e. values which if they were changed would render meaningless any cached progress the improviser possesses. The remaining values are passed as parameters to the ``parameterize`` function, which completes the construction of the improviser, including the actual distribution calculation and determining whether or not the problem in question is feasible given all the constraints
. This split is done to allow partial progress to be saved in the case of an infeasible problem (an example of this strategy is available in the :doc:`quickstart` section). Once the improviser has been created, the ``improvise`` and ``generator`` functions can be used to actually sample from the improvising distribution. ``improvise`` returns a single improvised word while ``generator`` returns an iterable object that continually generates words.

The improvisers provided natively by this library are:

* :mod:`Control Improvisation (CI)<citoolkit.improvisers.ci.CI>`: Outlined in detail in `[Fremont et al. 2017] <https://arxiv.org/abs/1704.06319>`_. Allows for a hard constraint that must always be satisfied, a soft constraint that can be violated within a provided tolerance, and explicit randomness bounds on the probability of generating each word. Requires the use of exact specifications.
* :mod:`Labelled Control Improvisation (LCI)<citoolkit.improvisers.lci.LCI>`: Outlined in detail in `[Vin et al. 2021] <https://tr.soe.ucsc.edu/research/technical-reports/UCSC-SOE-21-09>`_. Generalizes CI by splitting the randomness requirements in two to allow explicit randomness bounds on the probably of selecting a word with a particular label and of selecting any particular word once a label has been fixed. Labels are defined using `Label Functions`_ which assign a single label to each word. Requires the use of exact specifications and label functions.
* :mod:`Maximum Entropy Labelled Control Improvisation (MELCI)<citoolkit.improvisers.lci.MELCI>`: Outlined in detail in `[Vin et al. 2021] <https://tr.soe.ucsc.edu/research/technical-reports/UCSC-SOE-21-09>`_. Similar to LCI, but instead of enforcing explicit randomness bounds on selecting any particular word once a label has been fixed the maximum entropy distribution is selected. This gets a good distribution over randomness without needing explicit word probability bounds for each label. Requires the use of exact specifications and label functions.
* :mod:`Quantitative Control Improvisation (QCI)<citoolkit.improvisers.qci.QCI>`: Outlined in detail in `[Gittis et al. 2022] <https://arxiv.org/abs/2206.02775>`_. Generalizes CI by replacing the soft constraint with a cost constraint, requiring that the expectation of the cost of any word selected from our improvising distribution be under a threshold. Costs are defined using `Cost Functions`_ which assign a single cost to each word. Requires the use of exact specifications and cost functions.
* :mod:`Labelled Quantitative Control Improvisation (LQCI)<citoolkit.improvisers.lqci.LQCI>`: Outlined in detail in `[Gittis et al. 2022] <https://arxiv.org/abs/2206.02775>`_. Generalizes CI by applying the the generalizations of LCI and QCI together. Requires the use of exact specifications, label functions, and cost functions.
* :mod:`Maximum Entropy Labelled Quantitative Control Improvisation (MELQCI)<citoolkit.improvisers.lqci.MELQCI>`: Outlined in detail in `[Gittis et al. 2022] <https://arxiv.org/abs/2206.02775>`_. Generalizes CI by applying the the generalizations of MELCI and QCI together. Requires the use of exact specifications, label functions, and cost functions.
* :mod:`Approximate Labelled Quantitative Control Improvisation (ALQCI)<citoolkit.improvisers.alqci.ALQCI>`: Outlined in detail in `[Gittis et al. 2022] <https://arxiv.org/abs/2206.02775>`_. An extension of CI to support specs that support operations approximately with PAC guarantees. This improviser can also support exponentially many costs in polynomial time. Requires the use of approximate specifications, label functions, and cost functions.


Specifications
**************

Specifications provide ways of defining languages of *words* over an alphabet. Alphabets are often used to mean a set of strings where a word is a sequence of those strings. However, they can also define more abstract concepts, like a dictionary mapping keys to boolean/integer values (See :mod:`Z3Formula<citoolkit.specifications.z3_formula.Z3Formula>`. Specs have one abstract method they must implement:

* ``accepts``: The ``accepts`` function is how a specification defines a language. A word is in the language of a specification if and only if ``accepts`` returns true for that word.

All specifications extend from one of two subclasses of Spec: :mod:`ExactSpec<citoolkit.specifications.spec.ExactSpec>` and :mod:`ApproxSpec<citoolkit.specifications.spec.ApproxSpec>`. Each of these classes has 2 abstract functions that must be implemented, exactly in the case of ExactSpec and approximately in the case of ApproxSpec:

* ``language_size``: The ``language_size`` function allows one to determine the size of a specification's language. For exact specs one can provide lower and upper bounds on words to count, by passing the ``min_length`` and ``max_length`` parameters respectively. For approximate specs, one must provide minimum confidence and tolerance bounds.
* ``sample``: The ``sample`` function allows one to sample uniformly at random from the language of a specification.  For exact specs one can provide lower and upper bounds on words to count, by passing the ``min_length`` and ``max_length`` parameters respectively. For approximate specs, one must provide a tolerance bound.

For more details on these operations see the documentation at :mod:`ExactSpec<citoolkit.specifications.spec.ExactSpec>` and :mod:`ApproxSpec<citoolkit.specifications.spec.ApproxSpec>`.

As long as a specification can support these operations and extend the appropriate base class, it can be used as in a control improvisation improviser. However, the library primarily implements specifications that support these operations *efficiently*, so as to produce an efficient improviser.

The specifications provided natively by this library are:

* :mod:`Dfa<citoolkit.specifications.dfa.Dfa>` (Exact)
* :mod:`BoolFormula<citoolkit.specifications.bool_formula.BoolFormula>` (Approximate)
* :mod:`Z3Formula<citoolkit.specifications.z3_formula.Z3Formula>` (Approximate)

In addition, support the standard union, intersection, negation and difference operations. This is done primarily through the use of the ``AbstractSpec`` class, which creates logical trees of specifications. This allows to run the ``accepts`` function any combination of specs with any of the 4 supported operations. The ``AbstractSpec`` class also has an ``explicit`` method, which allows one to attempt to collapse a tree of abstract specifications into a single concrete Spec subclass. To implement the ``language_size`` and ``sample`` methods, the ``AbstractSpec`` class first tries to compute its explicit form. If this fails, sometimes we can still implement these methods through a more specialized approach. Otherwise, a ``NotImplementedError`` is raised. All of this logic is handled in the ``AbstractSpec`` class, and must be augmented to support new specifications.

Label Functions
***************

A label function assigns a single label to each word in its domain. Exact labelling functions support the ``decompose`` operation, in which the label function is decomposed into a set of specs accepting words with a particular label. Approximate labelling functions support the ``realize`` operation which takes a label and returns a spec that accepts only words with that label.

The label functions provided natively by this library are:

* :mod:`LabellingDfa<citoolkit.labellingfunctions.labelling_dfa.LabellingDfa>` (Exact)
* :mod:`LabellingZ3Formula<citoolkit.labellingfunctions.labelling_z3_formula.LabellingZ3Formula>` (Approximate)

Cost Functions
***************

A cost function assigns a single cost to each word in its domain. Exact cost functions support the ``decompose`` operation, in which the cost function is decomposed into a set of specs each accepting words with a particular cost. Approximate cost functions support the ``realize`` operation which takes an upper and lower bound on cost and returns a spec that accepts only words with a cost in that range.

The cost functions provided natively by this library are:

* :mod:`StaticCostDfa<citoolkit.costfunctions.static_cost_dfa.StaticCostDfa>` (Exact)
* :mod:`AccumulatedCostDfa<citoolkit.costfunctions.accumulated_cost_dfa.AccumulatedCostDfa>` (Exact)
* :mod:`CostZ3Formula<citoolkit.costfunctions.cost_z3_formula.CostZ3Formula>` (Approximate)

