Usage
============

The core classes of the ``citoolkit`` are broken into two packages: **Specifications** and **Improvisers**. 

Specifications
**************

Specifications provide ways of defining languages of *words* over an alphabet of *symbols*. In this library any string can be a symbol and a word is simply an ordered collection of words, typically a tuple. All specifications extend from the Spec class, which has 3 abstract functions must be implemented:

* ``accepts``: The ``accepts`` function is how a specification defines a language. A word is in the language of a specification if and only if ``accepts`` returns true for that word.
* ``language_size``: The ``language_size`` function allows one to determine the size of a specification's language. One can provide lower and upper bounds on words to count, by passing the ``min_length`` and ``max_length`` parameters respectively.
* ``sample``: The ``sample`` function allows one to sample uniformly at random from the language of a specification. One can provide lower and upper bounds on words to count, by passing the ``min_length`` and ``max_length`` parameters respectively.

As long as a specification can support these operations and extend the Spec base class, it can be used as in control improvisation. However, the library primarily implements specifications that support these operations *efficiently*, so as to produce an efficient improviser.

The specifications provided natively by this library are:

* Deterministic Finite Automaton (DFA): :mod:`citoolkit.specifications.dfa.Dfa`

In addition, support the standard union, intersection, negation and difference operations. This is done primarily through the use of the ``AbstractSpec`` class, which creates logical trees of specifications. This allows to run the ``accepts`` function any combination of specs with any of the 4 supported operations. The ``AbstractSpec`` class also has an ``explicit`` method, which allows one to attempt to collapse a tree of abstract specifications into a single concrete Spec subclass. To implement the ``language_size`` and ``sample`` methods, the ``AbstractSpec`` class first tries to compute its explicit form. If this fails, sometimes we can still implement these methods through a more specialized approach. Otherwise, a ``NotImplementedError`` is raised. All of this logic is handled in the ``AbstractSpec`` class, and must be augmented to support new specifications.

Improvisers
**************

The improviser classes are implementations of the algorithms to solve the Control Improvisation problem and its extensions. The Improviser base class has two abstract methods: ``improvise`` and ``generator``. ``improvise`` returns a single improvised word while ``generator`` returns an iterable object that continually generates words. Some improvisers provide also additional information about the distribution of words they are generating.

The improvisers provided natively by this library are:

* Control Improvisation (CI): Outlined in detail in `[Fremont et al. 2017] <https://arxiv.org/abs/1704.06319>`_. Allows for an explicit hard constraint, a soft constraint that can be violated within a provided tolerance, and explicit randomness bounds on the probability of generating each word.
