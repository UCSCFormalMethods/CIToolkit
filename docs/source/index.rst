CIToolkit Documentation
=====================================

Overview
============
``citoolkit`` is a library containing tools to create and solve instances of the Control Improvisation problem and its extensions. This library supports the following flavors of Control Improvisation:

* Control Improvisation (CI) `[Fremont et al. 2017] <https://arxiv.org/abs/1704.06319>`_
* Labelled Control Improvisation (LCI) `[Vin et al. 2021] <https://tr.soe.ucsc.edu/research/technical-reports/UCSC-SOE-21-09>`_
* Maximum Entropy Labelled Control Improvisation (MELCI) `[Vin et al. 2021] <https://tr.soe.ucsc.edu/research/technical-reports/UCSC-SOE-21-09>`_
* Quantitative Control Improvisation (QCI) `[Gittis et al. 2022] <https://arxiv.org/abs/2206.02775>`_
* Labelled Quantitative Control Improvisation (LQCI) `[Gittis et al. 2022] <https://arxiv.org/abs/2206.02775>`_
* Maximum Entropy Labelled Quantitative Control Improvisation (MELQCI) `[Gittis et al. 2022] <https://arxiv.org/abs/2206.02775>`_
* Approximate Labelled Quantitative Control Improvisation (ALQCI) `[Gittis et al. 2022] <https://arxiv.org/abs/2206.02775>`_

Various specifications are available for encoding constraints, label functions and cost functions. These are primarily based on deterministic finite automata, boolean formulas, and z3 formulas using the theory of bitvectors. For more details, see the :doc:`usage` section.

If you encounter any problems with the ``citoolkit`` please submit an issue on GitHub or contact Eric at evin@ucsc.edu

Table of Contents
==================
.. toctree::
   :maxdepth: 2

   installation
   testing
   quickstart
   usage
   API Reference <modules>

Indices
==================

* :ref:`genindex`
* :ref:`modindex`
