****************************
Turbulence Statistics Primer
****************************

.. note::

   This is worth briefly reviewing if your primary knowledge of turbulence comes from astrophysics.
   There are some nuiances that are important to appreciate.

   While our discussion focuses on motivating and defining the velocity structure function and correlation function, you can also applying these statistics to other scalar/vector fields.


On this page, we briefly define and motivate the turbulence statistics computed by this package (namely structure functions and correlation functions).
This page is organized as follows:

* First we briefly introduce the idea of describing turbulence probabilistically.
  We focus on the idea that a flow's velocity field can be treated as a random field that can be characterized in terms of its "increments".
  We also relate the concept of structure functions to these increments.

* Next we highlight the key predictions of Kolmogorov theory (as they pertain to structure functions).

  .. note::

     Given that Kolmogorov theory makes strong assumptions about flows (i.e. it's homogeneous, isotropic, incompressible), one might reasonably ask: *"why focus on this?"*
     We'll touch on this question, but the short answer is to provide a basic intuition for interpretting the structure function.

* Then we introduce concrete definitions (there is more than 1) of the structure function that are used in practice.

Probabilistically Describing Turbulence
=======================================

[Frisch1995]_ provides an excellent introduction to this topic and provides motivation for why we may want a probabilistic description of turbulence.
But, we highlight a few quick points.

The basic idea is that we treat a flow's velocity field, :math:`{\bf v}({\bf x}, t)` as a `random field <https://en.wikipedia.org/wiki/Random_field>`__ (sometimes it can be called a `stochastic process or random function <https://en.wikipedia.org/wiki/Stochastic_process>`__), whose evolution is controlled by the Navier Stokes equations.
At a high level, it is also to understand that :math:`v_i({\bf x}, t)`, an arbitrary component of :math:`{\bf v}({\bf x}, t)`, is itself a scalar random field.

A measure of :math:`{\bf v} ({\bf x}, t)` at a given position :math:`{\bf x}` and time, :math:`t`, you are just sampling the underlying probability distribution for the velocity at that :math:`{\bf x}` and :math:`t`. [#equiv-realization]_
For simplicity, let's just focus on velocity as random function of position, :math:`{\bf v}({\bf x})`.
We also assume that that there is no bulk flow :math:`\langle | {\bf v}({\bf x}) | \rangle = 0`. [#anglebracket]_

Velocity Increments
-------------------

A probabilistic theory of turbulence (like Kolmogorov theory) might predict/describe spatial correlations in the velocity field.
These spatial correlations can be described in terms of the increment in the value over some separation :math:`\boldsymbol{\ell}` (this is a vector: it has a magnitude and direction).
At position `{\bf x}`, we define the velocity increment over a separation :math:`\boldsymbol{\ell}`  as

.. math:: \delta {\bf v}({\bf x, \boldsymbol{\ell}}) = {\bf v}({\bf x} + \boldsymbol{\ell}) - {\bf v}({\bf x}).
   :label: vincrement

:math:`\delta {\bf v}({\bf x}, \boldsymbol{\ell})` **is itself a random field** because it describes the difference between two random values.
We briefly discuss how :math:`\delta {\bf v}({\bf x, \boldsymbol{\ell}})` relates to the correlation function in the collapsible block at the end of this subsection.

Let's consider how the assumptions of spatial homogeneity and isotropy allow us to simplify :math:`\delta {\bf v}({\bf x, \boldsymbol{\ell}})`.

* *Homogeneity:* When the velocity field is spatially homogenous, then the absolute position, {\bf x}, has no bearing on the value of the velocity field.
  This means that the velocity increments are also independent of position.
  Thus, we can now describe the velocity increment as :math:`\delta {\bf v}(\boldsymbol{\ell})`. [#stationary-homogeneous]_

* *Isotropy:* When the velocity field is isotropic, there is no mean flow :math:`\langle | {\bf v}({\bf x}) | \rangle = 0` (something we already assumed).
  Importantly, it **also** means that the value of the velocity increment is indepent of the :math:`\boldsymbol{\ell}` vector's direction.
  Thus, we can describe the velocity increment as :math:`\delta {\bf v}(\ell)`, where :math:`\ell = | \boldsymbol{\ell} |`.

Concept of Velocity Structure Function
--------------------------------------

Forget about the assumptions of *Homogeneity* and *Isotropy* for the moment.
In other words, let's think of the velocity increment as a random field with notation :math:`\delta {\bf v}({\bf x, \boldsymbol{\ell}})`.

The basic idea of a structure function is to characterize the probability distribution function (PDF) of velocity increments.
We will provide a theroretical definition for the structure function, borrowing from Kolmogorov's original definition [#astrovsf-qualification]_).
We define :math:`p`-th order velocity structure function at position :math:`{\bf x}` for separation :math:`\boldsymbol{\ell}` and at position :math:`{\bf x}`, directly measures the moment of :math:`\delta {\bf v}({\bf x, \boldsymbol{\ell}})`.

Given that velocity is a vector, this is a somewhat imprecise definition.
In general, you can think of the :math:`p`-th order structure function as a rank :math:`p` tensor.
Since velocity has 3 vector components, the :math:`p`-th order tensor contains :math:`3^p` entries.
3 of the tensor elements hold the :math:`p`-th order moment for the PDF for a distinct component of :math:`\delta {\bf v}({\bf x, \boldsymbol{\ell}})`.
The remaining elements hold :math:`p`-th order mixed moments describing joint PDF of unique component combinations of :math:`\delta {\bf v}({\bf x, \boldsymbol{\ell}})`.

.. admonition:: Aside about relationship with correlation function

   .. collapse:: Click here to show/hide more information


      Borrowing notation from [Pope2000]_, elements of the :math:`p=1`, :math:`p=2`, :math:`p=3` tensors are:

      .. math::
         :nowrap:

         \begin{eqnarray}
         D_{i} ({\bf x, \boldsymbol{\ell}}) &\equiv& \langle \delta v_i ({\bf x}, \boldsymbol{\ell}) \rangle = \langle (v_i({\bf x} + \boldsymbol{\ell}) - v_i({\bf x})) \rangle \\
         D_{ij}({\bf x, \boldsymbol{\ell}}) &\equiv& \langle \delta v_i ({\bf x}, \boldsymbol{\ell}) \delta v_j ({\bf x}, \boldsymbol{\ell}) \rangle = \langle (v_i({\bf x} + \boldsymbol{\ell}) - v_i({\bf x})) ( v_j({\bf x} + \boldsymbol{\ell}) - v_j({\bf x}))\rangle \\
         D_{ijk}({\bf x}, \boldsymbol{\ell}) &\equiv& \langle \delta v_i ({\bf x, \boldsymbol{\ell}}) \delta v_j ({\bf x}, \boldsymbol{\ell}) \delta v_k ({\bf x}, \boldsymbol{\ell}) \rangle

         \end{eqnarray}

      
      In this notation, the subscripts correspond to velocity components.
      When the subscripts are all equal to a single value, the tensor element specifies the moment of a single vector component.
      Otherwise the tensor elements specify mixed moments.

      The elements of the :math:`p=2` tensor are analogous to the elements of a covariance matrix.

      .. note::

         There need not be any special relationship between the direction of :math:`\boldsymbol{\ell}` and the coordinate basis used to define the velocity components.

In practice, we'll only consider the 3 components of the tensor that aren't mixed moments (these could be more important in anisotropic turbulence).
It is convenient to define the components of the velocity increment in a coordinate system that is defined such that :math:`\boldsymbol{\ell}` is aligned with one coordinate axes (\ [Pope2000]_ shows that the mixed moments of the second order velocity structure function tensor are 0 in Kolmogorv turbulence when you use such coordinates).


.. admonition:: TODO: define parallel

   details

.. admonition:: TODO: define perpendicular

   details


.. hint:: 

   Don't forget!
   These angle brackets represent ensemble averages (i.e. averages over PDFs) not spatial averages!
   (This is easy to forget, since this isn't how the structure function is usually presented).


.. admonition:: TODO: reintroduce isotropy and homogeneity

   details


In the next section, we discuss the predictions of Kolmogorov theory (namely predictions pertaining to the structure functions).
Later, we turn our attention to measuring the velocity structure functions, which describe the probability density functions of :math:`\delta {\bf v}(\ell)` for different separations .
Measurements of the velocity structure functions leverage the ergodic theorem (more on this later).



.. admonition:: Aside about relationship with correlation function

   .. collapse:: Click here to show/hide more information

      We momentarily ignore isotropy and homogeneity (we still assume no bulk flow).
      To discuss this point, we briefly employ the notation defined for 2nd order velocity structure functions tensor that was defined in the previous "aside", `:math:`D_{ij}(\boldsymbol{\ell})`.

      We can also define the related quantity, the 2-point velocity correlation function, for separation :math:`\boldsymbol{\ell}` at position :math:`{\bf x}` as a tensor:

      .. math::

         R_{ij}({\bf x, \boldsymbol{\ell}}) = \langle v_i ({\bf x} + \boldsymbol{\ell}) v({\bf x}) \rangle

      .. note::

         In this scenario, there need not be any special relationship between the direction of :math:`\boldsymbol{\ell}` and the coordinate basis used to define the velocity components.

      We show the relationship below.
      In the process, we reintroduce our homogeneity assumption.

      .. math::
         :nowrap:

         \begin{eqnarray}
         D_{ij}({\bf x}, \boldsymbol{\ell}) &=& \langle (v_i({\bf x} + \boldsymbol{\ell}) - v_i({\bf x})) ( v_j({\bf x} + \boldsymbol{\ell}) - v_j({\bf x}))\rangle \\
           &=& \langle v_i({\bf x} + \boldsymbol{\ell}) v_j({\bf x} + \boldsymbol{\ell}) \rangle
            - \langle v_i({\bf x} + \boldsymbol{\ell}) v_j({\bf x}) \rangle
            - \langle v_i({\bf x}) v_j({\bf x} + \boldsymbol{\ell}) \rangle
            + \langle v_i({\bf x}) v_j({\bf x}) \rangle \\
           &=& R_{ij}({\bf x},{\bf 0}) + R_{ij}({\bf x} + \boldsymbol{\ell}, {\bf 0}) - R_{ij}({\bf x},\boldsymbol{\ell}) - R_{ji}({\bf x},\boldsymbol{\ell}) \\ 
           &\downarrow& {\rm Homogeneity\ Assumption} \\ 
         D_{ij}(\boldsymbol{\ell}) &=& 2 R_{ij}(\boldsymbol{\ell}={\bf 0}) - R_{ij}(\boldsymbol{\ell}) - R_{ji}(\boldsymbol{\ell})
         \end{eqnarray}

      When :math:`i` and :math:`j` specify the same component, :math:`D_{ii}(\boldsymbol{\ell}) = 2 R_{ii}(\boldsymbol{\ell}={\bf 0}) - 2 R_{ji}(\boldsymbol{\ell})`.
      Equivalently, we could say that :math:`D_{ii}(\boldsymbol{\ell}) = 2 \langle v_i^2 \rangle - 2 R_{ji}(\boldsymbol{\ell})`.

      .. rubric::  Incompressibility and Isotropy

      When you account for incompressibility, all of the off-diagonal entries of the tensors have values of zero (e.g. [Pope2000]_, [Choudhari1998]_)


Kolmogorov theory
=================

We begin, by highlighting the key predictions of Kolmogorov theory, that derives from his 1941 papers.

This theory specifically applies to homogeneous, isotropic, incompressible hydrodynamic flows at high Reynolds numbers.
While there are a number of cases where fluids don't necessarily satisfy these assumptions, these predictions essentially provide a standard (at least in the astrophysical literature) by which turbulence measurements can be judged.
Even if a flow doesn't necessarily satisfy these assumptions, the predictions may still be relevant.

* For example, if the flow is anisotropic at large scales, at small enough scales, anisotropies become small and it may be alright to approximate it as anisotropic.

* Likewise, flows commonly are not homogenous. But you can approximate the turbulence region as "locally homogeneous".

* Additionally, turbulence statistics have been known to roughly match Kolmogorov predictions in compressible (even multiphase) flows.


This section assumes that the reader already has familiarity with hydrodynamic turbulence and idea of the turbulent energy cascade.
The reader should be familiar with the picture:

* Some kind of driving force produces large scale eddies.
  In other words, kinetic energy is injected into eddies with an length scale of :math:`L` or that are larger.

* These large-scale eddies break up into smaller eddies and transfer their kinetic energy down to these smaller eddies.
  This breakup and energy-transfer of eddies into smaller eddies continue down to smaller and smaller scales, until they reach the dissipiation scale, :math:`\ell_{\rm dis}`.

* Below :math:`\ell_{\rm dis}`, viscosity is dynamically relevant. 
    The turbulent kinetic energy is dissipated at scales smaller than or below :math:`\ell_{\rm dis}`.

Kolmogorov theory focuses on describing the turbulence properties at scales between :math:`\ell_{\rm dis}` and :math:`L`, or the inertial subrange.
Importantly, there is a lot of empirical evidence that turbulence is self-similar over the inertial subrange.




Key results of Kolmogorov theory
--------------------------------

There are a number of ways to derive the scaling relations predicted by Kolmogorov.
It is worth mentioning that the original Kolmogorov papers made some questionable assumptions to derive these predictions.
We highlight the 2 key, robust predictions from [Frisch1995]_'s (self-described) unusual treatment of Kolmogorov theory.

The idea that turbulence is self-similar leads to an important scaling relation.
The form of this law is dictated by the "scaling symmetry" allowed by the incompressible Navier Stokes equation when viscosity is zero (e.g. see Section 2.2 of [Frisch1995]_).
This dictates that:

.. math:: \delta {\bf v}({\bf r}, {\bf \lambda \ell}) \stackrel d = \lambda^h\ \delta{\bf v}({\bf r}, \boldsymbol{\ell}),
   :label: scaling-generic

where :math:`\lambda` is an arbitrary positive value and the exponent :math:`h` is a universal constant.
The precise form of this scaling law derives from the scaling transformation allowed by the Navier
The :math:`\stackrel d =` notation indicates that :math:`\delta {\bf v}({\bf r}, {\bf \lambda \ell})` and :math:`\lambda^h\ \delta{\bf v}({\bf r}, \boldsymbol{\ell})` have equivalent probability density functions.

[Frisch1995]_ points out that Kolmogorov's four-fifth law is a robust non-trivial prediction derived in Kolmogorov's 3rd 1941 paper that makes minimal additional assumptions:

.. math:: \langle (\delta v_\parallel (\boldsymbol{\ell}))^3 \rangle = - \frac{4}{5} \epsilon \ell,
   :label: four-fifths

where :math:`\epsilon` is the finite mean rate of energy dissipation for unit mass and :math:`\delta v_\parallel ({\bf r}, \boldsymbol{\ell})` refers to the *longitudinal* component of the velocity increment.
The longitudinal component is parallel to the offest :math:`\boldsymbol{\ell}` or :math:`\delta v_\parallel ({\bf r}, \boldsymbol{\ell}) = \delta {\bf v} ({\bf r}, \boldsymbol{\ell}) \cdot (\boldsymbol{\ell}/ \ell)`.

This result implies that :math:`h=1/3` in :math:numref:`scaling-generic`.

.. _vsf-practical-definitions::

Practial Velocity Structure Function Definitions
================================================

.. important::

   COME BACK TO THIS

   describe ergodic theorem

   In an arbitrary coordinate system with unit vectors ij.

   pg 52 of Frisch talk about only needing to measure in 1D (if the length of the domain is long enough)
   
   eqn 4.61 of Frisch 2nd order velocity structure function

   pg 57 of Frisch talk about dropping the spatial dependence and just treat it as isotropic

To probe turbulence, people often try to measure the "velocity structure function" to try probe the scaling described in :math:numref:`scaling-generic`.
A number of definitions.

A popular quantity in the fluid dynamics literature is the longitudinal velocity structure function.

.. math:: S_p(\ell) = 
   :label: sf-longitudinal

At least historically, experimentalists would focus on measuring the structure function along a single spatial dimension.

Can also measure the transverse or lateral dimension.
They would pick a single, consistent component of the velocity increment perpendicular to the separation :math:`\boldsymbol{\ell}`.

.. math:: S_p^\perp(\ell) = 
   :label: sf-transverse

Along a given dimension, you can also measure [Pope2000]

.. important::

   This package supports the calculation of this quantity!

The transverse or lateral structure function is also sometimes measured.



The velocity structure function (hereafter "astro VSF") of order :math:`p` is sometimes called:

.. math:: {\rm VSF}_p(\ell) = \langle (| \delta v_\parallel (\ell) | )^p \rangle = \langle ( | {\bf v}({\bf r} + \boldsymbol{\ell})  {\bf v}({\bf r} + \boldsymbol{\ell}) | )^p \rangle
   :label: astrovsf

As noted in [Mohapatra+2022]_, this is common in the astrophysical literature (e.g. Abruzzo, Li, ...).
[Frisch1995]_ also briefly references this quantity in eqn 4.61.
Kolmogorov theory only makes robust predictions for even orders, :math:`p`.


.. admonition:: Aside about another prediction

   .. collapse:: Click here to show/hide more information

      There is one other potentially interesting prediction from Kolmogorov theory that is worth mentioning.
      To discuss this point, we briefly employ the notation defined for 2nd order velocity structure functions tensor that was defined in a previous "aside", `:math:`D_{ij}(\boldsymbol{\ell})`.

      We previously discussed things with this notation under the assumptions of homogeneity and incompressibility.
      Let's now consider isotropy.
      We can infer an additional interesting property from the tensor description of the structure function ([Pope2000]_ goes into detail, but we list some highlights here).

      * let's treat continue treating :math:`\boldsymbol{\ell}` as a vector and let's define the velocity components with respect to the direction of :math:`\boldsymbol{\ell}`.

      * The structure function value for the velocity component parallel to :math:`\boldsymbol{\ell}` is called the longitudinal structure function, denoted as :math:`D_{LL}(\boldsymbol{\ell})`.

      * We would refer to the structure function value computed for the velocity compoinent along a single, arbitrary direction perpendicular to :math:`\boldsymbol{\ell}` as the transverse/lateral structure function, :math:`D_{NN}(\boldsymbol{\ell})`.

      * For concreteness if we picked a coordinate system where :math:`\boldsymbol{\ell}` is parallel to a unit vector :math:`\hat{{\bf e}}_1`, then :math:`D_{11}(\boldsymbol{\ell}) = D_{LL}(\boldsymbol{\ell})` and :math:`D_{22}(\boldsymbol{\ell}) = D_{33}(\boldsymbol{\ell}) = D_{NN}(\boldsymbol{\ell})` 

      * The primary result of interest here is that :math:`D_{NN}(\boldsymbol{\ell}) = (4/3) D_{LL}(\boldsymbol{\ell})`.
        Be advised, some additional assumptions go into this derivation, and it's not completely clear to me how robust they are.

Why not characterize turbulence in Fourier Space?
=================================================

One might ask: why does this package provide functions that operate in physical space?
Why not work in Fourier space?

This is a totally fair question.
In fact, characterizing turbulence in Fourier space is the most computationally efficient way to characterize isotropic turbulence simulations in periodic boxes.
After all, the power spectrum is the fourier transform of the 2point auto-correlation function (which encodes the same information as the second order velocity structure function).

However, things become much harder if your experiment is **not** periodic.

As soon as you want to mask a cell, things also become harder; fast fourier transforms implicitly assume that you have a full regular grid of data.
It is very hard to characterize how missing data will affect your results (it's for this very reason that cosmologists often infer cosmological properties using real-space summary statistics rather than fourier space statistics.


.. rubric:: Footnotes

.. [#equiv-realization] In the language used in statistical phyiscs, you might say that we are just considering a single realization of the velocity at that :math:`{\bf x}` and :math:`t`

.. [#anglebracket] Throughout this section we use angle brackets to denote an ensemble-average (or an average over a PDF). 
   For example, the mean and RMS value of a random variable :math:`x` are given by :math:`\langle x \rangle` and :math:`\langle x^2 \rangle`.

.. [#stationary-homogeneous] [Frisch1995]_ provides slightly detail about homogeneity.
   He presents the this property as the spatial analog to a random function being `time-stationary <https://en.wikipedia.org/wiki/Stationary_process>`__ (a staionary process always has `stationary increments <https://en.wikipedia.org/wiki/Stationary_increments>`__ )

.. [#astrovsf-qualification] In practice, certain common formulations of the velocity structure function don't actually measure moments (see eqn :math:numref:`astrovsf` for odd :math:`p`).
   We return to this point in :ref:`this section <vsf-practical-definitions>`.


.. rubric:: References

.. [Choudhari1998] Choudhari, A. R. 1998, The Physics of Fluids and Plasmas: An Introduction for Astrophysicists (Cambridge: Cambridge Univ. Press)

.. [Frisch1995] Frisch, U. 1995, Turbulence: the legacy of A. N. Kolmogorov (Cambridge: Cambridge Univ. Press)

.. [Mohapatra+2022] Mohapatra R., Jetti, M., Sharma, P., & Federrath, C. 2022, MNRAS, 510, 2327

.. [Pope2000] Pope S. B., 2000, Turbulent Flows (Camebridge: Cambridge Univ. Press)

