# from ._kernels_cy import vsf_props

# we should support some kind of "sequence unpacking" of the results
# https://docs.python.org/3/tutorial/datastructures.html#tuples-and-sequences

# maybe we differentiate between signed and unsigned or signed and magnitude


def calc_sf(
    values,
    pos,
    bins,
    flavor,
    *,
    weights=None,
    data_other=None,
    pos_other=None,
    weights_other=None,
    calc_variance=False,
    max_order=3,
    nproc=1,
):
    """
    Calculate one or more orders of the structure function.

    A structure function (of a given order) characterizes the
    statistical distribution of a (scalar or vector) field's increments.
      - An increment specifies the amount that a field changes over a
        given separation.
      - A structure function describes a statistical property of the
        field's increment order as a function of separation.
      - In this function,  the structure function for discrete
        separation bins (specified by the ``bins`` argument).

    The interpretation of a structure function of order ``p`` is
    summarized below in the Notes section (this is related to the ``kind``
    parameter).


    Parameters
    ----------
    values : array_like
        The field values for which the structure function is computed.
        This is either:
           - an m by n array that describes n points of dimension m. In
             this case, each vector is of
           - a n element array specifying
    pos : array_like
        Specifies the position of each data point. It should have the same
        shape as ``data``.
    bins : array_like
        1D array of monotonically increasing values that represent the
        edges of the separation bins. A separation ``x`` lies in bin ``i``
        if it satisfies ``bins[i] <= x < bins[i+1]``.
    kind : {"signed-scalar", "longitudinal", "magnitude"}
        Specifies the "kind" of structure function to compute. See the
        Notes section for more detail.
    weights : array_like, optional
        When specified
    max_order : int, optional
        The maximum order of the computed structure function. All
        structure functions of lower order are also computed.
    nproc : int, optional
        Number of processes to use for parallelizing this calculation. Default
        is 1. If the problem is small enough, the program may ignore this
        argument and use fewer processes.


    Notes
    -----
    For a given separation, a field's structure function of order ``p``
    describes the ``p``-th order raw moment of a probability density
    function (PDF) that is linked to the PDF of the field's increment
    for that separation.

    - A "raw moment" is sometimes called a "crude moment" or a "moment
      about the origin". For a given PDF, the "raw moment" is distinct
      from the central moment, unless the PDF's mean is 0.

    - A precise interpretation depends on whether a structure function
      considers the signed value of the field's increment or the
      magnitude of it.

        - The definition commonly adopted in the fluid dynamics literature
          takes moments of the signed value of the field increment. In
          this case, a structure function describes the moment of a
          field increment.

        - The definition commonly adopted in the astrophysics literature
          takes moments of the magnitude of the field increment. Thus the
          structure function describes moments of the magnitude of a field
          increment.

    When the order, ``p``, of a structure function is even then there is
    no difference between these definitions. However, when the value
    literature, the structure functions is defined as a magnitude of the
    are commonly defined in slightly
    different ways (usually there a number of variants that characterize
    different things. These are related to the ``kind`` argument.
    Supported flavors include
      - ``"signed_scalar"``: only valid when the


    """

    pass
