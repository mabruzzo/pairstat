************
Installation
************

Dependencies
============
The primary dependency is having a modern C++ compiler installed.
All other python dependencies will be handled by your python package manager.

General Procedure
=================

The easiest way to install the package as a user is to invoke:

.. code-block:: shell-session

   $ python -m pip install -v pyvsf@git+https://github.com/mabruzzo/pyvsf

Alternatively, you can clone the repository and invoke the following from the repository's root directory.

.. code-block:: shell-session

   $ python -m pip install -v .

The above command installs the minimum required dependencies.
To install extra dependencies:

  * for testing, replace ``.`` in the above statement with ``.[dev]``.
  * for some of the undocumented functionality, replace ``.`` with ``.[extra-features]``.
  * for building docs, replace ``.`` with ``.[docs]``.
  * If you use Z shell (the default shell on modern versions of macOS) you may need to put these snippets inside of single quotes (e.g. ``'.[dev]'``).

.. note::

   In the future, we plan to support installation from PYPI

OpenMP is used automatically used (if the compiler supports it).
To check if the package was compiled with OpenMP, you can invoke the following from the command-line (and check if the printed statement mentions OpenMP)

.. code-block:: shell-session

   $ python -m pyvsf

Using OpenMP for parallelization on macOS
=========================================

If you want to use OpenMP on macOS, you will need to install a C++ compiler that supports it.
The default C++ compiler on macOS is an apple-specific version of clang++.

- The easiest way to get a different compiler is use homebrew to install a version of g++.

- Once you have an installed version of g++, (like g++-14), you should invoke either

  .. code-block:: shell-session

     $ CXX=g++-14 python -m pip install -v pyvsf@git+https://github.com/mabruzzo/pyvsf

  or

  .. code-block:: shell-session

     $ CXX=g++-14 python -m pip install -v .

- **NOTE:** on macOS simply typing ``g++`` aliases the default clang++ compiler (``g++-14`` invokes a different compiler)

