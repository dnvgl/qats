.. _api:

Documentation of **qats** API
#############################

General
*******

The **qats** python package consists of several modules with classes and functions which are documented in the sections
below.

TsDB class
**********

.. autoclass:: qats.tsdb.TsDB
    :members:
    :undoc-members:

    .. rubric:: Attributes

    .. autoautosummary:: qats.tsdb.TsDB
        :attributes:

    .. rubric:: Methods

    .. autoautosummary:: qats.tsdb.TsDB
       :methods:

    .. rubric:: API


TimeSeries class
****************

.. autoclass:: qats.ts.TimeSeries
    :members:
    :undoc-members:

    .. rubric:: Attributes

    .. autoautosummary:: qats.ts.TimeSeries
        :attributes:

    .. rubric:: Methods

    .. autoautosummary:: qats.ts.TimeSeries
       :methods:

    .. rubric:: API

Statistics
**********

.. automodule:: qats.stats
    :members:

    .. rubric:: Function overview

    .. autoautosummary:: qats.stats
       :functions:

    .. rubric:: API

Signal processing
*****************

.. automodule:: qats.signal
    :members:

    .. rubric:: Function overview

    .. autoautosummary:: qats.signal
       :functions:

    .. rubric:: API


Rainflow counting
*****************

.. automodule:: qats.rainflow
    :members:

    .. rubric:: Function overview

    .. autoautosummary:: qats.rainflow
       :functions:

    .. rubric:: API

Weibull distribution
********************

Weibull class
=============

.. autoclass:: qats.weibull.Weibull
    :members:
    :undoc-members:

    .. rubric:: Attributes

    .. autoautosummary:: qats.weibull.Weibull
        :attributes:

    .. rubric:: Methods

    .. autoautosummary:: qats.weibull.Weibull
        :methods:

    .. rubric:: API

Functions
=========

.. automodule:: qats.weibull
    :exclude-members: Weibull

    .. autoautosummary:: qats.weibull
        :functions:

    .. rubric:: API

Gumbel distribution
*******************

Gumbel class
============

.. autoclass:: qats.gumbel.Gumbel
    :members:
    :undoc-members:

    .. rubric:: Attributes

    .. autoautosummary:: qats.gumbel.Gumbel
        :attributes:

    .. rubric:: Methods

    .. autoautosummary:: qats.gumbel.Gumbel
        :methods:

    .. rubric:: API

Functions
=========

.. automodule:: qats.gumbel
    :exclude-members: Gumbel

    .. autoautosummary:: qats.gumbel
        :functions:

    .. rubric:: API

Readers
*******

Readers for common standard file formats used for storing time series.

SIMA
====

.. automodule:: qats.readers.sima
    :members:

    .. rubric:: Function overview

    .. autoautosummary:: qats.readers.sima
        :functions:

    .. rubric:: API

SIMA H5
=======

.. automodule:: qats.readers.sima_h5
    :members:

    .. rubric:: Function overview

    .. autoautosummary:: qats.readers.sima_h5
        :functions:

    .. rubric:: API

Direct access
=============

.. automodule:: qats.readers.direct_access
    :members:

    .. rubric:: Function overview

    .. autoautosummary:: qats.readers.direct_access
        :functions:

    .. rubric:: API

Matlab
======

.. automodule:: qats.readers.matlab
    :members:

    .. rubric:: Function overview

    .. autoautosummary:: qats.readers.matlab
        :functions:

    .. rubric:: API

CSV
===

.. automodule:: qats.readers.csv
    :members:

    .. rubric:: Function overview

    .. autoautosummary:: qats.readers.csv
        :functions:

    .. rubric:: API

Other
=====

.. automodule:: qats.readers.other
    :members:

    .. rubric:: Function overview

    .. autoautosummary:: qats.readers.other
        :functions:

    .. rubric:: API