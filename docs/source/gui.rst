.. _gui:

Graphical User Interface
########################

.. todo:: Make more examples demonstrating typical operations.

Create link to GUI in start menu and desktop
********************************************

.. figure:: img/link_app.gif
    :alt: Create start menu link to GUI
    :figclass: align-center
    :target: _images/link_app.gif

    In the terminal, run :code:`qats config --link-app` to create start menu and desktop links to qats. Pin the start menu
    link to the taskbar to further ease accessibility.


Import time series from file
****************************

.. figure:: img/import.gif
    :alt: Import time series from file
    :figclass: align-center
    :target: _images/import.gif

    Here 41 time series from two files (different format) are imported. The first file is imported using the file open
    dialogue (hotkey *CTRL+I*) and the second by drag-and-drop. Finally the time series database is cleared (hotkey *Ctrl+Del*).

The file open dialogue lists every file format QATS can read, including formats added by installed reader plugins
(see :ref:`examples`). Files that no reader can read, whether opened or dropped, are reported in the log window; the
other files are imported.





Maxima/minima distribution
**************************

The *Statistics* and *Maxima/minima CDF* tabs use the same Weibull fit to the maxima (or minima) of each time series,
set in *Settings* under "Maxima/minima distribution":

- **Weibull fit method**: pwm, msm, lse or mle.
- **Tail**: fit only the largest maxima (lowest minima), as a fraction of all of them.
- **Tail model**: a censored fit to all maxima, where the smaller maxima count only by their number, or a fit to the
  exceedances of the threshold value u, the maxima above u as a sample of their own. See
  :meth:`~qats.stats.weibull.Weibull.fit`.
- **Location parameter**: fitted, at the mean level, at a value, or at u in a fit to the exceedances.
- **Plotting positions** of the maxima on Rayleigh paper.
- **Fit to the exceedances drawn**: how the *Maxima/minima CDF* tab draws a fit to the exceedances.

  - *exceedances* (default): as fitted, with only the maxima above u of each series at their own plotting positions,
    and the fitted distribution F_u, conditional on exceeding u. The probability axis is then
    Pr{X <= x | X > u}, and each series has its own u.
  - *full-sample*: all maxima, and the fitted distribution solved for all maxima, F = 1 - p (1 - F_u), where p is the
    fraction of the maxima above u. The maxima outside the fit are faded. This view can be compared with a censored
    fit.

  See :func:`~qats.stats.paper.probability_paper` for both views.

For example, to see how well a fit to the exceedances describes the largest maxima of two mooring lines, check
"Fit only the largest maxima" with 0.20, choose the tail model *exceedances* and display the two series: the
*Maxima/minima CDF* tab shows the maxima above each line's u against its fitted distribution. Choose *full-sample* to
see the same fits with all maxima instead.
