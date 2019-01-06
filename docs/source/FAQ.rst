FAQ
================

Frequently asked questions.


False alarm probability
------------------------

From an experiment with >10000 white noise-only TLS search runs, we can estimate false alarm probabilities (FAP) as follows:

======   =====
1-FAP    SDE 
======   =====
0.9      5.7
0.95     6.1
0.99     7.0
0.999    8.3
0.9999   9.1
======   =====

In noise-only data, 1% of the observed cases (1-FAP=0.99) had an SDE>7.0. If an SDE of 9.1 is observed in a data set, the probability of this happening from noise fluctuations is 0.01%. This assumes Gaussian white noise. Real data often has partially correlated (red) noise. Then, the FAP estimates are too optimistic, i.e., high SDE values will occur more often than measured in the experiment. Vice versa, the SDE values per given FAP value will be higher in red noise. 

TLS returns the FAP value per SDE as ``results.FAP`` (see Python interface).


Truncation of the power spectrum
------------------------------------------

The Figure below (left panel) is taken from the paper (Figure 3) and shows (a): the :math:`\chi^2` distribution (b): The signal residue (c): the raw signal detection efficiency and (d): the signal detection efficiency (SDE) used by TLS, smoothed with a walking median. This plot was made using the default parameters in TLS.

In the right panel, the only change is ``transit_depth_min=200*10**-6``. That is, we decide not to fit any transits shallower than 200ppm (instead of 10ppm). As a consequence, no transits were fit for many short periods (these are smoother in phase space). The resulting spectrum contains maximum :math:`\chi^2` values (where the signal is taken as unity) for many periods, resulting in SDE values of zero. With a lower baseline, the actual SDE peaks may be **higher** (remember: the SDE power spectrum is normalized to its standard deviation). Despite the higher peaks, the information content is lower, as true signals may be missed, and no additional information is introduced.



|pic1| any text |pic2|

.. |pic1| image:: faq_1.png
   :width: 45%

.. |pic2| image:: faq_2.png
   :width: 45%
