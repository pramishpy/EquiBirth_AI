"""Group fairness measurement, mitigation, and disparity decomposition.

Modules:
- ``metrics``: group metrics with CIs and significance; equal opportunity is the headline.
- ``mitigation``: reweighting, Fairlearn reductions, and threshold optimization, each priced
  with its disclosed cost to discrimination or calibration.
- ``decomposition``: g-computation/standardization decomposition of the Black-White gap.

Nothing here "solves" fairness: each lever buys an improvement at a stated cost, and the
disparate-impact vs disparate-treatment tension is examined rather than resolved.
"""
