"""Feature engineering behind enforced leakage gates, and deterministic splits.

Modules:
- ``build``: the primary pre-delivery feature matrix; every engineered feature declares sources.
- ``full_dataset``: full-population 2018-2024 assembly from cleaned Parquet (canonical path).
- ``dataset``: earlier per-year spread-sample assembly (retained for the sample-era bundle).
- ``splits``: deterministic stratified held_out_final / dev_train / dev_val / dev_test splits.
- ``leakage``: the FR6.1 excluded-variable detector and FR6.3 univariate gate.
"""
