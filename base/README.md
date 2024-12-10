The `base/` directory contains all modules that are not either for plotting (which is in `plotting/`) or specifying household-level models of consumption and savings (which is in `models/`).

It includes the following modules:

- `calibration.py`: This specifies all exogenous calibration parameters, and solves fully for every part of the GE calibration in both the simple IKC and the full quantitative environments, with the exception of internally calibrated household parameters (which are specific to individual household models and obtained in `main_sec34.ipynb`).

- `capital_sticky_prices.py`: This provides blocks needed for the quantitative environment in `main_sec67.ipynb`, specifying a production side where capital is an input adjusted subject to adjustment costs and there is Rotemberg price stickiness.

- `fiscal.py`: This provides convenience code for computing multipliers, and for obtaining paths of $B_t$ and $T_t$ corresponding to different fiscal plans.

- `jacobian_manipulation.py`: This provides various routines for directly manipulating sequence-space Jacobians - for instance, converting between the asset and consumption Jacobians $\mathbf{A}$ and $\mathbf{M}$, or implementing cognitive discounting.

- `winding_number.py`: This implements the "winding number" determinacy test for quasi-Toeplitz Jacobians that is described in the companion note "Determinacy and Existence in the Sequence Space" and implemented in the `appendix_determinacy.ipynb` and `main_sec67.ipynb` notebooks.