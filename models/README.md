The `models/` directory specifies models of the household, which determine the $\mathbf{M}$ matrix that appears in the intertemporal Keynesian cross.

It has three modules:

- `models_analytical.py` is for the 5 household models that have analytical representations for $\mathbf{M}$: RA, TA, BU, TABU, and ZL. It both directly calculates these models' $\mathbf{M}$ and $\mathbf{A}$ (asset Jacobian) matrices given parameters (as needed for `main_sec34.ipynb` and `main_sec5.ipynb`) and provides full-fledged blocks for each model (as needed for the quantitative model in `main_sec67.ipynb`).

- `models_heterogeneous.py` is for the 3 household models that involve more heterogeneity and do not have analytical representations: HA-one, HA-two, and HA-hiliq (a different calibration of HA-one). It provides blocks for these models and also evaluates their steady states given parameters.

- `two_account_calvo.py` specifies the `HetBlock` (for the SSJ toolkit) for the two-account household model with Calvo adjustment of the illiquid asset, which is the most complex model we consider and not built into the toolkit.
