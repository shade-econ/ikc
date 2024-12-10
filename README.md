# ikc

This repository replicates the figures and tables in "[The Intertemporal Keynesian Cross](https://shade-econ.github.io/ikc/ikc.pdf)" (Auclert, Rognlie, and Straub JPE 2024). \[[Published version](https://doi.org/10.1086/732531)\]

The code requires the [Sequence Space Jacobian (SSJ) toolkit](https://github.com/shade-econ/sequence-jacobian/) version 1.0, in addition to standard numerical Python packages (`numpy`, `scipy`, `matplotlib`, `numba`, `pandas`) and Jupyter notebooks. We have tested it using Python 3.10+. The SSJ toolkit can be installed using `pip install sequence-jacobian`; please see the toolkit site for additional instructions.

If you run into any difficulties with the code, please feel free to post on the repository's issue tracker.

## Organization

Most of the results of the paper are obtained in three main Jupyter notebooks, each of which is associated with some sections in the main text, and produces figures and tables for those sections, in addition to closely related appendix figures and tables:

- `main_sec34.ipynb` (sections 3 and 4) calibrates our household models and compares their iMPCs to those in the data
- `main_sec5.ipynb` (section 5) studies the effects of fiscal policy subject to the IKC
- `main_sec67.ipynb` (sections 6 and 7) sets up our quantitative environment and studies the iMPCs out of capital gains and effects of fiscal policy in that environment

Below we have a more complete outline of the contents of each of the main notebooks.

### Part I: IKC environment, calibration and iMPCs - sections 3 and 4 (`main_sec34.ipynb` )

- Figure 1 - iMPCs in the Norwegian and Italian data
  - Norwegian iMPCs: read from `data/FIG2_c1R_inc_weight.xlsx` (obtained from Fagereng, Holm, Natvik)
  - Italian iMPCs: read from `impc_lb_italy.txt`, computed in `appendix_shiw` notebook (source data in `data/shiw2016.csv`)
- Figure 2 - iMPCs in the Norwegian data and several models
- Figure 3 - iMPCs in eight standard models
- Figure 4 - log iMPCs out of unexpected and expected income shocks
- Figure D1 - asset Jacobians (companion to Figure 3)
- Figure D2 - policies and distributions in HA-one
- Figure D3 - policies and distributions in HA-two
- Table 2 - calibrating models of the intertemporal consumption function
- Table D1 - distributional statistics for all heterogeneous-agent models

### Part II: fiscal policy in the IKC environment - section 5 (`main_sec5.ipynb`)

- Figure 5 - multipliers according to the IKC
- Figure 6 - effect of adding cognitive discounting to, and of truncating the tails of M
- Table 4, part I - multipliers in the IKC environment
- Figure A2 - PE to GE: fiscal policy, monetary policy, and deleveraging
- Figure E1 - lump-sum taxation
- Figure E2 - multipliers across all eight models in the IKC environment
- Figure E5 - effect of discounting applied separately to M and MT (companion to Figure 6)

The notebook also verifies numerically propositions 4, 5, 6, 7 and 11

### Part III: quantitative environment (sections 6 and 7) (`main_sec67.ipynb`)

- Figure 7 - iMPCs out of capital gains vs income
- Figure 8 - government spending shock in quantitative two-account model 
- Figure 9 - multipliers in the quantitative models
- Figure 10 - decomposing the consumption responses in the two-account and TABU model
- Figure 11 - role of monetary policy: contrasting active Taylor rule, ZLB, and constant real rate
- Table 3 - calibration of the quantitative environment
- Table 4, part II - multipliers in the quantitative environment
- Figure F1 - robustness to holding equity only in the illiquid account
- Figures G1-G4 - sensitivity of impulse responses to parameters

### Additional notebooks

- `appendix_solutions.ipynb` considers four approaches to solving the IKC and produces Figure A.1 
- `appendix_shiw.ipynb` produces the data underlying Figure 1 as well as Figure C.1
- `appendix_ikc_determinacy.ipynb` tests determinacy of the models in the IKC environment via the winding number

#### Note: 
The code to produce Figures E3 and E4 (nonlinearities and state dependence) is currently missing since running it requires us to update the SSJ repo first. Stay tuned for an update.

