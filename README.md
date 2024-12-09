# ikc

This repository replicates the figures and tables in "[The Intertemporal Keynesian Cross](https://doi.org/10.1086/732531)" (Auclert, Rognlie and Straub JPE 2024)

The code runs in Python 3.12. It uses the [Sequence Space Jacobian toolkit](https://github.com/shade-econ/sequence-jacobian/) version 1.0.

To obtain a new environment that will run the code using `conda`, at the command line you can write  
```
conda create -n ikc_env numpy scipy matplotlib numba pandas jupyter python=3.12
conda activate ikc_env
pip install sequence-jacobian
``` 
before launching Jupyter or running any code.

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

