# Structure Functions and KE Transfers using Regularized Least-Squares (RLS)

This repository provides code to estimate **structure functions (SFs)** and **kinetic energy (KE) transfers** from both **two-layer QG model outputs** and **ocean drifter datasets**. The methodology leverages **third-order structure functions** and a **regularized least-squares (RLS) framework** to improve the estimation of KE transfers and their divergences.

---

## Repository Structure

### Data Processing
- **`calcSF_QGmodel`**  
  Runs the two-layer quasi-geostrophic (QG) model. Uses 15 years of horizontal velocity outputs and estimates structure functions from the final 5 years of simulation.

- **`process_Drifter_dataset`**  
  Processes QC’ed drifter datasets from the **GLAD** and **LASER** experiments to compute structure functions.

---

### KE Transfers and Injections
- **`TwoLayer_QGModel_SFs_Fluxes`**  
  Uses the model-derived structure functions to compute PDFs and estimate KE transfers and injections from third-order SFs.

- **`Drifter_SFs_Fluxes`**  
  Computes structure functions, statistical degrees of freedom, KE transfers, and injections using drifter datasets.

---

### Core Libraries
- **`drifter_analysis_SFs.py`**  
  Functions for estimating structure functions from drifter datasets.

- **`StrucFunction.py`**  
  Tools for calculating structure functions from 2D gridded fields (supports double-periodic and non-periodic domains).

- **`spectralanalysis.py`**  
  Functions for spectral analysis: wavenumber spectra, confidence intervals, structure functions from spectra, and KE transfers using Fourier methods.

- **`strucfunc2KEflux.py`**  
  Functions to estimate KE transfers and injections from third-order structure functions using RLS fits.

---

## References
If you use this code, please cite:  

- Gutierrez-Villanueva, M. O., Cornuelle, B., Gille, S. T., & Balwada, D. (2025). An Improved Methodology to Estimate Cross-Scale Kinetic Energy Transfers from Third-Order Structure Functions using Regularized Least-Squares. [https://doi.org/10.31223/X5M71S](https://doi.org/10.31223/X5M71S)

---
