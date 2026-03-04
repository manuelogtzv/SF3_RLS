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

- Gutierrez-Villanueva, M. O., B. D. Cornuelle, S. T. Gille, M. R. Mazloff, and D. Balwada, 2026: An improved methodology to estimate cross-scale kinetic energy transfers from third-order structure functions using regularized least-squares. J. Atmos. Oceanic Technol., , e250025, https://doi.org/10.1175/JTECH-D-25-0025.1, in press.
---
