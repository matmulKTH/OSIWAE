# OSIWAE: Online Sequential Importance-Weighted Autoencoder

This repository contains the code for the paper "Recursive Learning of Asymptotic Variational Objectives". It includes implementations of the **OSIWAE algorithm**, along with the associated proposal classes, as well as three key numerical simulations: `run_ovsmc`, `run_slam`, and `run_growth`. These simulations demonstrate the effectiveness of the OSIWAE method across different scenarios.

## Table of Contents
1. [Introduction](#introduction)
2. [Environment Setup](#environment-setup)
3. [Key Files](#key-files)


## Introduction
This project demonstrates the implementation of the **OSIWAE** method, an extension of the Importance Weighted Autoencoder (IWAE) for state-space models (SSMs). The code also includes the associated proposal classes used by the OSIWAE algorithm to improve proposal adaptation.

Three numerical experiments are provided to reproduce the results from the paper:
- **Multivariate Linear Gaussian (MVG) Model**: Comparison with OVSMC and RML for parameter estimation in a multivariate Gaussian state-space model.
- **SLAM**: Application to simultaneous localization and mapping.
- **Growth Model**: Application to a highly non-linear state-space model.

The primary goal of these experiments is to showcase that OSIWAE consistently outperforms both OVSMC and RML in a variety of use cases, improving parameter learning and proposal adaptation.

## Environment Setup

To set up the environment for running the OSIWAE simulations, follow these steps:

1. Use the provided `.yml` file to create the environment:
   ```bash
   conda env create -f osiwae_env.yml
2. Activate the envirement: 
    ```bash
    conda activate aistats


## Key Files

After setting up the environment, here’s an overview of the key files in the project:

- **`particle_algorithm.py`**: Class OSIWAE contains the main implementation of the OSIWAE algorithm. 
- **`proposals.py`**: Includes the different proposal models used in the algorithm.
- **`models.py`**: Contains the three models we simulate: Multivariate Linear Gaussian, SLAM, and Growth Model.
- **`extra.py`**: Implements the particle sampling algorithms used within the simulations.

Additionally, the following scripts are used to run the numerical experiments:
- **`run_slam.py`**: For running the SLAM numerical experiment.
- **`run_mvg.py`**: For running the Multivariate Linear Gaussian (MVG) numerical experiment.
- **`run_growth.py`**: For running the Growth Model numerical experiment.

