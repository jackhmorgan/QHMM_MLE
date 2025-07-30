# Code for "On Quantum and Quantum-Inspired Maximum Likelihood Estimation and Filtering of Stochastic Volatility Models"

This repository contains the code accompanying the [paper of the same name](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5274549) by Ghysels, Morgan, and Mohammadbagherpoor.

## Repository Structure

- **`HMM/`**  
  Contains the implementation of all Hidden Markov Model (HMM)-based approaches discussed in the paper.
  - **`QHMM_Example.ipynb`**  
    Demonstrates an example workflow using the HMM module, including model setup, training, and inference.

- **`causal_break_test.ipynb`**  
  Implements the causal break test used to generate **Figure 3** in the paper.

- **`SPY_training.py`**  
  Script for training the empirical model on SPY data.

- **`PC_to_NPC_to_QHMM_Final.py`**  
  Code for the primary empirical result shown in **Figure 4**. This includes the transition from classical to quantum-inspired models.

---

Please cite the paper if you use this code in your own research.
