## Algorithms with Calibrated Machine Learning Predictions

This repository contains code and experiments for the ICML paper [Algorithms with Calibrated Machine Learning Predictions](https://arxiv.org/abs/2502.02861).

## Abstract
The field of algorithms with predictions incorporates machine learning advice in the design of online algorithms to improve real-world performance. While this theoretical framework often assumes uniform reliability across all predictions, modern machine learning models can now provide instance-level uncertainty estimates. In this paper, we propose calibration as a principled and practical tool to bridge this gap, demonstrating the benefits of calibrated advice through two case studies: the ski rental and online job scheduling problems. For ski rental, we design an algorithm that achieves optimal prediction-dependent performance and prove that, in high-variance settings, calibrated advice offers more effective guidance than alternative methods for uncertainty quantification. For job scheduling, we demonstrate that using a calibrated predictor leads to significant performance improvements over existing methods. Evaluations on real-world data validate our theoretical findings, highlighting the practical impact of calibration for algorithms with predictions.

## Overview

This repository provides implementations for experiments in the paper. We use part of the [Citibike dataset](https://citibikenyc.com/system-data) for ski-rental and the [UCI Sepsis](https://archive.ics.uci.edu/dataset/827/sepsis+survival+minimal+clinical+records) dataset for scheduling and ski rental problems with predictions and calibration.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/heyyjudes/algs-cali-pred.git
   cd algs-cali-pred
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required libraries:**
   The main dependencies are listed in `requirements.txt`. Install them with:
   ```bash
   pip install -r requirements.txt
   ```

## Experiments

- **Ski Rental**: [ski_rental.ipynb](ski-rental.ipynb) Demonstrates algorithms for the ski rental problem using calibrated machine learning predictions (Section 3 of our paper).
- **Online Scheduling**: [scheduling.ipynb](scheduling.ipynb) Implements scheduling algorithms with calibrated predictions (Section 4 of our paper).

Both notebooks rely on several other files: 
- `calibration.py` contains classes for Histogram Calibration, Bin Calibration, and Platt Scaling. 
- `model.py` contains models we use for the base predictors. 
- `ski_rental.py` contains helper functions for calculating competitive ratio.  


## Citation

If you use this code or find it helpful, please cite our paper:
@article{SVW2025algorithms,
  title={Algorithms with calibrated machine learning predictions},
  author={Shen, Judy Hanwen and Vitercik, Ellen and Wikum, Anders},
  journal={arXiv preprint arXiv:2502.02861},
  year={2025}
}