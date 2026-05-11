# ml-vacuum-membrane-distillation
Machine learning models for predicting permeate flux in Vacuum Membrane Distillation using SVR and MLP regressors.
## Overview
This project focuses on the development of machine learning models for predicting permeate flux in Vacuum Membrane Distillation (VMD) systems used in desalination processes.
The study compares the performance of:
 Support Vector Regressor (SVR)
 Multilayer Perceptron (MLP) Regressor
## Objectives
 Apply supervised machine learning in chemical engineering applications
- Predict permeate flux using operational parameters
- Compare regression model performance
## Input Parameters
- Feed Temperature
- Permeate Side Pressure
- Feed Flow Rate
## Output Parameters
- Permeate Flux
- Specific Energy Consumption
## Machine Learning Techniques Used
### Support Vector Regressor (SVR)
- RBF Kernel
- GridSearchCV for hyperparameter tuning
- Cross-validation approach
### Multilayer Perceptron (MLP)
- ANN-based regression model
- Adam optimizer
- Early stopping implementation
## Results
### SVR Performance
- R² Score = 0.976647
- RMSE = 0.121482
- MAE = 0.099582
### MLP Performance
- R² Score = 0.970376
- RMSE = 0.136824
- MAE = 0.106236

## Repository Structure
ml-vacuum-membrane-distillation/
├── report/
├── code/
├── data/
└── results/
## Technologies Used
- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

## Applications
- Desalination process optimization
- AI in chemical engineering
- Process modeling and prediction

## Contributors
- Isha Joshi
- Ishaan Ghosh
- Muskan Sharma
- Puneet Kumar Jha
