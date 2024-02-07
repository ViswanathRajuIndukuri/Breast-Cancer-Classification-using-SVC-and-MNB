# Breast Cancer Classification using Support Vector Classifier and Multinomial Naive Bayes

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org) 
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)

Breast Cancer Classification using Support Vector Classifier (SVC) and Multinomial Naive Bayes (MNB) models. This project aims to classify breast cancer tumors as malignant or benign based on various features extracted from fine needle aspirate (FNA) images.

## Dataset
The dataset used for this project is the popular [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)) obtained from the UCI Machine Learning Repository. It contains features computed from digitized images of breast mass FNA.

## Project Structure
- `data`: Folder containing the dataset (`breast-cancer-wisconsin.csv`).
- `breast_cancer_classification.ipynb`: Jupyter Notebook containing the code for data preprocessing, model building, hyperparameter tuning, and performance evaluation.
- `README.md`: This file, providing an overview of the project.

## Models and Performance
### Support Vector Classifier (SVC)
- Trained SVC models with default and tuned hyperparameters using all features and selected features.
- Tuned hyperparameters (C, gamma, kernel) using GridSearchCV.
- Best performing model: SVC with all features and tuned hyperparameters (98.24% accuracy).

### Multinomial Naive Bayes (MNB)
- Trained MNB models with default and tuned hyperparameters using all features and selected features.
- Tuned hyperparameter (alpha) using GridSearchCV.
- Best performing model: MNB with selected features and tuned hyperparameters (94.73% accuracy).

## Model Comparison
- Compared SVC and MNB models based on accuracy.
- Best performing SVC model: SVC with all features and tuned hyperparameters (98.24% accuracy).
- Best performing MNB model: MNB with selected features and tuned hyperparameters (94.73% accuracy).

## Conclusion
The Support Vector Classifier with all features and tuned hyperparameters emerged as the best-performing model for breast cancer classification in this project. However, Multinomial Naive Bayes also provided competitive results, especially with feature selection and hyperparameter tuning.
